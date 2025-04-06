import argparse
import math
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader

from models.model import HDRUIC
from utils.utils import *

from torch.utils.tensorboard import SummaryWriter   
import os

from dataset import SIG17_Training_Dataset,SIG17_Validation_Dataset, SIG17_Test_Dataset
from tqdm import tqdm

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        device = input.device 
        self.mean = self.mean.to(device)  
        self.std = self.std.to(device)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        
        for i, block in enumerate(self.blocks):
            block = block.to(device)
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)  
        mu_label = range_compressor(label, self.mu)
        return nn.L1Loss()(mu_pred, mu_label)
    
    
class JointReconPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super(JointReconPerceptualLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_vgg = VGGPerceptualLoss(False)

    def forward(self, input, target):
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(input, target)
        loss_vgg = self.loss_vgg(input_mu, target_mu)
        loss = loss_recon + self.alpha * loss_vgg
        return loss, loss_recon, loss_vgg



class HDRRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, alpha=0.01):
        super().__init__()
        self.lmbda = lmbda
        self.l1muloss = L1MuLoss()
        self.loss_vgg = VGGPerceptualLoss(False)
        self.distortion_loss = JointReconPerceptualLoss(alpha)
        self.alpha = alpha  
        

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["distortion_loss"], out["L1muloss"], out["vgg_loss"] = self.distortion_loss(output["x_hat"], target)
        out["loss"] = self.lmbda * out["distortion_loss"] + out["bpp_loss"]
        return out



class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, writer, type='mse'
):
    model.train()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpploss = AverageMeter()
    l1loss = AverageMeter()
    vggloss = AverageMeter()
    distortionloss = AverageMeter()
    
    for i, batch_data in enumerate(train_dataloader):
        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), batch_data['input2'].to(device)
        label = batch_data['label'].to(device)
        
        ldr0,ldr1,ldr2 = batch_data['ldr0'].to(device), batch_data['ldr1'].to(device), batch_data['ldr2'].to(device) 
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous(), ldr0, ldr1, ldr2)
        
        out_criterion = criterion(out_net, label)
        
        writer.add_scalar('Train loss', out_criterion["loss"].item(), epoch*len(train_dataloader)+i) 
        writer.add_scalar('Train bpp', out_criterion["bpp_loss"].item(), epoch*len(train_dataloader)+i) 
        writer.add_scalar('Train distortion', out_criterion["distortion_loss"].item(), epoch*len(train_dataloader)+i) 
        
        
        loss.update(out_criterion["loss"])
        bpploss.update(out_criterion["bpp_loss"])
        l1loss.update(out_criterion["L1muloss"])
        vggloss.update(out_criterion["vgg_loss"])
        distortionloss.update(out_criterion["distortion_loss"])
        
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tL1mu loss: {out_criterion["L1muloss"].item():.3f} |'
                f'\tdistortion loss: {out_criterion["distortion_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
            
    writer.add_scalars('epoch_loss', {'train': loss.avg}, epoch)
    writer.add_scalars('epoch_l1_loss', {'train': l1loss.avg}, epoch)
    writer.add_scalars('epoch_bpp_loss', {'train': bpploss.avg}, epoch)
    writer.add_scalars('epoch_vgg_loss', {'train': vggloss.avg}, epoch)
    writer.add_scalars('epoch_distortion_loss', {'train': distortionloss.avg}, epoch)
    

def test_epoch(epoch, test_dataloader, model, criterion, writer):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    l1_loss = AverageMeter()
    vgg_loss = AverageMeter()
    distortion_loss = AverageMeter()
    aux_loss = AverageMeter()
    val_psnr = AverageMeter()
    val_mu_psnr = AverageMeter()

    with torch.no_grad():
        for batch_data in test_dataloader:
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), batch_data['input2'].to(device)
            label = batch_data['label'].to(device)
            
            ldr0,ldr1,ldr2 = batch_data['ldr0'].to(device), batch_data['ldr1'].to(device), batch_data['ldr2'].to(device)
            out_net = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous(), ldr0, ldr1, ldr2)
            
            pred_hdr = torch.squeeze(out_net["x_hat"].detach().cpu()).numpy().astype(np.float32)
            pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1] 
                

            xhat = out_net["x_hat"].clamp_(0,1)   
            
            input_mu = range_compressor(xhat, mu=5000)
            target_mu = range_compressor(label, mu=5000)
            input_mu = input_mu.detach().cpu().numpy()
            target_mu = target_mu.detach().cpu().numpy()
 
            psnr = batch_psnr(xhat, label, 1.0)
            mu_psnr = batch_psnr_mu(xhat, label, 1.0) 
            val_psnr.update(psnr.item())
            val_mu_psnr.update(mu_psnr.item())
            
 
            out_criterion = criterion(out_net, label)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            l1_loss.update(out_criterion["L1muloss"])
            vgg_loss.update(out_criterion["vgg_loss"])
            distortion_loss.update(out_criterion["distortion_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tdistortion Loss: {distortion_loss.avg:.3f} |"
        f"\tl1 loss: {l1_loss.avg:.3f} |"
        f"\tvgg loss: {vgg_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    
    writer.add_scalar('test_uPSNR', val_mu_psnr.avg, epoch)
    writer.add_scalar('test_PSNR', val_psnr.avg, epoch)
    writer.add_scalar('test_loss', loss.avg, epoch)
    
    return val_mu_psnr.avg, val_psnr.avg, bpp_loss.avg


def test_single_img(model, img_dataset, device):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=1, shuffle=False)
    model.eval()
    per_img_bpp = AverageMeter() 
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader)):
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), \
                                                 batch_data['input1'].to(device), \
                                                 batch_data['input2'].to(device)
            output = model(batch_ldr0, batch_ldr1, batch_ldr2) 
            N, _, H, W = batch_ldr0.size() 
            num_pixels = N * H * W
            patch_bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )  
            per_img_bpp.update(patch_bpp)
            
            img_dataset.update_result(torch.squeeze(output['x_hat'].detach().cpu()).numpy().astype(np.float32))
    
    pred, label = img_dataset.rebuild_result() 
    return pred, label, per_img_bpp.avg


def save_checkpoint(state, save_path):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=100,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    parser.add_argument(
        "--gpu",
        default='0',
        help="select GPU for train",
    )
    parser.add_argument("--dataset_dir", type=str, default='./dataset',help='dataset directory')
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")


    train_dataset = SIG17_Training_Dataset(root_dir=args.dataset_dir, sub_set="sig17_trainset", is_training=True)
    
    test_dataset = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=True, crop_size=256)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = "cuda"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    
    upscale = 4
    window_size = 8
    height = (256 // upscale // window_size + 1) * window_size
    width = (256 // upscale // window_size + 1) * window_size
    
    net = HDRUIC(upscale=2, img_size=(height, width), in_chans=18,
                  window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6],
                  embed_dim=60, num_heads=[6, 6, 6, 6, 6, 6, 6, 6], mlp_ratio=2)  
      
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    criterion = HDRRateDistortionLoss(lmbda=args.lmbda)  
    last_epoch = 0

    dictory = {}
    if args.checkpoint: 
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
        
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        
    cur_psnr = [-1.0]
    
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer,
            type
        )
        
        mu_psnr, psnr, bpp = test_epoch(epoch, test_dataloader, net, criterion, writer)
        
        if mu_psnr > cur_psnr[0]:
            state = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            
            torch.save(state, save_path + "best_checkpoint.pth.tar")
            print('save best_checkpoint')
            cur_psnr[0] = mu_psnr
            
            with open(os.path.join(save_path, 'best_checkpoint.json'), 'w') as f:
                f.write('best epoch:' + str(epoch) + '\n')
                f.write('Validation set: Average mu_PSNR: {:.4f}, PSNR: {:.4f}, bpp: {:.4f}\n'.format(
                    mu_psnr,
                    psnr,
                    bpp
                ))
                
        lr_scheduler.step()
            
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            save_path,
        )            
            


if __name__ == "__main__":
    main(sys.argv[1:])
