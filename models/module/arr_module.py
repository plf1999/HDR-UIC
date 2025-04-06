import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ShortCut
from .quant_ops import quant_act_pams


class BitSelector(nn.Module):
    def __init__(self, n_feats, ema_epoch=1, search_space=[4,6,8]):
        super(BitSelector, self).__init__()
 
        self.quant_bit1 = quant_act_pams(k_bits=search_space[0], ema_epoch=ema_epoch)
        self.quant_bit2 = quant_act_pams(k_bits=search_space[1], ema_epoch=ema_epoch)
        self.quant_bit3 = quant_act_pams(k_bits=search_space[2], ema_epoch=ema_epoch)

        self.search_space =search_space
        
        self.net_small = nn.Sequential(
            nn.Linear(n_feats+3, len(search_space)) 
        )
        nn.init.ones_(self.net_small[0].weight)
        nn.init.zeros_(self.net_small[0].bias)
        nn.init.ones_(self.net_small[0].bias[-1])

    def forward(self, x):
        colorfulness = x[4]  
        weighted_bits = x[3]
        bits = x[2]
        grad = x[0] 
        x = x[1]
        
        layer_std_s = torch.std(x, (2,3)).detach()  
        
        x_embed = torch.cat([grad, layer_std_s, colorfulness], dim=1)
        
        bit_type = self.net_small(x_embed) 
        flag = torch.argmax(bit_type, dim=1)
        p = F.softmax(bit_type, dim=1)

        if len(self.search_space)== 3:
            p1 = p[:,0]
            p2 = p[:,1]
            p3 = p[:,2]
            bits_hard = (flag==0)*self.search_space[0] + (flag==1)*self.search_space[1] + (flag==2)*self.search_space[2]
            bits_soft = p1*self.search_space[0]+p2*self.search_space[1]+ p3*self.search_space[2]
            bits_out = bits_hard.detach() - bits_soft.detach() + bits_soft
            bits += bits_out
            weighted_bits += bits_out / (self.search_space[0]*p1.detach()+self.search_space[1]*p2.detach()+self.search_space[2]*p3.detach())

            q_bit1 = self.quant_bit1(x) 
            q_bit2 = self.quant_bit2(x)
            q_bit3 = self.quant_bit3(x)
            out_soft = p1.view(p1.size(0),1,1,1)*q_bit1 + p2.view(p2.size(0),1,1,1)*q_bit2 + p3.view(p3.size(0),1,1,1)*q_bit3
            out_hard = (flag==0).view(flag.size(0),1,1,1)*q_bit1 + (flag==1).view(flag.size(0),1,1,1)*q_bit2 + (flag==2).view(flag.size(0),1,1,1)*q_bit3
            residual = out_hard.detach() - out_soft.detach() + out_soft

        elif len(self.search_space)== 2:
            p1 = p[:,0]
            p2 = p[:,1]
            bits_hard = (flag==0)*self.search_space[0] + (flag==1)*self.search_space[1]
            bits_soft = p1*self.search_space[0]+p2*self.search_space[1]
            bits_out = bits_hard.detach() - bits_soft.detach() + bits_soft
            bits += bits_out
            weighted_bits += bits_out / (self.search_space[0]*p1.detach()+self.search_space[1]*p2.detach())
            q_bit1 =self.quant_bit1(x)
            q_bit2 = self.quant_bit2(x)
            out_soft = p1.view(p1.size(0),1,1,1)*q_bit1 + p2.view(p2.size(0),1,1,1)*q_bit2 
            out_hard = (flag==0).view(flag.size(0),1,1,1)*q_bit1 + (flag==1).view(flag.size(0),1,1,1)*q_bit2
            residual = out_hard.detach() - out_soft.detach() + out_soft
        
        return [grad, residual, bits, weighted_bits, colorfulness]



# class ResidualBlock_CADyQ(nn.Module):

class ARR(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, conv, act, kernel_size, res_scale, k_bits=32, bias=False, ema_epoch=1,search_space=[4,6,8],loss_kdf=False):
        super(ARR, self).__init__()
        self.bitsel1 = BitSelector(in_channels, ema_epoch=ema_epoch, search_space=search_space)

        self.body = nn.Sequential(
            conv(in_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=kernel_size, stride=1, padding=1),
            act,
            BitSelector(out_channels, ema_epoch=ema_epoch, search_space=search_space),
            conv(out_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=kernel_size, stride=1, padding=1),
        )
        self.loss_kdf= loss_kdf
        self.res_scale = res_scale

        self.quant_act3 = quant_act_pams(k_bits, ema_epoch=ema_epoch)
        self.shortcut = ShortCut()


    def forward(self, x):
        colorfulness = x[5] 
        weighted_bits = x[4]
        f = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]
        x = self.shortcut(x)
        grad,x,bits,weighted_bits,colorfulness= self.bitsel1([grad,x,bits,weighted_bits, colorfulness])  
        
        residual = x
        
        x = self.body[0:2](x)
                
        grad,x,bits,weighted_bits,colorfulness= self.body[2]([grad,x,bits,weighted_bits, colorfulness]) 
        
        out = self.body[3](x) 
        f1 = out
        out = out.mul(self.res_scale)
        out = self.quant_act3(out)
        out += residual 
        if self.loss_kdf:
            if f is None:
                f = f1.unsqueeze(0)
            else: 
                f = torch.cat([f, f1.unsqueeze(0)], dim=0)
        else:
            f = None

        return out
        # return [grad, out, bits, f, weighted_bits, colorfulness]
        

