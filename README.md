# Breaking Boundaries: Unifying Imaging and Compression for HDR Image Compression
Xuelin Shen, Linfeng Pan, Zhangkai Ni, Yulin He, Wenhan Yang, Shiqi Wang, Sam Kwong \
This repository provides the official implementation for the paper "Breaking Boundaries: Unifying Imaging and Compression for HDR Image Compression," IEEE Transactions on Image Processing, vol. 34, pp. 510-521, 2025. [Paper](https://ieeexplore.ieee.org/abstract/document/10841962)

## Abstract
High Dynamic Range (HDR) images present unique challenges for Learned Image Compression (LIC) due to their complex domain distribution compared to Low Dynamic Range (LDR) images. In coding practice, HDR-oriented LIC typically adopts preprocessing steps (e.g., perceptual quantization and tone mapping operation) to align the distributions between LDR and HDR images, which inevitably comes at the expense of perceptual quality. To address this challenge, we rethink the HDR imaging process which involves fusing multiple exposure LDR images to create an HDR image and propose a novel HDR image compression paradigm, Unifying Imaging and Compression (HDR-UIC). The key innovation lies in establishing a seamless pipeline from image capture to delivery and enabling end-to-end training and optimization. Specifically, a Mixture-ATtention (MAT)-based compression backbone merges LDR features while simultaneously generating a compact representation. Meanwhile, the Reference-guided Misalignment-aware feature Enhancement (RME) module mitigates ghosting artifacts caused by misalignment in the LDR branches, maintaining fidelity without introducing additional information. Furthermore, we introduce an Appearance Redundancy Removal (ARR) module to optimize coding resource allocation among LDR features, thereby enhancing the final HDR compression performance. Extensive experimental results demonstrate the efficacy of our approach, showing significant improvements over existing state-of-the-art HDR compression schemes.

## Environment
Python 3.8.18 \
PyTorch 1.13.0 \
CompressAI 1.2.4 

## Dataset
The datasets evaluated in our paper are listed below, please download it at the following link: \
[Kalantari et al.](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) \
[Tel et al.](https://drive.google.com/drive/folders/1CtvUxgFRkS56do_Hea2QC7ztzglGfrlB)

## Running the model
### Training
```
python train.py
```

### Testing
```
python test.py --checkpoint /path/to/checkpoint
```
