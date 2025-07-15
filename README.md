# P2FA
This repository contains the PyTorch code for the paper: [Pixel2Feature Attack (P2FA): Rethinking the Perturbed Space to Enhance Adversarial Transferability](https://openreview.net/forum?id=bPJo5uSkOJ)(ICML 2025)
## Datasets
1000 images for the NIPS 2017 adversarial competition.
```
directory/
├── 0.png
├── 1.png
├── ...
└── 999.jpg
```
normalization:
```
import torchvision.transforms as T

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
## Pre-trained Models
| Models                                        | Packages | Accuracy (%) | 
|-----------------------------------------------|----------|--------------|
| Inception-v3 (Inc-v3)                         | Pytorch  | 95.4         |
| Inception-v4 (Inc-v4)                         | timm     | 94.8         |
| Inception-ResNet-v2 (IncRes-v2  )             | timm     | 97.2         | 
| ResNet-50 (Res-50)                            | Pytorch  | 91.8         |
| ResNet-152 (Res-152)                          | Pytorch  | 93.6         |
| VGG-16 (Vgg-16)                               | Pytorch  | 85.5         |
| VGG-19 (Vgg-19)                               | Pytorch  | 87.5         |
| Adv-Inception-v3 (Adv-Inc-v3)                 | timm     | 86.8         |
| Ens-Adv-Inception-ResNet-v2 (Ens-IncRes-v2)   | timm     | 94.5         |
## P2FA

> `python main.py `

## Citation
If you find the idea or code useful for your research, please consider citing our paper:
```
@inproceedings{liupixel2feature,
  title={Pixel2Feature Attack (P2FA): Rethinking the Perturbed Space to Enhance Adversarial Transferability},
  author={Liu, Renpu and Wu, Hao and Zhang, Jiawei and Cheng, Xin and Luo, Xiangyang and Ma, Bin and Wang, Jinwei},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
