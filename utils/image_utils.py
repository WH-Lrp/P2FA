import torchvision.transforms as T

def imagenet_normalize(x):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(x)

def imagenet_denormalize(x):
    denormalize = T.Normalize(mean=[-2.1179, -2.0357, -1.8044], std=[4.3668, 4.4643, 4.4444])
    return denormalize(x)
