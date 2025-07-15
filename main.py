import argparse
import os
import numpy as np
import timm
import torch
import torchvision.models
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
from attacks import P2FA
from utils import imagenet_denormalize

parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=20, type=int, help='batch size')
parser.add_argument('--device', default=0, type=int, help='GPU device Id')
parser.add_argument('--seed', default=1234, type=int, help='random seed')
parser.add_argument('--eps', default=16 / 255, type=float, help='epsilon')
parser.add_argument('--steps', default=10, type=int, help='steps T')
parser.add_argument('--source', default='inc_v3', choices=['inc_v3', 'inc_res_v2', 'res_152', 'inc_v4'],
                    type=str, help='source model')

args = parser.parse_args()
DEVICE = f'cuda:{args.device}'
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

inc_v3 = torchvision.models.inception_v3(pretrained=True, transform_input=False, num_classes=1000).eval().to(DEVICE)
inc_v4 = timm.create_model('inception_v4', pretrained=True).eval().to(DEVICE)
inc_res_v2 = timm.create_model('inception_resnet_v2', pretrained=True).eval().to(DEVICE)
res_50 = torchvision.models.resnet50(pretrained=True).eval().to(DEVICE)
res_152 = torchvision.models.resnet152(pretrained=True).eval().to(DEVICE)
vgg_16 = torchvision.models.vgg16(pretrained=True).eval().to(DEVICE)
vgg_19 = torchvision.models.vgg19(pretrained=True).eval().to(DEVICE)

target_models = {
    'inc_v3': inc_v3,
    'inc_v4': inc_v4,
    'inc_res_v2': inc_res_v2,
    'res_50': res_50,
    'res_152': res_152,
    'vgg_16': vgg_16,
    'vgg_19': vgg_19,
}
source_model = target_models[args.source]

if args.source == 'inc_v3':
    layer_name = 'Mixed_5b'
elif args.source == 'inc_res_v2':
    layer_name = 'conv2d_4a'
elif args.source == 'res_152':
    layer_name = 'layer2.7.relu'
elif args.source == 'inc_v4':
    layer_name = 'features.6'


@torch.no_grad()
def main():
    global attacker
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.NIPS2017AdversarialCompetition(transforms)
    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    attacker = P2FA(source_model, eps=args.eps, steps=3, layer_name=layer_name, device=DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    acc = np.zeros((len(target_models),))
    asr1 = np.zeros((len(target_models),))
    asr2 = np.zeros((len(target_models),))

    for idx, (images, labels) in enumerate(data_loader):
        print(f'Batch: [{idx + 1}/{len(data_loader)}]')
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        # targets = F.one_hot(labels.type(torch.int64), 1000).float().to(DEVICE)
        with torch.enable_grad():
            adv = attacker(images, labels)

        for tm_idx, target_model in enumerate(target_models.values()):
            logits = target_model(images)
            logits_adv = target_model(adv)
            predict_labels = logits.argmax(dim=1)
            predict_adv_labels = logits_adv.argmax(dim=1)

            acc[tm_idx] += np.sum(np.array(predict_labels.detach().cpu() == labels.detach().cpu()).astype(int))
            asr1[tm_idx] += np.sum(np.array(predict_adv_labels.detach().cpu() != labels.detach().cpu()).astype(int))
            asr2[tm_idx] += np.sum((np.array(predict_labels.detach().cpu() == labels.detach().cpu()) & (
                np.array(predict_adv_labels.detach().cpu() != labels.detach().cpu()))).astype(int))

    asr1 /= len(dataset)
    asr2 /= acc
    acc /= len(dataset)

    for tmn_idx, target_model_name in enumerate(target_models.keys()):
        print(f'The ACC of {target_model_name}: {(100 * acc[tmn_idx]):>.1f}%, '
              f'The ASR1 of {target_model_name}: {(100 * asr1[tmn_idx]):>.1f}%, '
              f'The ASR2 of {target_model_name}: {(100 * asr2[tmn_idx]):>.1f}%')
    print(f'mACC: {(100 * np.mean(acc)):>.1f}%, '
          f'mASR1: {(100 * np.mean(asr1)):>.1f}%, '
          f'mASR2: {(100 * np.mean(asr2)):>.1f}%')

if __name__ == '__main__':
    main()