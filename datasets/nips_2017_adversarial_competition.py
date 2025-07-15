import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class NIPS2017AdversarialCompetition(Dataset):

    def __init__(self, transform=None):
        self.images_root_dir = r'datasets/NIPS2017_adversarial_competition/dev_dataset/my_images'
        self.real_labels = torch.load(r'datasets/NIPS2017_adversarial_competition/dev_dataset/real_labels.pt')
        self.transform = transform

    def __getitem__(self, item):
        filename = str(item) + '.png'
        label = self.real_labels[item]  # [0, 999]
        filepath = os.path.join(self.images_root_dir, filename)
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.real_labels.shape[0]
