import torch
from attacks import Attack
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, Tensor
from utils import imagenet_denormalize, imagenet_normalize

class P2FA(Attack):

    def __init__(self, model, eps=16 / 255, steps=10, decay=1.0, ensemble_number=30, eta=28.0,
                 layer_name='Mixed_5b', device='cuda:0'):
        """
        :param model: DNN model
        :param eps: the maximum perturbation
        :param steps: the number of iterations
        :param decay: the decay factor
        :param device: gpu device
        """
        super().__init__("P2FA", model, device)
        self.eps = T.Normalize(mean=[0., 0., 0.], std=[0.229, 0.224, 0.225])(
            torch.tensor(eps, device=device).expand((1, 3, 1, 1)))
        self.steps = steps
        self.alpha = self.eps / self.steps
        self.decay = decay
        self.ensemble_number = ensemble_number
        self.layer_name = layer_name
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.feature_maps = None
        self.register_hook()
        self.normalize = imagenet_normalize
        self.denormalize = imagenet_denormalize
        self.loss_fn_ce = nn.CrossEntropyLoss()
        self.eta = eta

    def hook(self, module, input, output):
        self.feature_maps = output
        return None

    def register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook=self.hook)

    def get_maskgrad(self, images: Tensor, labels: Tensor) -> Tensor:
        images = images.clone().detach()
        images.requires_grad = True
        logits = self.model(images)
        labels = labels.long()
        loss = self.loss_fn_ce(logits, labels)
        maskgrad = torch.autograd.grad(loss, images)[0]
        maskgrad /= torch.sqrt(torch.sum(torch.square(maskgrad), dim=(1, 2, 3), keepdim=True))
        return maskgrad.detach()

    def get_aggregate_gradient(self, images: Tensor, labels: Tensor) -> Tensor:
        _ = self.model(images)
        images_denorm = self.denormalize(images)
        images_masked = images.clone().detach()
        aggregate_grad = torch.zeros_like(self.feature_maps)
        targets = F.one_hot(labels.type(torch.int64), 1000).float().to(self.device)
        for _ in range(self.ensemble_number):
            g = self.get_maskgrad(images_masked, labels)
            # get fitted image
            images_masked = self.normalize(images_denorm + self.eta * g)
            logits = self.model(images_masked)
            loss = torch.sum(logits * targets, dim=1).mean()
            aggregate_grad += torch.autograd.grad(loss, self.feature_maps)[0]
        aggregate_grad /= torch.sqrt(torch.sum(torch.square(aggregate_grad), dim=(1, 2, 3), keepdim=True))
        return -aggregate_grad

    def forward(self, images, labels):
        box_min = imagenet_normalize(torch.zeros_like(images))
        box_max = imagenet_normalize(torch.ones_like(images))
        box_min = torch.clamp(images - self.eps, min=box_min)
        box_max = torch.clamp(images + self.eps, max=box_max)
        adv = images.clone().detach()
        gg = torch.zeros_like(adv)
        _ = self.model(images)
        g = torch.zeros_like(self.feature_maps)
        for _ in range(self.steps):
            aggregate_grad = self.get_aggregate_gradient(adv, labels)
            _ = self.model(adv)
            feature_maps = self.feature_maps.clone()
            g = self.decay * g + aggregate_grad
            feature_maps += 100000.0 * g / torch.sqrt(torch.sum(torch.square(g), dim=(1, 2, 3), keepdim=True))
            for _ in range(10):
                adv.requires_grad = True
                _ = self.model(adv)
                loss = torch.sum(torch.square(self.feature_maps - feature_maps), dim=(1, 2, 3)).mean()
                grad = torch.autograd.grad(loss, adv)[0]
                gg = self.decay * gg + grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                adv = torch.clamp(adv - self.alpha * torch.sign(gg), min=box_min, max=box_max).detach()
        return adv