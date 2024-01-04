"""
Neighborhood Attention Transformer.
https://arxiv.org/abs/2204.07143

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torchvision
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, appearance_layers=[0,1,2,3]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        x = input
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in appearance_layers:
                feats.append(x)
                
        return  feats
    

class DINOv2(torch.nn.Module):
    def __init__(self, resize=True, size=224, model_type='dinov2_vitl14'):
        super(DINOv2, self).__init__()
        self.size=size
        self.resize = resize
        self.transform = torch.nn.functional.interpolate
        self.model = torch.hub.load('facebookresearch/dinov2', model_type)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, appearance_layers=[1,2]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        if self.resize:
            input = self.transform(input, mode='bicubic', size=(self.size, self.size), align_corners=False)
        # mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(input.device)
        # std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(input.device)
        input = (input-self.mean) / self.std
        feats = self.model.get_intermediate_layers(input, self.model.n_blocks, reshape=True)
        feats = [f.detach() for f in feats]

        return feats