import random

import albumentations
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from earlyexit import ExitBlock2D, ExitBlockAdaptive, ExitBlock3layer
from flamby.datasets.fed_isic2019 import FedIsic2019


class Baseline(nn.Module):
    """Baseline model
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    """

    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(Baseline, self).__init__()
        self.name = "baseline"
        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
        nftrs = self.base_model._fc.in_features
        print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, 8)

        self.exit_pos = [4, 7, 10, 14]
        drop_prob = 0.2
        self.dropout = nn.Dropout(p=drop_prob)
        
        i = 0
        flag = 0
        self.bns = []
        def dfs(layer):
            nonlocal i,flag
            if flag == 1:
                return
            name = str(layer)
            if name.startswith("BatchNorm"):
                self.bns.append(layer)
                i += 1
                if i == 49:
                    flag = 1
            elif "BatchNorm" in name:
                for l in layer.children():
                    dfs(l)
        dfs(self)
        # for layer in self.base_model.children():
        #     if name.startswith("BatchNorm"):
        #         self.bns.append(layer)

    def forward(self, image):
        # out = self.base_model(image)
        mself = self.base_model
        
        x = mself._swish(mself._bn0(mself._conv_stem(image)))

        # Blocks
        for idx, block in enumerate(mself._blocks):
            drop_connect_rate = mself._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(mself._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.exit_pos:
                x = self.dropout(x)

        # Head
        x = mself._swish(mself._bn1(mself._conv_head(x)))
        x = mself._avg_pooling(x)
        if mself._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = mself._dropout(x)
            x = mself._fc(x)
        out = x
        return out

    def getallfea(self, x):
        fealist = []
        i = 0
        fea = x
        flag = 0
        def dfs(layer):
            nonlocal fea,i,flag
            if flag == 1:
                return
            name = str(layer)
            if name.startswith("BatchNorm"):
                fealist.append(fea.clone().detach())
                i += 1
                if i == 49:
                    flag = 1
                fea = layer(fea)
            elif "BatchNorm" in name:
                for l in layer.children():
                    dfs(l)
            else:
                fea = layer(fea)
        dfs(self)
        return fealist


    def get_sel_fea(self, x, plan):
        return self.base_model.extract_features(x)


class BaselineEarlyExit(nn.Module):
    """EarlyExit model,
    Exit after each of the blocks
    """

    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(BaselineEarlyExit, self).__init__()
        self.name = "earlyexit"
        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
        nftrs = self.base_model._fc.in_features
        print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, 8)
        
        self.exit_pos = [4, 7, 10, 14] # now only first three
        self.c = [40, 80, 112, 192] 
        num_hidden = len(self.exit_pos)
        self.complexity_factor = 1.2
        hidden_sizes = [int(((self.complexity_factor ** 0.5) ** (num_hidden - idx)) * 1280) for idx in range(num_hidden)]
        
        # self.exit_hidden_sizes = [int(((self.complexity_factor ** 0.5) ** (num_hidden - idx)) * 512) for idx in range(num_hidden)]
        
        exit_blocks = []
        for idx in range(num_hidden):
            in_channel = self.c[idx]
            hidden = hidden_sizes[idx]

            # complexity_factor = 1.2
            # hidden =  int(complexity_factor ** ((15 - self.exit_pos[idx]) / 2.0 ) * 128)
            
            
            # dim = (15 - self.exit_pos[idx]) 
            # hidden = int(in_channel * dim * 2 / 3.0 + 2) # zhi qian xie cheng 2 le
            # # hidden = int(dim * dim * 2 / 3.0 + 8) * 128
            print(in_channel, hidden)

            # exit_blocks += [ExitBlock2D(in_channel, int(in_channel * 2 / 3.0 + 8), 8)]
            # exit_blocks += [ExitBlockAdaptive(in_channel, dim, hidden, 8)]
            # exit_blocks += [ExitBlock3layer(in_channel, dim, hidden, 8)]
            exit_blocks += [ExitBlock2D(in_channel, hidden, 8)]
        self.exit_blocks = nn.ModuleList(exit_blocks)
        # for key in self.state_dict().keys():
        #     print(key, self.state_dict()[key].get_device())
        # print("done")

    def forward(self, image):
        mself = self.base_model
        out_blocks = []
        
        x = mself._swish(mself._bn0(mself._conv_stem(image)))

        # Blocks
        for idx, block in enumerate(mself._blocks):
            drop_connect_rate = mself._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(mself._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.exit_pos:
                out_blocks += [x]
                # print(idx, x.flatten().shape)
        
        out_exits = []
        for out_block, exit_block in zip(out_blocks, self.exit_blocks):
            out = exit_block(out_block)
            out_exits += [out]

        # Head
        x = mself._swish(mself._bn1(mself._conv_head(x)))
        x = mself._avg_pooling(x)
        if mself._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = mself._dropout(x)
            x = mself._fc(x)
        
        out = torch.stack(out_exits + [x], dim=1)
        # print(out)
        return out

    def getallfea(self, x):
        fealist = []
        i = 0
        fea = x
        flag = 0
        def dfs(layer):
            nonlocal fea,i,flag
            if flag == 1:
                return
            name = str(layer)
            if name.startswith("BatchNorm"):
                fealist.append(fea.clone().detach())
                i += 1
                if i == 49:
                    flag = 1
                fea = layer(fea)
            elif "BatchNorm" in name:
                for l in layer.children():
                    dfs(l)
            else:
                fea = layer(fea)
        dfs(self)
        return fealist


    def get_sel_fea(self, x, plan):
        return self.base_model.extract_features(x)


if __name__ == "__main__":

    sz = 200
    train_aug = albumentations.Compose(
        [
            albumentations.RandomScale(0.07),
            albumentations.Rotate(50),
            albumentations.RandomBrightnessContrast(0.15, 0.1),
            albumentations.Flip(p=0.5),
            albumentations.Affine(shear=0.1),
            albumentations.RandomCrop(sz, sz),
            albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
            albumentations.Normalize(always_apply=True),
        ]
    )

    mydataset = FedIsic2019(0, True, "train", augmentations=train_aug)

    model = Baseline()

    for i in range(50):
        X = torch.unsqueeze(mydataset[i][0], 0)
        y = torch.unsqueeze(mydataset[i][1], 0)
        print(X.shape)
        print(y.shape)
        print(model(X))
