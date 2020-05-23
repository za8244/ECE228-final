import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np
import code

def default_loader(path):
    return Image.open(path).convert('RGB')

class INAT(data.Dataset):
    def __init__(self, root, is_train=True):

        nClass = 234
        nImage = 0
        self.imageNames = []
        self.imageClasses = []
        for i in range(4030, 4264):
            imageNames = os.listdir(root+ "/" + str(i))
            print(imageNames)
            for imageName in imageNames:
                path = root+ "/" + str(i) + "/" + imageName
                self.imageNames.append(path)
                print("path", path)
            self.imageClasses.append(i)
        
        nImage = len(self.imageNames)
        print(self.imageNames)
        print(nImage)
        # code.interact(local=locals())
        if is_train:
            self.imageNames = self.imageNames[:int(nImage*0.8)]
            self.imageClasses = self.imageClasses[:int(nImage*0.8)]
        else:
            self.imageNames = self.imageNames[nImage*0.8:]
            self.imageClasses = self.imageClasses[nImage*0.8:]

        print("nImage", nImage)
        print("nClass", 4264-4030+1)

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        self.im_size = [299, 299]  # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):
        path = self.imageNames[index]
        img = self.loader(path)

        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
        else:
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        return img, self.imageClasses[index]

    def __len__(self):
        return len(self.imageNames)

loader = INAT("Mammalia", 1) 
print(loader.__getitem__(1))
print(loader.__getitem__(3))
print(loader.__getitem__(4))
print(loader.__getitem__(9))