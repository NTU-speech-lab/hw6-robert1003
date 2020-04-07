#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# import packages
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from _dataset import Adverdataset
from _utils import read_label
from _attack import fgsm

# hyperparams
args = {
    'device': 'cuda',
    'epsilon': 0.3,
    'input': sys.argv[1],
    'output': sys.argv[2],
    'model': 'densenet169',
    'valid': True,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
args = argparse.Namespace(**args)

# create dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(args.mean, args.std, inplace=False)
])
dataset = Adverdataset(args.input, label=None, transform=transform)
dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

# load pretrained model
if args.model == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.model == 'vgg19':
    model = models.vgg19(pretrained=True)
elif args.model == 'resnet50':
    model = models.resnet50(pretrained=True)
elif args.model == 'resnet101':
    model = models.resnet101(pretrained=True)
elif args.model == 'densenet121':
    model = models.densenet121(pretrained=True)
elif args.model == 'densenet169':
    model = models.densenet169(pretrained=True)
else:
    assert False

# attack!
ori_images, adv_images = fgsm(model, dataLoader, args.epsilon, args.device)
mean = np.array(args.mean).reshape(3, 1, 1)
std = np.array(args.std).reshape(3, 1, 1)
for i, x in enumerate(adv_images):
    x = x * std + mean
    x = torch.from_numpy(x).permute(1, 2, 0).numpy()
    adv_images[i] = np.clip(x, 0, 1)
for i, x in enumerate(ori_images):
    x = x * std + mean
    x = torch.from_numpy(x).permute(1, 2, 0).numpy()
    ori_images[i] = np.clip(x, 0, 1)

# output pictures
for image, fname in zip(adv_images, sorted(os.listdir(args.input))):
    plt.imsave(os.path.join(args.output, fname), image)

# valid
if args.valid:
    label, category = read_label('./data/labels.csv', './data/categories.csv')
    dif = 0
    tot = 0
    for adv_image, y in zip(adv_images, label):
        _, py = model(transform(Image.fromarray(np.uint8(adv_image * 255))).unsqueeze(0).to(args.device)).topk(1)
        tot += 1
        if py[0][0] != y:
            dif += 1

    print('succ rate:', dif / tot)

    L_inf = 0
    for ori_image, adv_image in zip(ori_images, adv_images):
        L_inf += np.abs(ori_image - adv_image).max() * 255
    print('L_inf:', L_inf / tot)
