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
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from _dataset import Adverdataset
from _utils import read_label
from _attack import fgsm

# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# hyperparams
args = {
    'device': 'cuda',
    'epsilon': 0.1,
    'input': sys.argv[1],
    'output': sys.argv[2],
    'model': 'densenet121',
    'valid': True,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
args = argparse.Namespace(**args)

# create dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=args.mean, std=args.std, inplace=False)
])
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1 / i for i in args.std]),
    transforms.Normalize(mean=[-i for i in args.mean], std=[1, 1, 1]),
    transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
    transforms.ToPILImage()
])
df, label_name = read_label(
    os.path.join(args.input, 'labels.csv'),
    os.path.join(args.input, 'categories.csv')
)
dataset = Adverdataset(os.path.join(args.input, 'images'), label=df, transform=transform)
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
for i, x in enumerate(adv_images):
    adv_images[i] = inv_transform(x)
for i, x in enumerate(ori_images):
    ori_images[i] = inv_transform(x)

# output pictures
imgIter = tqdm(zip(adv_images, sorted(os.listdir(os.path.join(args.input, 'images')))), desc='[*] Saving')
for image, fname in imgIter:
    image.save(os.path.join(args.output, fname))