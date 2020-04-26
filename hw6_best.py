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
from tqdm import tqdm, trange
from PIL import Image
from _dataset import Adverdataset
from _utils import read_label
from _attack import deepfool

# hyperparams
args = {
    'device': 'cuda',
    'epsilon': 1.00001 / 255.0 / 0.229,
    'max_iter': 50,
    'overshoot': 0.01,
    'num_classes': 5,
    'input': sys.argv[1],
    'output': sys.argv[2],
    'start': int(sys.argv[3]),
    'end': int(sys.argv[4]),
    'model': 'densenet121',
    'valid': True,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
args = argparse.Namespace(**args)

# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# define transform
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

# read data
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
ori_images, adv_images = deepfool(model, dataLoader, transform, inv_transform, args.start, args.end, args.epsilon, args.max_iter, args.overshoot, args.num_classes, args.device)

imgIter = tqdm(zip(adv_images, sorted(os.listdir(os.path.join(args.input, 'images')))[args.start:args.end+1]), desc='[*] Saving')
for image, fname in imgIter:
    image.save(os.path.join(args.output, fname))
