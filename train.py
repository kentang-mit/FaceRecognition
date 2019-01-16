import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from models.resnet import *
from torchvision.datasets import ImageFolder
import time


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        normalize,
    ])

'''
val_transform = transforms.Compose([
        transforms.Resize((288,144)),
        transforms.ToTensor(),
        normalize
    ])
'''

MAX_EPOCH = 20
BATCH_SIZE = 128
NUM_CLASSES = 10575

data_dir = '/home/kentang/casia-maxpy-clean/CASIA-maxpy-clean'
loss_fn = nn.CrossEntropyLoss()
dataset = datasets.ImageFolder(data_dir, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

model = ResNet(NUM_CLASSES).cuda()

#redefine optimizers:
optim = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9,0.999))

#model.classifier.classifier = nn.Sequential()
for epoch in range(MAX_EPOCH):
    #st = time.time()
    for idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        result = model(data)
        loss = loss_fn(result,target)
        prediction = result.max(1)
        precision = torch.sum(result==target)/float(BATCH_SIZE)
        if idx%100==0:
            print('iteration %s:'%idx,end=' ')
            print('loss=%.3f, precision=%.6f'%loss.item(), precision.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        if idx % 1000 == 0:
            save_path = './snapshot/model_iteration_%s.pth'%idx
            torch.save(model.state_dict(), save_path)
    #ed = time.time()
    #print('One traversal over the dataset takes %.4f secs.'%(ed-st))

