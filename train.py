import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from models.resnet import *
from torchvision.datasets import ImageFolder
import time
from torch.optim import lr_scheduler
from eval_lfw import evaluationAPI

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])


transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.RandomHorizontalFlip(),
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

MAX_EPOCH = 50
BATCH_SIZE = 512
NUM_CLASSES = 10575
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/data2/datasets/CASIA/imgs'
loss_fn = nn.CrossEntropyLoss().to(device)
dataset = datasets.ImageFolder(data_dir, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

model = ResNet(NUM_CLASSES)
model = nn.DataParallel(model).to(device)


#redefine optimizers:
optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
scheduler = lr_scheduler.MultiStepLR(optim, milestones=[20, 35, 45], gamma=0.1)
model.train()
#model.classifier.classifier = nn.Sequential()
for epoch in range(MAX_EPOCH):
    if epoch > 0:
        evaluationAPI(epoch)
    #st = time.time()
    scheduler.step()
    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        result = model(data, target)
        loss = loss_fn(result,target)
        _, prediction = torch.max(result, 1)
        precision = torch.mean((prediction==target).float())
        if idx%1==0:
            print('epoch %s iteration %s/%s:'%(epoch+1,idx,len(dataloader)),end=' ')
            print('loss=%.3f, precision=%.6f'%(loss.item(), precision.item()))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if idx % 400 == 0:
            save_path = '/data2/snapshot/epoch_%s_iteration_%s.pth'%(epoch+1,idx)
            torch.save(model.state_dict(), save_path)
    #ed = time.time()
    #print('One traversal over the dataset takes %.4f secs.'%(ed-st))

