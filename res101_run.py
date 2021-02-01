import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, models, transforms

import time
import os

from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('gpu', default=6, help='set gpu')
    parser.add_argument('label', default=11, type=int, help='set label')
    parser.add_argument('--epoch', default=10, type=int, help='epoch')
    parser.add_argument('--lrate', default=0.1, type=float, help='lrate')
    parser.add_argument('--size', default=128, type=int, help='input size')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    args = parser.parse_args()
    return args


args = parse_args() 
gpu_idx = args.gpu
label_idx = args.label
epoch_num = args.epoch
learning_rate = args.lrate
input_size = args.size
batch_size = args.batch


os.environ["CUDA_VISIBLE_DEVICES"]=gpu_idx
device = 'cuda:0'

data_dir_base = './dataset/'
data_dir_add = ['11', '12', '13', '14', '15', '17']

transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=16),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
transform_val = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_dataset = datasets.ImageFolder(data_dir_base + str(label_idx) + '/train', transform = transform_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = datasets.ImageFolder(data_dir_base + str(label_idx) + '/val', transform = transform_val)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def train(epoch, model, train_dataloader, criterion, optimizer, scheduler):
    print('\n Epoch: %d' % epoch)
    
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    
    
    pbar = tqdm(train_dataloader)
    for batch_idx, batch in enumerate(pbar):
        inputs, labels = batch
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = outputs.max(1)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total += labels.size(0)
        train_loss += loss.item()
        correct += preds.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': '%.4f' % (train_loss/(batch_idx+1)), 'acc': '%.4f' % (correct/total)})
        
    print('Train Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'%(epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def val(epoch, model, val_dataloader, criterion):
    global best_acc
    global best_model_wts
    global date_str
    global file_name
    
    model.eval()
    val_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        pbar = tqdm(val_dataloader)
        for batch_idx, batch in enumerate(pbar): 
            inputs, labels = batch
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            loss = criterion(outputs, labels)
            total += labels.size(0)
            val_loss += loss.item()
            correct += preds.eq(labels).sum().item()

            pbar.set_postfix({'loss': '%.4f' % (val_loss/(batch_idx+1)), 'acc': '%.4f' % (correct/total)})
            
        if correct/total > best_acc:
            best_acc = correct/total
            best_model_wts = model.state_dict()

            if not os.path.exists('./save'):
                os.makedirs('./save')
            torch.save(best_model_wts, file_name)

        print('Val Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'%(epoch, val_loss/(batch_idx+1), 100.*correct/total, correct, total))

###########################################################################################

model = models.resnet101(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
model.fc = nn.Sequential(nn.Linear(2048, 2048),
                             nn.BatchNorm1d(num_features=2048),
                             nn.ReLU(),
                             nn.Linear(2048, 1024),
                             nn.BatchNorm1d(num_features=1024),
                             nn.ReLU(),
                             nn.Linear(1024, 2))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=10e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


best_acc = 0
best_model_wts = model.state_dict()

date_str = "%02d"%time.localtime().tm_mon + str(time.localtime().tm_mday) + str(time.localtime().tm_hour) + '%02d' %(time.localtime().tm_min)
file_name = './save/res101_' + str(label_idx) + '_' + date_str + '.pth'

since = time.time()

for epoch in range(epoch_num):
    train(epoch, model, train_dataloader, criterion, optimizer, scheduler)
    val(epoch, model, val_dataloader, criterion)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

file_name_acc = './save/res101_' + str(label_idx) + '_' + date_str + '_' + ('%.2f' % best_acc)[2:] + '.pth'
os.rename(file_name, file_name_acc)