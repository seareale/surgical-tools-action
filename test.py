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

from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('label', default=11, type=int, help='set label')
    parser.add_argument('pth', help='set checkpoint')
    parser.add_argument('--size', default=224, type=int, help='input size')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    args = parser.parse_args()
    return args


args = parse_args() 
label_idx = args.label
cp_path = args.pth
input_size = args.size
batch_size = args.batch


os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = 'cuda:0'

data_dir_base = './dataset/'
data_dir_add = ['11', '12', '13', '14', '15', '17']

transform_val = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_dataset = datasets.ImageFolder(data_dir_base + str(label_idx) + '/val', transform = transform_val)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def val(epoch, model, val_dataloader, criterion):
    global best_acc
    global best_model_wts
    
    model.eval()
    val_loss = 0
    total = 0
    correct = 0
    
    mat_count = 0
    
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
            
            mat = confusion_matrix(labels.cpu(), preds.cpu())
            mat_count += mat
            
            pbar.set_postfix({'loss': '%.4f' % (val_loss/(batch_idx+1)), 'acc': '%.4f' % (correct/total)})
            
        print('Val Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'%(epoch, val_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return mat_count


model = models.resnet101(pretrained=True)
model.fc = nn.Sequential(nn.Linear(2048, 2048),
                             nn.BatchNorm1d(num_features=2048),
                             nn.ReLU(),
                             nn.Linear(2048, 1024),
                             nn.BatchNorm1d(num_features=1024),
                             nn.ReLU(),
                             nn.Linear(1024, 2))
model.load_state_dict(torch.load(cp_path))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

since = time.time()
mat = val(0, model, val_dataloader, criterion)
time_elapsed = time.time() - since
print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('----------------------------------------------')
print('\n', mat)