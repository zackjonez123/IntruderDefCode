"""
******* This is the Neural Network Prototype **********

"""
import os
#from tarfile import data_filter
#import matplotlib.pyplot as plt
import numpy as np

#from sklearn.base import accuracy_score
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.optim.lr_scheduler import StepLR

from image_capture import grayscale
from PIL import Image
from torch.utils.data import DataLoader
from config import testarg, trainarg
# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
epochs = 4 # can increase for more accuracy
batch_size = 4
learning_rate = 0.001

#transform images into tensors
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#load datasets
# train_dataset = 
# test_dataset = 
# train_loader = 
# test_loader = 

classes = ('Friendly', 'Hostile')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1) # images are only 0 and 255, so input and ouput channels are 2, 3x3 is a common kernel size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # input, output, and kernel size are subject to change
        #self.drop1 = nn.Dropout(0.25) # probability of an element being zeroed
        #self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16*5*5, 100) # input and output
        self.fc2 = nn.Linear(100, 84) # 100 and 84 can be changed
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        #x = x.view(-1, 16*5*5) # another way to flatten, supposed to get the correct size
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.drop2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x



# Train data
# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     total_steps = len(train_loader)
#     for epoch in range(epochs):
#         for i, (images, labels) in enumerate(train_loader):
#             # inputs layer: 3 input channels, 6 output channels, 5x5 kernel size
#             images = images.to(device)
#             labels = labels.to(device)

#             # forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             # back pass
#             optimizer.zerograd()
#             loss.backward()
#             optimizer.step()

#             if (i+1) % 2000 == 0:
#                 print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

#     print('Finished Training')
#     PATH = './cnn.pth'
#     torch.save(model.state_dict(), PATH)


# Test Data
# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

def grayscaleloader(path:str)->Image.Image:
    with open(path, 'rb') as file:
        pic = Image.open(file).convert('L')
    return pic

def main():
    
    trainkwargs = {'batch_size':10, 'shuffle':True}
    testkwargs = {'batch_size':10, 'shuffle':True}
    path = os.path.join('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics')
    full_data = datasets.ImageFolder(root=path, transform=transforms.ToTensor(), loader=grayscaleloader)
    print('Subfolder int assignment', full_data.class_to_idx)
    traindata, testdata = torch.utils.data.random_split(full_data, [0.8, 0.2]) # 80% train, 20% test
    loadtrain = DataLoader(traindata, **trainkwargs)
    loadtest = DataLoader(testdata, **testkwargs)
    for batchnum, (pic, label) in enumerate(loadtrain):
        print('batch #', batchnum, 'pic shape', pic.shape, 'label shape', label.shape)

    model = Net().to(device)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

if __name__ == '__main__':
    main()
