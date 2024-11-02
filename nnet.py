"""
******* This is the Neural Network Prototype **********

"""
import cv2
import crop # My crop.py module
import os
#from tarfile import data_filter
#import matplotlib.pyplot as plt
import numpy as np

#from sklearn.base import accuracy_score
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.optim import sgd
from torch.optim.lr_scheduler import StepLR

from image_capture import grayscale
from PIL import Image
from torch.utils.data import DataLoader
from config import testarg, trainarg

# Calculates the number of outputs given input channels and kernel/filter size
def calc(i, k, s):
    o = ((i-k) / s) + 1
    return o

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 239, kernel_size=4, stride=2) # calc(480, 4, 2) == 239
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(calc(480, 4, 3), 401, kernel_size=4, stride=3) # 441 - kernel + 1 = 401 outputs
        #self.drop1 = nn.Dropout(0.25) # probability of an element being zeroed
        #self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(119*119*239, 100) # torch.Size([10, 239, 119, 119]) Pool
        self.fc2 = nn.Linear(100, 3) # 100 and 84 can be changed
        #self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape, "Conv1")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(x.shape, "Pool")
        #x = self.drop1(x)
        #x = self.conv2(x)
        #x = F.relu(x)
        #x = self.pool2(x)
        x = torch.flatten(x, 1)
        #print(x.shape, "Linear input")
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.drop2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        #x = self.fc3(x)
        return output

# Train data
def train(model, device, loadtrain, optimizer, epochs):
    model.train()
    total_steps = len(loadtrain)
    #for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(loadtrain):
                # inputs layer: 3 input channels, 6 output channels, 5x5 kernel size
        images = images.to(device)
        labels = labels.to(device)

                # forward pass
        outputs = model(images)
        loss = F.nll_loss(outputs, labels)

                # back pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx) % 5 == 0: 
            print (f'Epoch [{epochs}/{epochs}], Step [{batch_idx}/{total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')
        # PATH = './cnn.pth'
        # torch.save(model.state_dict(), PATH)


# Test Data
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print("pred", pred.view(-1), "target", target.view(-1))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

def grayscaleloader(path:str)->Image.Image:
    with open(path, 'rb') as file:
        pic = Image.open(file).convert('L')
    return pic

def cropAll(in_path, write_path, name):
    dir_list = os.listdir(in_path)
    count = 0
    for j in range(len(dir_list)):
        read_path = in_path+"\\"+dir_list[j]
        cropped_img = crop.crop(read_path)
        cv2.imwrite(write_path+'\\'+name+str(count)+'.jpg', cropped_img)
        count+=1

def main():

    # Device config
    device = torch.device("cpu")

    # Hyper parameters
    epochs = 3 # can increase for more accuracy
    batch_size = 10
    learning_rate = 1.0

    # Load images into train and test datasets
    trainkwargs = {'batch_size':10, 'shuffle':True}
    testkwargs = {'batch_size':10, 'shuffle':True}
    path = os.path.join('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes')
    full_data = datasets.ImageFolder(root=path, transform=transforms.ToTensor(), loader=grayscaleloader) 
    print('Subfolder int assignment', full_data.class_to_idx)
    traindata, testdata = torch.utils.data.random_split(full_data, [0.8, 0.2]) # 80% train, 20% test
    loadtrain = DataLoader(traindata, **trainkwargs)
    loadtest = DataLoader(testdata, **testkwargs)
    # for batchnum, (pic, label) in enumerate(loadtrain):
    #     print('batch #', batchnum, 'pic shape', pic.shape, 'label shape', label.shape)

    model = Net().to(device)

    
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, epochs + 1):
        train(model, device, loadtrain, optimizer, epoch)
        test(model, device, loadtest)
        scheduler.step()

    
    torch.save(model.state_dict(), "zack_cnn.pt") # Coefficients to send to Raspberry Pi


    # test_img = Image.open('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedOpen\\open34.jpg')
    # #turn test_img into a tensor before calling forward
    # t1 = transforms.Grayscale(1)
    # t2 = transforms.Compose([transforms.ToTensor()])
    # # gray_img = t1(test_img)
    # # img_tensor = t2(gray_img)
    # img_tensor = t2(test_img)
    # #print(img_tensor.shape)
    # model.forward(img_tensor)

   
   


    # Sample Data Paths
    open_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\opendoor'
    closed_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\closedoor'
    zack_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack'

    # Crop and save sample data images into 'classes'
    # open_write_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedOpen'
    # cropAll(open_path, open_write_path, name='open')
    # closed_write_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedClosed'
    # cropAll(closed_path, closed_write_path, name='closed')
    # zack_write_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedZack'
    # cropAll(zack_path, zack_write_path, name='zack')
    
    

    # k = 40
    # i = 480
    # print("The number of outputs is: ", calc(i, k))
    
    
    
    

if __name__ == '__main__':
    main()
