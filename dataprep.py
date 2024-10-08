import re
import numpy
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

#friendly_dir = 'C:/Users/kelly/Desktop/IDEs and Sims/IntruderDef/pics'

# gets image and converts to a tensor
def imgtensor(img_type, num): 
    dtype = torch.float
    device = torch.device("cpu")
    image = cv2.imread('C:/Users/kelly/Desktop/IDEs and Sims/IntruderDef/pics/'+img_type+'/newthresh_closed_L1_doorway'+str(num)+'.jpg')
    dimensions = tuple(image.shape[1::-1]) 
    print(dimensions)
    itens = torch.tensor(image, dtype=dtype, device=device, requires_grad=False)
    return itens
    

def main():

    #transforms.Compose([
     #transforms.CenterCrop(10),
     #transforms.PILToTensor(),
     #transforms.ConvertImageDtype(torch.float),])
    # Input data

    # Output data


    imgtensor('closedoor', 1)
    
if __name__ == '__main__':
    main()