import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional
import torch.nn.functional as F
from nnet import Net
from nnet import grayscaleloader
import image_capture
import crop
import customconv
import cv2
import os

def path(capt_path):
    model = Net() # Stores the CNN as the object "model"
    model.load_state_dict(torch.load("results_cnn.pt", weights_only=True)) # Loads file of coefficients
    model.eval() # So torch doesn't change coefficients
    
    grayscale = grayscaleloader(capt_path) # Reads the current image
    img = transforms.functional.to_tensor(grayscale) # Loads the current image as a tensor
    img = img[None, :, :, :] # Makes image tensor 4 dimensional for the CNN
    with torch.no_grad():
        data = img.to("cpu")
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #print(pred[0][0])

    # Create conditional statements:
    # - if output == closed or open ==> flag "friendly" (do nothing)
    if pred[0][0] == 0 or pred[0][0] == 2:
        #print("Friendly")
        return 0
    else:
        #print("Occupied")
        return 1
    # else:
    #     #data_path = "/home/pi/code/captured_images/cropped10.jpg" # Path to the filter image
    #     data_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\filters\\cropv22.jpg') # Path to the filter image
    #     kernel = cv2.imread(data_path)
    #     threshold = 640
    #     result = customconv.decision(capt_path, kernel, threshold)
    #     if result == True:
    #         return 1
    #         #print("Hostile")
    #     else:
    #         return 0
    #         #print("Friendly")
        

def main():
    print("hello")
    # model = Net() # Stores the CNN as the object "model"
    # model.load_state_dict(torch.load("results_cnn.pt", weights_only=True)) # Loads file of coefficients
    # model.eval() # So torch doesn't change coefficients
    # # Use grayscaleloader to load image = grayscale
    # #image_capture.cropCurrent('current_img') # Takes picture with USB cam
    # capt_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\testcrop\\cole63.jpg" # path to captured image (from USB cam)
    # grayscale = grayscaleloader(capt_path) # Reads the current image
    # img = transforms.functional.to_tensor(grayscale) # Loads the current image as a tensor
    # img = img[None, :, :, :] # Makes image tensor 4 dimensional for the CNN
    # with torch.no_grad():
    #     data = img.to("cpu")
    #     output = model(data)
    #     pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #     print(pred[0][0])

    # # Create conditional statements:
    # # - if output == closed or open ==> flag "friendly" (do nothing)
    # if pred[0][0] == 0 or pred[0][0] == 2:
    #     print("Friendly")
    # else:
    #     print("Occupied")
    # - if output == occupied ==> run customconv.py
    # - if customconv == "friendly" ==> do nothing
    # - otherwise ==> turn servo motor
    # else:
    #     #data_path = "/home/pi/code/captured_images/cropped10.jpg" # Path to the filter image
    #     data_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\filters\\cropv22.jpg') # Path to the filter image
    #     kernel = cv2.imread(data_path)
    #     threshold = 640
    #     result = customconv.decision(capt_path, kernel, threshold)
    #     if result == True:
    #         return 1
    #         #print("Hostile")
    #     else:
    #         return 0
    #         #print("Friendly")
if __name__ == '__main__':
    main()