#!/home/pi/code/virtenv1/bin python3
# Copyright 2024, Zack Jones, All rights reserved.
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional
import torch.nn.functional as F
from nnet import Net
from nnet import grayscaleloader
import image_capture
import crop
#import customconv
import facedetector
import cv2
import servo
import time

def main():
    '''
    Daemon: Take pics every 2 seconds and run CNN, output results (Done)
    If results are hostile, turn servo motor and continue, else -> continue (Done)
    Stops on shutdown (Done)
    Start on startup (Done)
    Remote control through a socket (Spring)
    Log file (Spring)
    '''
    model = Net() # Stores the CNN as the object "model"
    model.load_state_dict(torch.load("results_cnn.pt", weights_only=True)) # Loads file of coefficients
    model.eval() # So torch doesn't change coefficients
    
    while True:
        time.sleep(2) # Run every 2 seconds
        now=time.time() # Start recording time

        # Use grayscaleloader to load image as a threshold image (purely black and white)
        image_capture.cropCurrent('current_img') # Takes picture with USB cam and crops it for the CNN
        capt_path = "/home/pi/code/captured_images/current_img.jpg" # path to captured image (from USB cam)
        grayscale = grayscaleloader(capt_path) # Reads the captured image
        img = transforms.functional.to_tensor(grayscale) # Loads the current image as a tensor
        img = img[None, :, :, :] # Makes image tensor 4 dimensional for the CNN
        with torch.no_grad():
            data = img.to("cpu")
            output = model(data) # Run CNN with captured image
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        # Conditional statements:
        # - if output == closed or open ==> flag "friendly" (do nothing)
        if pred[0][0] == 0 or pred[0][0] == 2:
            print("Friendly")
            print("Open/Closed Time (seconds)", time.time()-now) # CNN run time
        # - if output == occupied ==> run facedetector.py
        # - if facedetector == "friendly" ==> do nothing
        # - otherwise ==> turn servo motor
        else:
            gray_path = "/home/pi/code/captured_images/gray_current_img.jpg" # path to the captured image in grayscale
            result = facedetector.decision(gray_path, 1) # run facedetector with the grayscale captured image
            if result == True:
                print("Hostile")
                servo.servo1() # activate servo
                print("Hostile Time (seconds)", time.time()-now) # CNN + facedetector runtime
            else:
                print("Friendly")
                print("Friendly Time (seconds)", time.time()-now) # CNN + facedetector runtime
        
if __name__ == '__main__':
    main()
