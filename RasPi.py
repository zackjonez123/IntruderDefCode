import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional
import torch.nn.functional as F
from nnet import Net
from nnet import grayscaleloader
import image_capture
import crop


def main():
    model = Net() # Stores the CNN as the object "model"
    model.load_state_dict(torch.load("results_cnn.pt", weights_only=True)) # Loads file of coefficients
    model.eval() # So torch doesn't change coefficients
    # Use grayscaleloader to load image = grayscale
    image_capture.cropCurrent('current_img') # Takes picture with USB cam
    capt_path = "/home/pi/code/captured_images/current_img.jpg" # path to captured image (from USB cam)
    grayscale = grayscaleloader(capt_path) # Reads the current image
    img = transforms.functional.to_tensor(grayscale) # Loads the current image as a tensor
    img = img[None, :, :, :] # Makes image tensor 4 dimensional for the CNN
    with torch.no_grad():
        data = img.to("cpu")
        output = F.sigmoid(model(data))
        print(output)

    # Create conditional statements:
    # - if output == closed or open ==> flag "friendly" (do nothing)
    # - if output == occupied ==> run customconv.py
    # - if customconv == "friendly" ==> do nothing
    # - otherwise ==> turn servo motor
if __name__ == '__main__':
    main()