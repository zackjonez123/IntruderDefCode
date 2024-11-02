import torch
from nnet import Net

def main():
    model = Net()
    model.load_state_dict(torch.load("zack_cnn.pt"), weights_only=True)
    model.eval()
    # Use grayscaleloader to load image = grayscale
    img = transforms.functional.to_tensor(grayscale)
    img = img[None, :, :, :]
    with torch.no_grad():
        data = img.to("CPU")
        output = F.sigmoid(model(data))
        print(output)