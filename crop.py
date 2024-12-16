import cv2

def crop(read_path):
    """ Crops an image into a 240x240 square for the Convolutional Neural Network (CNN)

    Args:
        read_path (string): path to the image

    Returns:
        cropped_img (image): cropped image (240x240)
    """
    # Read input image
    input_img = cv2.imread(read_path)

    # Crop the image
    cropped_img = input_img[130:370, 210:450] # 240x240
   
    return cropped_img
