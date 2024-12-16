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
    #print(input_img.shape)

    # Crop the image
    cropped_img = input_img[130:370, 210:450] # bad crop = 180:320, 260:400
    # cv2.imshow("cropped image", cropped_img)
    # print(cropped_img.shape)
   
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_img

def main():
    crop('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\test\\coleT4\\coleT49.jpg')


if __name__ == '__main__':
    main()