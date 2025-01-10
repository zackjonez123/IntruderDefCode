'''
****** Captures images and saves them as binary images ******
'''

import cv2
import time
import crop

def main():
    """ Runs picloop()
    """
    #picloop('empty_L1')
    #picloop('empty_L0')
    #picloop('closed_L1')
    #picloop('closed_L0')
    picloop('Zack_L1')
    #picloop('Zack_L0')

def grayscale(name):
    """ Reads image captured from camera, writes it as a grayscale and threshold image.

    Args:
        name (string): desired name of the image file
    """
    # Load the input image
    image = cv2.imread('/home/pi/code/captured_images/'+name+'.jpg') #Pi path == /home/pi/code/captured_images/ w/ .jpg
    # Use the cvtColor() function to grayscale the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('/home/pi/code/captured_images/gray_'+name+'.jpg', gray_image) 
    th = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite('/home/pi/code/captured_images/newthresh_'+name+'.jpg', th) # Pi path == '/home/pi/code/captured_images/newthresh_' w/ .jpg


def picloop(img_type):
    """ Takes a number of pictures sequentially.

    Args:
        img_type (string): desired indicator for the type of image (open door, closed door, friendly, etc.)

    """
    #Take a certain number of pictures at a time (defined by the variable num)
    #changes the name of the image file to avoid overwritting
    name_count = 0
    name = ''
    count = 0
    num = 100 # Take 100 images
    while count < num:
        name_count += 1
        name = img_type + '_doorway' + str(name_count)
        pic(name)
        count += 1
        print(name)
        grayscale(name) # Converts to binary images
        print(count)
    return None

def cropCurrent(name):
     """ Takes image, crops, and saves as grayscale and binary images.

    Args:
        name (string): desired name of the image file
    """
    pic(name)
    pic(name)
    pic(name)
    pic(name)
    pic(name)
    pic(name)
    path = '/home/pi/code/captured_images/'+name+'.jpg'
    grayscale(name)
    cropped_image = crop.crop(path)
    cv2.imwrite(path, cropped_image)
    return None

def pic(name):
    """ Access USB cam and take picture.

    Args:
        name (string): desired name of the image file
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    (grabbed, frame) = cap.read()
    showimg = frame
    cv2.waitKey(1)
    time.sleep(0.3) # Wait 300 miliseconds
    image = '/home/pi/code/captured_images/'+name+'.jpg'
    cv2.imwrite(image, frame)
    cap.release()
    return None

if __name__ == '__main__':
    main()
