import cv2
import os
import imutils
import time
import numpy
import crop

def main():
    #state of doorway_lights on/off (L1 = on, L0 = off)
    #picloop('empty_L1')
    #picloop('empty_L0')
    #picloop('closed_L1')
    #picloop('closed_L0')
    picloop('Zack_L1')
    #picloop('Zack_L0')

def grayscale(name):
    # Load the input image
    image = cv2.imread('/home/pi/code/captured_images/'+name+'.jpg') #Pi path == /home/pi/code/captured_images/ w/ .jpg
    # Use the cvtColor() function to grayscale the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_save = cv2.imwrite('/home/pi/code/captured_images/gray_'+name+'.jpg', gray_image)
    # Use the cvtColor() function to threshold grayscale the image
    th = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    th_save = cv2.imwrite('/home/pi/code/captured_images/newthresh_'+name+'.jpg', th) # Pi path == '/home/pi/code/captured_images/newthresh_' w/ .jpg
    #thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #thresh_save = cv2.imwrite('/home/pi/code/captured_images/thresh_'+name+'.jpg', thresh)



def picloop(img_type):
    #Take a certain number of pictures at a time (defined by the variable num)
    #changes the name of the image file to avoid overwritting
    name_count = 0
    name = ''
    #while loop to take 10 pictures at a time (b/c num is assigned to 10)
    count = 0
    num = 100
    while count < num:
        name_count += 1
        name = img_type + '_doorway' + str(name_count)
        pic(name)
        count += 1
        print(name)
        grayscale(name)
        print(count)
    return None

def cropCurrent(name):
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
    #save_frame_camera_key(0, 'data/temp', 'camera_capture')
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

def save_frame_camera_key(device_num, dir_path, basename, ext='jpg', delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        return

    base_path = os.path.join(dir_path, basename)

    n = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('c'):
            cv2.imwrite('{}_{}.{}'.format(base_path, n, ext), frame)
            n += 1
        elif key == ord('q'):
            break

    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    main()