"""
    ****** This is the manual convolutional pass for the IDS *********

    Returns:
        _type_: _description_
"""
import numpy as n
import cv2
import os

def crop(path):
    path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack'
    # Read input image
    #input_img = cv2.imread(path)
    dir_list1 = os.listdir(path)
    # Display cropped image
    #cv2.imshow("cropped image", cropped_img)

    # New path
    new_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack_cropped'
    dir_list2 = os.listdir(new_path)
    # Save cropped image
    name_count = 0
    name = ''
    #while loop to take 10 pictures at a time (b/c num is assigned to 10)
    count = 0
    for i in range(len(dir_list1)):
        img_read = cv2.imread(path+'\\'+dir_list1[i])
        # Crop the image
        cropped_img = img_read[120:250, 220:360]
        cv2.imwrite(new_path+'\\cropped'+str(count)+'.jpg', cropped_img)
        count += 1
    

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




# Convolution with previous images (x = input image, h = test image)
def convolution(in_path, data_path):
    y = []
    # Read input image
    input_img = cv2.imread(in_path)
    # Create list of data images
    path = data_path
    dir_list = os.listdir(path) 
    # Loop through data pool
    for i in range(len(dir_list)):
        # Get data pool image
        data_img_read = cv2.imread(path+'\\'+dir_list[i])
        data_img_flip = cv2.flip(data_img_read, 1)

        # Convert images into one-dimensional arrays to make convolution simpler
        in_arr = n.array(input_img)
        dp_arr = n.array(data_img_flip)
        flat_in_arr = in_arr.ravel()
        flat_dp_arr = dp_arr.ravel()

        f = flat_in_arr
        g = flat_dp_arr

        # Convolve input and data pool images
        prod_arr = [] 
        for j in range(len(f)):
            for k in range(len(g)):
                if f[j] < 50: # Correcting pixel values 
                    f[j] = 0
                else:
                    f[j] = 255

                if g[k] < 50:
                    g[k] = 0
                else:
                    g[k] = 255

                product = f[j] * g[k] # From convolution formula
                #print(product)
                prod_arr.append(product)
        s = sum(prod_arr)
        y.append(s) # Convolution result

    # Find the biggest convolution
    biggest = 0
    for i in range(len(y)):
        if y[i] > y[i-1]:
            biggest = y[i]
    return biggest
    # Boolean result

def numpy_conv(in_path, data_path):
    y = []
    # Read input image
    input_img = cv2.imread(in_path)
    # Create list of data images
    path = data_path
    dir_list = os.listdir(path) 
    # Loop through data pool
    for i in range(len(dir_list)):
        # Get data pool image
        data_img_read = cv2.imread(path+'\\'+dir_list[i])
        data_img_flip = cv2.flip(data_img_read, 1)

        # Convert images into one-dimensional arrays to make convolution simpler
        in_arr = n.array(input_img)
        dp_arr = n.array(data_img_flip)
        flat_in_arr = in_arr.ravel()
        flat_dp_arr = dp_arr.ravel()

        f = flat_in_arr
        g = flat_dp_arr

        for j in range(len(f)):
                for k in range(len(g)):
                    if f[j] < 50: # Correcting pixel values 
                        f[j] = 0
                    else:
                        f[j] = 255

                    if g[k] < 50:
                        g[k] = 0
                    else:
                        g[k] = 255
                    conv = n.convolve(f[j], g[k])
                    y.append(conv)
    # Find the biggest convolution
    biggest = 0
    for i in range(len(y)):
        if y[i] > y[i-1]:
            biggest = y[i]
    return biggest
    # Boolean result



def main():
    in_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack_cropped\\cropped5.jpg')
    data_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack_cropped')
    #convolution(in_path, data_path)
    #numpy_conv(in_path, data_path)
    crop(in_path)
if __name__ == '__main__':
    main()
