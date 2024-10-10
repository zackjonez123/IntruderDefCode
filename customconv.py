"""
    ****** This is the manual convolutional pass for the IDS *********

    Returns:
        _type_: _description_
"""
import numpy as n
import cv2
import os

def crop():
    path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack'
    
    # Read input image
    input_img = cv2.imread(path)
    # print(input_img.shape)
    # cv2.imshow("original image", input_img)
    dir_list1 = os.listdir(path)

    # Display cropped image
    # cropped_img = input_img[270:400, 250:420]
    # cv2.imshow("cropped image", cropped_img)

    # New path
    new_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack_cropped'
    # dir_list2 = os.listdir(new_path)
    
    count = 6
    for i in range(len(dir_list1)):
        name = 'newthresh_Zack_L1_doorway'+str(count)+'.jpg'
        img_read = cv2.imread(path+'\\'+name)
        # Crop the image
        cropped_img = img_read[270:400, 250:420]
        cv2.imwrite(new_path+'\\cropped'+str(count)+'.jpg', cropped_img)
        count += 1
    

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def rbg_correction(in_path, data_path):
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
        
        for j in range(len(input_img)):
             for k in range(len(data_img_flip)):
                  # Correcting r, g, b values for data pool image
                  for w in range(3):
                    if data_img_flip[j,k][w] < 50:
                        data_img_flip[j,k][w] = 0
                    else:
                        data_img_flip[j,k][w] = 255
                    if input_img[j,k][w] < 50:
                            input_img[j,k][w] = 0
                    else:
                        input_img[j,k][w] = 255
    return input_img
    return data_img_flip

# Convolution with previous images (x = input image, h = test image)
def convolution(in_path, data_path):
    y = []
    # Read input image
    input_img = cv2.imread(in_path)
    # Create list of data images
    path = data_path
    dir_list = os.listdir(path) 
    #print(dir_list)
    prod_arr = []
    # Loop through data pool
    for i in range(len(dir_list)):
        # Get data pool image
        data_img_read = cv2.imread(path+'\\'+dir_list[i])
        data_img_flip = cv2.flip(data_img_read, 1)
        
        for j in range(len(input_img)):
             for k in range(len(data_img_flip)):
                # Correcting r, g, b values for data pool image
                  for w in range(3):
                    if data_img_flip[j,k][w] < 50:
                        data_img_flip[j,k][w] = 0
                    else:
                        data_img_flip[j,k][w] = 255
                    if input_img[j,k][w] < 50:
                            input_img[j,k][w] = 0
                    else:
                        input_img[j,k][w] = 255

        for j in range(len(input_img)):
             for k in range(len(data_img_flip)):
                # Only convolve the "r" value of each image
                prod = input_img[j,k][0] * data_img_flip[j,k][0]
                prod_arr.append(prod)
        s = sum(prod_arr)
        y.append(s)
        print(y)
  # Find the biggest convolution
    # biggest = 0
    # for i in range(len(y)):
    #     if y[i] > y[i-1]:
    #         biggest = y[i]
    # return biggest

        # Convert images into one-dimensional arrays to make convolution simpler
        # in_arr = n.array(input_img)
        # dp_arr = n.array(data_img_flip)
        # flat_in_arr = in_arr.ravel()
        # flat_dp_arr = dp_arr.ravel()

        # f = flat_in_arr
        # g = flat_dp_arr
        # print(f.shape)
        # print(g.shape)
        # Convolve input and data pool images
        # prod_arr = [] 
        # for j in range(len(f)):
        #     for k in range(len(g)):
        #         if f[j+2] < 10 or g[k+2] == 0:
        #             f[j] = 0
        #             product = 0
        #         else:
        #             f[j] = 255
        #             g[k] = 255
        #             product = f[j] * g[k] # From convolution formula

                
        #         print(product)
        #         prod_arr.append(product)
        # s = sum(prod_arr)
        # y.append(s) # Convolution result

   
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
    in_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack_cropped\\cropped17.jpg')
    data_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack_cropped')
    convolution(in_path, data_path)
    #numpy_conv(in_path, data_path)
    #crop()
if __name__ == '__main__':
    main()
