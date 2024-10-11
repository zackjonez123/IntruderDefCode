"""
    ****** This is the manual convolutional pass for the IDS *********

    Returns:
        _type_: _description_
"""
import numpy as n
import cv2
import os


# def rbg_correction(in_path, data_path):
#     # Read input image
#     input_img = cv2.imread(in_path)
#     # Create list of data images
#     path = data_path
#     dir_list = os.listdir(path) 
#     # Loop through data pool
#     for i in range(len(dir_list)):
#         # Get data pool image
#         data_img_read = cv2.imread(path+'\\'+dir_list[i])
#         data_img_flip = cv2.flip(data_img_read, 1)
        
#         for j in range(len(input_img)):
#              for k in range(len(data_img_flip)):
#                   # Correcting r, g, b values for data pool image
#                   for w in range(3):
#                     if data_img_flip[j,k][w] < 50:
#                         data_img_flip[j,k][w] = 0
#                     else:
#                         data_img_flip[j,k][w] = 255
#                     if input_img[j,k][w] < 50:
#                             input_img[j,k][w] = 0
#                     else:
#                         input_img[j,k][w] = 255
#     return input_img
    
# *********************************************************************************************************



def conv(in_image, kernel):
    # Array of convolutions
    convolved_arr = []
    #print(in_image.shape)
    # Size of filter image
    fil_width = kernel.shape[1]
    fil_height = kernel.shape[0]
    # Size of input image
    in_width = in_image.shape[1]
    in_height = in_image.shape[0]

    # To keep the filter from going out of the in_image bounds
    width = in_width - fil_width 
    height = in_height - fil_height 

    for i in range(height):
        for j in range(width):
            splice = in_image[i : i + fil_height, j : j + fil_width] # splicing the input image
            convolved = n.sum(n.multiply(splice, kernel))
            norm_convolved = convolved / 1000 # Normalize results 
            convolved_arr.append(norm_convolved)

    return convolved_arr

# New function for kernel
# def data_conv(in_path, data_path):
#     in_image = cv2.imread(in_path)
    
   
# Loop through the directory of cropped images
 # Create list of data images
    # path = data_path
    # dir_list = os.listdir(path) 
    # print(dir_list)
    # data_img_read = cv2.imread(path+'\\'+dir_list[0])
    # data_img_flip = cv2.flip(data_img_read, 1)
    # resized_data = cv2.resize(data_img_flip, (data_img_flip.shape[0], data_img_flip.shape[0]))
    # cv2.imshow("data",resized_data)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Loop through data pool
    # sum_arr = []
    # for i in range(len(dir_list)):
    #     count = 0
    #     # Get data pool image
    #     data_img_read = cv2.imread(path+'\\'+dir_list[i])
    #     data_img_flip = cv2.flip(data_img_read, 1)
    #     resized_data = cv2.resize(data_img_flip, (data_img_flip.shape[0], data_img_flip.shape[0]))
    #     conv2 = conv(resized_data)
    #     for j in range(conv1.shape[0]):
    #         for k in range(conv1.shape[1]) :
    #             if conv1[j,k] == conv2[j,k]:
    #                 count+=1
    #     sum_arr.append(count)
    # #print(sum_arr)
    # biggest = 0
    # for w in range(len(sum_arr)):
    #     if sum_arr[w] > sum_arr[w-1]:
    #         biggest = w
    # return biggest

# convolve within the loop
# return array of results

# def compare(in_path, data_path):
#     in_image = cv2.imread(in_path)
#     resized_img = cv2.resize(in_image, (in_image.shape[0], in_image.shape[0]))
#     conv1 = conv(resized_img)
#     #conv2 = data_conv(data_path)
    
def stats(in_image, kernel, descript):
    conv1 = conv(in_image, kernel)
    #print(conv1)
    print("Max value of"+descript+"is: ", n.max(conv1))
    print("Min value of"+descript+"is: ", n.min(conv1))
    average = n.sum(conv1) / len(conv1)
    print("Average value of"+descript+"is: ", average)
    # Set threshold and evaluate

def main():
    in_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack\\newthresh_Zack_L1_doorway7.jpg')
    data_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack_cropped\\cropped6.jpg')
    test_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\testpics\\cole1.jpg')

    zack_image = cv2.imread(in_path)
    kernel = cv2.imread(data_path)
    
    print(conv(zack_image, kernel))

    #test_image = cv2.imread(test_path)
    #print(stats(zack_image, kernel, "Zack Input"))
    #print(stats(test_image, kernel, "Test Input"))

    #print(data_conv(in_path, data_path))
    # print(resized_img.shape)
    # cv2.imshow("resized",resized_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #convolution(in_path, data_path)
    #numpy_conv(in_path, data_path)

if __name__ == '__main__':
    main()
