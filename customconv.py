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
    
def stats(in_image, kernel):
    conv1 = conv(in_image, kernel)
    #print(conv1)
    max1 = n.max(conv1)
    min1 = n.min(conv1)
    #print("Max value of"+descript+"is: ", max1)
    #print("Min value of"+descript+"is: ", min1)
    average = n.sum(conv1) / len(conv1)
    #print("Average value of"+descript+"is: ", average)
    
    stats = [max1, min1, average]
    return stats

def evaluation(in_path, test_path, kernel):
    dir_list1 = os.listdir(in_path)
    in_avmax = []
    in_avmin = [] 
    in_avavg = [] 
    inp = []
    dir_list2 = os.listdir(test_path)
    test_avmax = [] 
    test_avmin = [] 
    test_avavg = []
    test = [] 
    # Run stats on all images
    for i in range(len(dir_list2)):
        test_img = cv2.imread(test_path+"\\"+dir_list2[i])
        t = stats(test_img, kernel)
        # test_avmax.append(t[0])
        # test_avmin.append(t[1])
        # test_avavg.append(t[2])
        #test.append(t[3])
        test.append(t[0])
    for j in range(len(dir_list1)):
        in_img = cv2.imread(in_path+"\\"+dir_list1[j])
        x = stats(in_img, kernel)
        # in_avmax.append(x[0])
        # in_avmin.append(x[1])
        # in_avavg.append(x[2])
        inp.append(x[0])
        #inp.append(x[4])

    # tavg = n.sum(test_avavg) / len(test_avavg)
    # tavgmax = n.sum(test_avmax) / len(test_avmax)
    # tavgmin = n.sum(test_avmin) / len(test_avmin)
    
    test_stats = n.max(test)

    # inavg = n.sum(in_avavg) / len(in_avavg)
    # inavgmax = n.sum(in_avmax) / len(in_avmax)
    # inavgmin = n.sum(in_avmin) / len(in_avmin)
    in_stats = n.min(inp)

    print("Test Image stats are: ", test_stats)
    print("Zack Image stats are: ", in_stats)

def decision(in_path, kernel, threshold):
    img_read = cv2.imread(in_path)
    y = stats(img_read, kernel)
    if y[0] > threshold:
        return False # Friendly
    else:
        return True # Hostile


def main():
    in_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack')
    data_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zack_cropped\\cropped10.jpg')
    test_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\testpics')

    #zack_image = cv2.imread(in_path)
    kernel = cv2.imread(data_path)
    threshold = 750
    dir_list2 = os.listdir(test_path)
    for i in range(len(dir_list2)):
        path = test_path+"\\"+dir_list2[i]
        print(decision(path, kernel, threshold))
    #evaluation(in_path, test_path, kernel)
    #print(stats(zack_image, kernel, "Zack"))

    #test_image = cv2.imread(test_path)
    # print(stats(zack_image, kernel, "Zack Input"))
    # print(stats(test_image, kernel, "Test Input"))



# *********** Notes *************
# (w/ Filter cropped10.jpg)
# - between Zack and Test images, the averages and mins are similar, but the maxs are:
# Test avg max =717, Zack avg max = 775
# The max of maximums for test are 746, the minimum of maximums for Zack are 758

if __name__ == '__main__':
    main()
