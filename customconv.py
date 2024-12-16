"""
    ****** This is the manual convolutional pass for the IDS ******
"""
import numpy as n
import cv2
import os

def conv(in_image, kernel):
    """ Performs convolution between the input image and filter (kernel) image
        as the kernel shifts across the input image. 
        These values are stored in convolved_arr

    Args:
        in_image (image): captured image from the doorway
        kernel (image): cropped image of the friendly occupant's eyes and nose

    Returns:
        convolved_arr (array): convolution results
    """
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
            splice = in_image[i : i + fil_height, j : j + fil_width] # splicing the input image by the kernel's dimensions
            convolved = n.sum(n.multiply(splice, kernel)) # multiplying each pixel in the spliced input image with the pixels of the kernel image, then summing the products
            norm_convolved = convolved / 1000 # Normalize results 
            convolved_arr.append(norm_convolved) # store the convolution result into convolved_arr

    return convolved_arr
    
def stats(in_image, kernel):
    """ Runs conv() and calculates the maximum convolution result

    Args:
        in_image (image): captured image from the doorway
        kernel (image): cropped image of the friendly occupant's eyes and nose

    Returns:
        max1 (float): maximum convolution value 
    """
    conv1 = conv(in_image, kernel) # run conv()
    max1 = n.max(conv1) # calculate the max of convolved_arr
    
    return max1

def evaluation(in_path, test_path, kernel):
    dir_list1 = os.listdir(in_path)
    inp = []
    dir_list2 = os.listdir(test_path)
    test = [] 
    # Run stats on all images
    for i in range(len(dir_list2)):
        test_img = cv2.imread(test_path+"\\"+dir_list2[i])
        t = stats(test_img, kernel)
        test.append(t)
    for j in range(len(dir_list1)):
        in_img = cv2.imread(in_path+"\\"+dir_list1[j])
        x = stats(in_img, kernel)
        inp.append(x)
    
    test_min = n.min(test)
    test_max = n.max(test)

    in_max = n.max(inp)
    in_min = n.min(inp)

    print("Test Image stats are: ", test_min, test_max)
    print("Zack Image stats are: ", in_min, in_max)

def decision(in_path, kernel, upper_thresh, lower_thresh):
    """ Determines whether the maximum convolution value of the input image is in the accepable threshold.
        The acceptable threshold was determined by taking the maximum and minimum of the max convolution values
        for 100 "Friendly" images (max ==> upper_thresh, min ==> lower_thresh).
        Any max convolution value outside this range is "Hostile"

    Args:
        in_path (string): path to the input image
        kernel (image): cropped image of the friendly occupant's eyes and nose
        upper_thresh (int): highest acceptable max convolution value
        lower_thresh (int): lowest acceptable max convolution value

    Returns:
        boolean: False for "Friendly", True for "Hostile
    """
    img_read = cv2.imread(in_path) # Read the input image
    y = stats(img_read, kernel) # Get the maximum of the convolution values

    # check if within acceptable threshold
    if lower_thresh < y and y < upper_thresh:
        return False # Friendly
    else:
        return True # Hostile


def main():
    in_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedOccupied')
    data_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\filters\\filter2.jpg')
    test_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\testcrop\\cole')

    # timg = cv2.imread(test_path)
    # print(n.shape(timg))

    #******** Determining threshold **********
    #zack_image = cv2.imread(in_path)
    kernel = cv2.imread(data_path)
    upper_thresh = 715
    lower_thresh = 625
    # dir_list1 = os.listdir(in_path)
    # dir_list2 = os.listdir(test_path)
    # for i in range(len(dir_list2)):
    #     path2 = test_path+"\\"+dir_list2[i]
    #     print(decision(path2, kernel, threshold))

    # for j in range(len(dir_list2)):
    #     path1 = in_path+"\\"+dir_list1[j]
    #     print(decision(path1, kernel, threshold))

    evaluation(in_path, test_path, kernel)
    # print(stats(zack_image, kernel, "Zack"))

    # test_image = cv2.imread(test_path)
    # print(stats(zack_image, kernel, "Zack Input"))
    # print(stats(test_image, kernel, "Test Input"))



# *********** Notes *************
# (w/ Filter cropped10.jpg)
# - between Zack and Test images, the averages and mins are similar, but the maxs are:
# Test avg max =717, Zack avg max = 775
# The max of maximums for test are 746, the minimum of maximums for Zack are 758

#11/16/24
# Test Image stats are:  846.936
# Zack Image stats are:  804.111
# w/ cropv22.jpg

if __name__ == '__main__':
    main()
