"""
    ****** This is the manual convolutional pass for the IDS ******
    Copyright 2024, Zack Jones, All rights reserved.
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
