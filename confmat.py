"""
********** Confusion Matrix ********
Intended for testing the accuracy of facedetection.py
Copyright 2024, Zack Jones, All rights reserved.
"""

import os
import cv2
import facedetection

def conv_matrix(path):
    """Counts the number of "friendly" and "hostile" images within an image folder, determined by facedetection.py

    Args:
        path (string): path to the image folder

    Returns:
        fvh (int array): number of "friendly" and "hostile" images, fvh[0] == "friendly", fvh[1] == "hostile"
    """
    f_count = 0
    h_count = 0
    dir_list = os.listdir(path)
    for i in range(len(dir_list)):
        capt_path = path+'\\'+dir_list[i] # path to captured image (from USB cam)
        res = facedetection.decision(capt_path, 1)
        if res == False:
            f_count += 1
        else:
            h_count += 1
        # f_total = 1 - ((100 - f_count) / 100)
        # h_total = 1 - ((100 - h_count) / 100)
    fvh = [f_count, h_count] # Friendly vs Hostile
    return fvh


def main():
   """
   Example:
   """
    print("***Confusion Matrix***")
    print("*Cole vs Zack 240 Crop - Conv Test*")
    # Paths to image folders
    test_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\proto\\gray_c')
    in_path = ('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\proto\\gray_z')
    res1 = conv_matrix(in_path)
    res2 = conv_matrix(test_path)
    print("Friendly ID: Expected = 100, Actual = ", res1[0])
    print("Hostile ID: Expected = 100, Actual = ", res2[1])

if __name__ == '__main__':
    main()
