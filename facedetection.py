'''
    ****** Face Detection and Processing ******
'''
from imutils import face_utils
import numpy as np
import dlib
import cv2
import os
import math

Facial_IDX = {"nose":(27, 35), "left eye":(42, 48), "right eye":(36, 42)} # Indexes for landmarks of interest (eyes and nose)

def landmarks(path):
    """ Detects the face and calculates the distances between landmarks of interest

    Args:
        path (string): path to the input image

    Returns:
        dists (array (float)) : array of distances between landmarks of interest
    """
    # Read images
    img = cv2.imread(path)

    # Detect Faces
    detector = dlib.get_frontal_face_detector()
    rects = detector(img, 1)

    # Generate Landmarks
    predictor = dlib.shape_predictor('C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\shape_predictor_68_face_landmarks.dat')
    dists = []
    # Loop over face detections
    for (i, rect) in enumerate(rects):
        # Get x, y coordinates of landmarks
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # draw rectangle around the face

        # Distances between landmarks of interest
        a = math.dist(shape[36], shape[39]) # left_eye_width
        b = math.dist(shape[42], shape[45]) # right_eye_width
        c = math.dist(shape[39], shape[27]) # left_eye_to_nose
        d = math.dist(shape[42], shape[27]) # right_eye_to_nose
        e = math.dist(shape[27], shape[30]) # nose_length
        f = math.dist(shape[30], shape[33]) # nose_height
        g = math.dist(shape[31], shape[35]) # nose_width

        # Append distances to dists array
        dists.append(a)
        dists.append(b)
        dists.append(c)
        dists.append(d)
        dists.append(e)
        dists.append(f)
        dists.append(g)
    return dists
            
def decision(path, MoE):
    """ Determines whether or not a face's landmarks are within the acceptable threshold ("friendly")

    Args:
        path (string): path to the image
        MoE (int): margin of error

    Returns:
        boolean: True ==> "Hostile", False ==> "Friendly"
    """
    face_map = landmarks(path) # get landmark distances from the image
    
    # "Friendly" thresholds for each landmark of interest
    a_thresh = [15+MoE, 12-MoE] # left_eye_width
    b_thresh = [14+MoE, 11-MoE] # right_eye_width
    c_thresh = [13+MoE, 7-MoE] # left_eye_to_nose
    d_thresh = [13+MoE, 7-MoE] # right_eye_to_nose
    e_thresh = [22+MoE, 17-MoE] # nose_length
    f_thresh = [9+MoE, 7-MoE] # nose_height
    g_thresh = [14+MoE, 12-MoE] # nose_width

    # Loop through each distance from the image
    for i in range(len(face_map)):
        if face_map[i] <= a_thresh[1] or face_map[0] >= a_thresh[0]:
            return True # Hostile
        elif face_map[1] <= b_thresh[1] or face_map[1] >= b_thresh[0]:
            return True # Hostile
        elif face_map[2] <= c_thresh[1] or face_map[2] >= c_thresh[0]:
            return True # Hostile
        elif face_map[3] <= d_thresh[1] or face_map[3] >= d_thresh[0]:
            return True # Hostile
        elif face_map[4] <= e_thresh[1] or face_map[4] >= e_thresh[0]:
            return True # Hostile
        elif face_map[5] <= f_thresh[1] or face_map[5] >= f_thresh[0]:
            return True # Hostile
        elif face_map[6] <= g_thresh[1] or face_map[6] >= g_thresh[0]:
            return True # Hostile
        else:
            return False # Friendly
