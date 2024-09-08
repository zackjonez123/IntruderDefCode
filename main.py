    """
    ********General Outline of Intruder Defense Code*************

    Goals:
    - Webcam provides live footage of doorway
    - Images are run through a classifier which determines if someone is "friendly" (me) or "hostile" (my brother, and maybe other people)
    - If classified as "friendly", servo motor will not move, if "hostile", it will

    Steps:
    **Training**
    - Gather data (w/ image_capture.py)
    - Successfully classify data (image_classifier.py)
    - Check performance and optimize classifier
    **Implementation**
    - Get live footage from webcam
    - Run live footage through classifier (within main.py)
    - Load coefficients and predict in Rasberry Pi
    - Test results (1 or 0 will be indicated through terminal to represent servo motor on/off)

    """

def main():

def live_classifier(img):
    # ***Pseudo code*** 
    # if image_classifier(img) ---> friendly
    # return False (servo off)
    # print "0"
    # if image_classifier(img) ---> hostile
    # return True (servo on)
    # print "1"



def live_cam(img):

    return img




if __name__ == '__main__':
    main()