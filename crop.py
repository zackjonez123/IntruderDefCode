import cv2

def crop(read_path):
    
    # Read input image
    input_img = cv2.imread(read_path)
    #print(input_img.shape)

    # Crop the image
    cropped_img = input_img[0:480, 80:560]
    # cv2.imshow("cropped image", cropped_img)
    # print(cropped_img.shape)
   
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_img

