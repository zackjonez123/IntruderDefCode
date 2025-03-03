'''
Functions for each step of the User Input Process
- Sample images
- Training CNN *
- Face Detection *
- Testing
** Need to give the user instructions throughout the process, which will be coded in main **
'''

from cv2 import threshold
import image_capture
import nnet
import facedetection
import confmat
import os

def dataPool(drect, name, num):
    image_capture.picloop(drect, name, num)
    nnet.cropAll(drect, drect, name)
    print("Data Collection Complete")
    return None

def confirm(prompt):
    while True:
        item = input(prompt)
        check = input("Is this correct? (Y/n): ")
        if check.lower() in ('y'):
            break
        elif check.lower() in ('n'):
            continue
        else:
            print(f'Error: Input {check} urecognized.')
            continue
    return item

def step1():
    print("Step 1 - Capturing Images")
    
    flag = 0
    while flag == 0:
        cases = input("Please enter the number of case folders you created: (ex. 3) ")
        cases = int(cases)
        print(cases)
        check = input("Is this correct? (Y/n): ")
        i = 1
        if check.lower() in ('y'):
            flag = 1
            while flag == 1 and i <= cases:
                print('Case ('+str(i)+'/'+str(cases)+'):')
                prompt = "Please enter the directory of the case you would like the capture: "
                drect = confirm(prompt)
                flag +=1
                while flag == 2:
                    prompt = "Please enter the desired name of your images (ex. 'closed' for a closed door): "
                    name = confirm(prompt)
                    flag +=1
                    while flag == 3:
                        prompt = "Please enter the desired number of images you would like to take for this case (ex. 100): "
                        num = confirm(prompt)
                        num = int(num)
                        image_capture.picloop(drect, name, num)
                        i += 1
                        flag = 1
        else:
            continue

    
    print("Congradulations! Step 1 is complete!")
    print(" ")

def step2():
    print("Step 2 - Training the Convolutional Neural Network (CNN)")
    flag = 0
    while flag == 0:
        prompt = "Please enter the directory of your classes: "
        drect = confirm(prompt)
        nnet.cnn(drect)
        
        print("0 loss after training grants the best results.")
        while True:
            repeat = input("Would you like to retrain the model? (Y/n): ")
            if repeat.lower() in ('y'):
                break
            elif repeat.lower() in ('n'):
                flag = 1
                break
            else:
                print(f'Error: Input {repeat} urecognized.')
                continue

    print("Congradulations! Step 2 is complete!")
    print(" ")

def step3():
    print("Step 3 - Training the Face Detection software")
    while True:
        prompt = "Please enter the directory of your 'occupied' case (the one with faces): "
        drect = confirm(prompt)
        thresholds = facedetection.threshold_calc(drect)
        break
    print("Congradulations! Step 3 is complete!")
    print(" ")
    return thresholds
        
    
def step4(thresholds):
    print("Step 4 - System Testing")
    print(" ")
    print("To complete step 4, you must take another set of 'occupied' images. Place these in a seperate directory from your original 'occupied' case")
    print(" ")
    flag = 0
    while flag == 0:
        prompt = "Please enter the directory of the case you would like the capture: "
        drect = confirm(prompt)
        flag +=1
        while flag == 1:
            prompt = "Please enter the desired name of your images (ex. 'closed' for a closed door): "
            name = confirm(prompt)
            flag +=1
            while flag == 2:
                prompt = "Please enter the desired number of images you would like to take for this case (ex. 100): "
                num = confirm(prompt)
                num = int(num)
                image_capture.picloop(drect, name, num)
                break
        else:
            continue

    while True:
        fcount = 0
        hcount = 0
        prompt = "Please enter the directory of your 'occupied' test case: "
        drect = confirm(prompt)
        while True:
            moe = input("Please enter the desired Margin of Error for the system (recommended = 1.0): ")
            moe = float(moe)
            dir_list = os.listdir(drect)
            for j in range(len(dir_list)):
                res = facedetection.decision(drect+'\\'+dir_list[j], moe, thresholds)
                if res == True:
                    hcount +=1
                else:
                    fcount +=1
            print("The number of friendly faces = "+str(fcount)+"/"+str(len(dir_list)))
            print("The number of hostile faces = "+str(hcount)+"/"+str(len(dir_list)))
            repeat = input("Would you like to change the Margin of Error? (Y/n): ")
            if repeat.lower() in ('y'):
                fcount = 0
                hcount = 0
                continue
            else:
                break
        break
       
    print("Congradulations! Step 4 is complete!")
    print(" ")
    return None


def main():
    print("Welcome to the IDS UI")
    print(" ")
    print("This system configuration is split into 4 parts: ")
    print("- Step 1) Capturing Images")
    print("- Step 2) Training the Convolutional Neural Network")
    print("- Step 3) Facial Detection")
    print("- Step 4) Final Testing")
   
    print(" ")
    print("If you are using a remote connection on the Raspberry Pi to capture images, CTRL+C after Step 1 is complete")
    print("And move the images from the Rasberry Pi onto your PC for Steps 2-4")
    print(" ")
    print("To start taking sample images, please create a directory for each of your cases.")
    print("For example, if you have a door that can be closed, create a directory like: 'pics\\cases\\closed'")
    print("and also, pics\\cases\\open and pics\\cases\\occupied, for when the same door is opened or occupied")
    print("The directories to these image folders will be needed for Step 2")
    print(" ")
    
    while True:
        prompt = "Please enter the step you would like to start at (either 1 or 2): (ex. '1' for step 1) "
        step = confirm(prompt)
        if step == str(1):
            step1()
            step2()
            thresh = step3()
            step4(thresh)
            break
        elif step == str(2):
            step2()
            thresh = step3()
            step4(thresh)
            break
        else:
            print(f'Error: Input {step} urecognized.')
            continue
print("In step 2, a file called 'results_cnn.pt' was created. Move this file, to your Rasberry Pi, along with the other requird Python files.")
print("Your IDS system is now trained and ready to use. If you run into any problems, such as false alarms, re-train the system with higher-quality images.")

if __name__ == '__main__':
    main()