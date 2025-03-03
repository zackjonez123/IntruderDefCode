**Intruder Defense System (IDS) \- Zack Jones**

![build](https://github.com/user-attachments/assets/0834cd5f-1200-4f52-accf-259891ea4328)

**Abstract**

* **Major components**  
  * USB camera  
  * Wireless microcontroller   
  * Convolutional Neural Network (CNN)  
  * Servo motor  
* **Basic Workflow**  
  * The camera provides live footage to a microcontroller  
  * The microcontroller runs a machine-learning algorithm on images  
  * The microcontroller autonomously determines whether or not to activate a servo motor based on image processing result
 
![Work Flow Diagram](https://github.com/user-attachments/assets/065a01b5-8ff3-48c9-b54b-0e98ebfcb073)

* **Likely applications**  
  * unlocking a doggy door for only the owner's dog  
  * stopping a food slicer if a finger is too near the blade  
  * Sound an alarm if an intruder enters the doorway  \- **planning to demonstrate**

The **intent** of this project is to create a working concept that can be utilized in various applications. Home security and safety are paramount to homeowners, so they are the most likely to take an interest in the product.

**Specifications (12/01/24)**

* **Hardware:**  
  * Logitech C270 Webcam, Resolution: 720p  
  * Raspberry Pi 4  
  * Tower Pro SG-5010 Servo  
* **Wall Mount:**  
  * 1 ½” x 36” Aluminum Flat Bar   
  * 3 Sheetrock Anchors and Screws  
  * ¾” Velcro (for attaching hardware)  
* **Software:**  
  * *image\_capture.py* \- captures images from the USB webcam (saves as *grayscale* and *threshold* images)  
  * *crop.py* \- crops captured images into 240x240 images for the CNN  
  * *config.py* \-  configures the train and test arguments for the CNN  
  * *nnet.py* \- CNN  
  * *customconv.py* \- allows the system to differentiate between people using convolution  
  * *servo.py* \-  servo motor controls  
  * *RasPi.py* \- main file  
  * *confmat.py* \-  Confusion Matrix to test the accuracy of the system  
  * *facedetection.py* \- Uses object detection and geometric analysis (landmarks) to differentiate between people, replaces *customconv.py*
  * *ui.py* \- Allows a user to train the system via prompts in the terminal window (Added 3/3/25)

![FlowDiagram](https://github.com/user-attachments/assets/de042c25-57fc-4edc-8e0b-25ea10561957)

![examples1](https://github.com/user-attachments/assets/f0354690-043d-4f63-9861-4e5277662c8d)

**Planned Developments**

* **Improved Accuracy**  
  * More robust geometric analysis  
  * Revisit convolution method

* **Improved Speed**  
  * Cutting down the time it takes for the system to make a decision   
    * Currently takes \~10 seconds  
* **User Input**  
  * Would allow any user to:  
    * Train the CNN  
    * Modify parameters  
    * Test results  
  * Remote control (socket)  
  * Log Files

**Demo Video**  
[https://youtu.be/Fp7NFitlmFc?si=6M6vlmg8O0c3po1B](https://youtu.be/ewDh8F-WyO0?si=QsnjZ1JKTuA9oP7W)

