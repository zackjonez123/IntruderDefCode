**Intruder Defense System (IDS) \- Zack Jones**

**Abstract**

* **Major components**  
  * USB camera  
  * Wireless microcontroller   
  * Convolutional Neural Network (CNN)
  * Servo motor  
*  **Basic Workflow**  
  * The camera provides live footage to a microcontroller  
  * The microcontroller runs a machine-learning algorithm on images  
  * The microcontroller autonomously determines whether or not to activate a servo motor based on image processing result  
* **Likely applications**  
  * unlocking a doggy door for only the owner's dog  
  * stopping a food slicer if a finger is too near the blade  
  * Sound an alarm if an intruder enters the doorway  \- **planning to demonstrate**

The **intent** of this project is to create a working concept that can be utilized in various applications. Home security and safety are paramount to homeowners, so they are the most likely to take an interest in the product.

**Specifications (12/01/24)**

* **Hardware:**  
  * Logitech C270 Webcam, Resolution: 720p  
  * Rasberry Pi 4  
  * Tower Pro SG-5010 Servo  
* **Wall Mount:**  
  * 1 ½” x 36” Aluminum Flat Bar   
  * 3 Sheetrock Anchors and Screws  
  * ¾” Velcro (for attaching hardware)  
* **Software:**  
  * *image\_capture.py* \- captures images from the USB webcam  
  * *crop.py* \- crops captured images into 240x240 images for the CNN  
  * *config.py* \-  configures the train and test arguments for the CNN  
  * *nnet.py* \- CNN  
  * *customconv.py* \- allows the system to differentiate between people using convolution  
  * *servo.py* \-  servo motor controls  
  * *RasPi.py* \- main file  
  * *confmat.py* \-  Confusion Matrix to test the accuracy of the system
  * *facedetector.py* \- Uses object detection and geometric analysis to differentiate between people, replaces *customconv.py*

**Planned Developments**

* **Improved Accuracy**  
  * Wireless door sensor (to keep images consistent)  
  * More complex geometric analysis 
* **Improved Speed**  
  * Cutting down the time it takes for the system to make a decision   
    * Currently takes approx. 10 seconds  
* **User Input**  
  * Would allow any user to:  
    * Train the CNN  
    * Modify parameters  
    * Test results  
  * Remote control (socket)  
  * Log Files

	
