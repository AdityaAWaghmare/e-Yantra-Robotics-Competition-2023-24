'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_3895
# Author List:		Aditya Waghmare, Ashwin Agrawal, Soham Pawar, Siddhant Godbole
# Filename:			task_4a.py


####################### IMPORT MODULES #######################
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image                    
##############################################################



################# ADD UTILITY FUNCTIONS HERE #################

"""
You are allowed to add any number of functions to this code.
"""
def get_subimage(image, x1, y1, x2, y2):
    subimage = image[y1:y2, x1:x2]
    if subimage.size == 0:
        return None
    subimage = cv2.rotate(subimage, cv2.ROTATE_90_CLOCKWISE)
    return subimage

def preprocess_image(image):
    
    if isinstance(image, np.ndarray):  # Check if the input is a NumPy array
        image = Image.fromarray(image)  # Convert NumPy array to PIL Image

    image = image.convert('RGB')
    image = np.array(image)
        
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    # image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
    # image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
    # image = cv2.resize(image, (182, 182), interpolation=cv2.INTER_CUBIC)
    # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    # image = cv2.resize(image, (260, 260), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
    
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   	])
    input_tensor = transform(image)
        
    input_batch = input_tensor.unsqueeze(0) 
    
    return input_batch

def classify_subimage(subimage, model):
    if subimage is None:
        return None, None

    # Preprocess the subimage for classification
    subimage = preprocess_image(subimage)

    # Perform classification using the pre-trained model
    with torch.no_grad():
        output = model(subimage)

    # Get the predicted class and confidence
    confidence, predicted_class = torch.max(output.data, 1)

    return confidence.item(), predicted_class.item()

def extract_subimages_from_video(frame):
    
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to separate the white border and other elements
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    subimages = []
    for contour in contours:
        # Ignore small contours (noise)
        if cv2.contourArea(contour) < 200:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore regions containing ArUco markers (you may need to adjust the dimensions)
        if w <= 90 and h <= 90:
            continue
        if w >=125 or h >= 125:
            continue
        
        # Draw bounding box on the original frame
        x = x+10
        y = y+10
        w = w-20
        h = h-20
        x2 = x+w
        y2 = y+h
        
        dict = {'box': (x , y , x2 , y2 )}
        subimages.append(dict)
        
    return subimages


##############################################################


def task_4a_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable
    
    Arguments:
    ---
    You are not allowed to define any input arguments for this function. You can 
    return the dictionary from a user-defined function and just call the 
    function here

    Returns:
    ---
    `identified_labels` : { dictionary }
        dictionary containing the labels of the events detected
    """  
    identified_labels = {}  
    
##############	ADD YOUR CODE HERE	##############

    # Map the class labels to their corresponding folder names
    class_to_label = {
        0: "combat",
        1: "human_aid_rehabilitation",
        2: "military_vehicles",
        3: "fire",
        4: "destroyed_buildings"
    }
    # Load the pre-trained classification model
    pretrained_model = torch.load('trained_model_final_b4_300.pth')  # Adjust the model file name accordingly
    pretrained_model.eval()

    # Open a connection to the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    # Set frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Adjust as needed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Adjust as needed
    # # Disable auto white balance
    # cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    #exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)
    cv2.waitKey(10000)
    #once
    # Read a frame from the camera
    ret, frame1 = cap.read()
    while not ret:
        ret, frame1 = cap.read()
        
    frame1 = frame1[0:1080, 0*1920//1280:970*1920//1280]
    # Extract subimages
    # subimages1 = extract_subimages_from_video(frame1)
    subimages1 = {
         'A': (1283, 793, 1374-7, 885), #military_vehicles
         'B': (1068+10, 289, 1156, 377-10), #human_aid_rehabilitation
         'C': (854, 284, 940, 371), #fire
         'D': (851, 805+5, 938-5, 895-2), #destroyed_buildings
         'E': (512, 781, 595, 869) #combat
         }
    # l = []
    
    for key, value in subimages1.items():
        subimg = get_subimage(frame1, *value)
        confidence, prediction = classify_subimage(subimg, pretrained_model)
        
        if prediction is not None:
            predicted_class_name = class_to_label[prediction]
            identified_labels[key] = predicted_class_name
    
##################################################
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)
    
    #my code
    # Map the class labels to their corresponding folder names
    class_to_label = {
        0: "combat",
        1: "human_aid_rehabilitation",
        2: "military_vehicles",
        3: "fire",
        4: "destroyed_buildings"
    }
    # Load the pre-trained classification model
    pretrained_model = torch.load('trained_model.pth')  # Adjust the model file name accordingly
    pretrained_model.eval()

    # Open a connection to the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    # Set frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Adjust as needed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Adjust as needed
    # # Disable auto white balance
    # cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    #exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)
    cv2.waitKey(10000)
    #once
    # Read a frame from the camera
    ret, frame1 = cap.read()
    while not ret:
        ret, frame1 = cap.read()
        
    frame1 = frame1[0:1080, 0*1920//1280:970*1920//1280]
    # Extract subimages
    # subimages1 = extract_subimages_from_video(frame1)
    subimages1 = [{'box': (851, 805+5, 938-5, 895-2)}, 
         {'box': (1283, 793, 1374-7, 885)}, 
         {'box': (512, 781, 595, 869)},
         
         {'box': (1068+10, 289, 1156, 377-10)}, 
         {'box': (854, 284, 940, 371)}]
    l = []
    
    for subimage_info in subimages1:
        subimg = get_subimage(frame1, *subimage_info['box'])
        confidence, prediction = classify_subimage(subimg, pretrained_model)
        
        if confidence is not None and prediction is not None:
                t = []
                t.append(subimage_info)
                # t.append(confidence)
                t.append(prediction)
                l.append(t)
     
               
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[0:1080, 0*1920//1280:970*1920//1280]
        # Extract subimages
        subimages = subimages1
        # frame = cv2.GaussianBlur(frame, (1, 1), 0)

        # Classify and draw bounding boxes on the original frame
        for subimage_info, prediction in l:
            # subimg = get_subimage(frame, *subimage_info['box'])
            # confidence, prediction = classify_subimage(subimg, pretrained_model)

            if prediction is not None:
                predicted_class_name = class_to_label[prediction]

                # Draw bounding box on the original frame
                x1, y1, x2, y2 = subimage_info['box']
                color = (0, 255, 0)  # Green color for bounding boxes
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Put label on the bounding box
                label = f"{predicted_class_name} "
                cv2.putText(frame, label, (x1 - 30, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness)

        # Display the resulting frame
        desired_width = 1280
        desired_height = 720
        resized_frame = cv2.resize(frame, (desired_width, desired_height))

        cv2.imshow('Live Feed with Bounding Boxes', resized_frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
