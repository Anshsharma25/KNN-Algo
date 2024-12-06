import cv2
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')

# Create a small dataset for KNN (bounding box size vs distance)
# Dummy examples (modify based on actual calibration)
bbox_size = np.array([[50, 50], [100, 100], [150, 150], [200, 200], [250, 250]])  # (width, height) of bounding boxes
distance = np.array([3, 2, 1.5, 1, 0.5])  # Corresponding distances in meters

# Initialize KNN regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(bbox_size, distance)  # Train the KNN model

# Open video capture
cap = cv2.VideoCapture(0)

# Set start time
start_time = time.time()

# Define boundary parameters
boundary_distance = 1.0  # 1 meter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)
    
    # Flag to track if any object is out of range
    out_of_range = False
    inside_objects = []  # List to track objects inside the boundary
    closest_outside_distance = float('inf')  # Initialize to a large value
    closest_outside_object = None  # To keep track of the closest outside object

    # Loop through the detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = box.cls[0]

            # Calculate bounding box width and height
            box_width = x2 - x1
            box_height = y2 - y1

            # Predict the distance using KNN (based on bounding box size)
            predicted_distance = knn.predict([[box_width, box_height]])[0]

            # Check if object is within 1 meter
            if predicted_distance <= boundary_distance:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for inside
                cv2.putText(frame, f"Inside 1m ({predicted_distance:.2f}m)", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                inside_objects.append(model.names[int(class_id)])  # Add object name to list
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for outside
                cv2.putText(frame, f"Outside 1m ({predicted_distance:.2f}m)", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                out_of_range = True  # Set flag if any object is out of range
                
                # Update closest outside object if this one is closer
                if predicted_distance < closest_outside_distance:
                    closest_outside_distance = predicted_distance
                    closest_outside_object = model.names[int(class_id)]  # Store the closest object's name

            # Draw label with confidence
            cv2.putText(frame, f'{model.names[int(class_id)]} {conf:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw boundary line for 1 meter (you can adjust the position as needed)
    boundary_y = int(frame_height / 2)  # Middle of the frame as an example
    cv2.line(frame, (0, boundary_y), (frame_width, boundary_y), (0, 255, 0), 2)  # Green for boundary

    # Display the frame using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=False)  # Allow the loop to continue running
    plt.pause(0.001)  # Pause briefly to allow the plot to update

    # If any object was out of range, print a message
    if out_of_range:
        print("Object detected outside 1 meter")

    # Print which objects are inside the 1-meter boundary
    if inside_objects:
        print(f"Objects inside 1 meter: {', '.join(inside_objects)}")

    # Print the closest outside object
    if closest_outside_object is not None:
        print(f"Closest object outside 1 meter: {closest_outside_object} at {closest_outside_distance:.2f}m")

    # Check if 10 seconds have passed
    if time.time() - start_time > 10:
        break

cap.release()
#---------------------------------------------------------------------------------------by using yolov8 

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import threading

# Load your custom garbage detection YOLO model
garbage_model = YOLO('C:/Users/HP/Downloads/R1-20241015T142216Z-001/R1/runs/detect/train/weights/best.pt')

# Load the dry/wet classification model
dry_wet_model = YOLO('C:/Users/HP/Downloads/Dry-Model-20241010T131307Z-001/Dry-Model/Dataset/runs/detect/train2/weights/best.pt')

# Open video capture
cap = cv2.VideoCapture(0)  # Change to your video source if necessary

running = True  # Flag to control the loop

# Function to check for user input to stop the camera
def check_for_exit():
    global running
    while running:
        user_input = input()
        if user_input.lower() == 'q':
            running = False

# Start the input thread
input_thread = threading.Thread(target=check_for_exit)
input_thread.start()

while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform garbage detection
    garbage_results = garbage_model(frame)

    # Loop through the detected garbage objects
    for result in garbage_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = box.cls[0]

            # Draw bounding box for detected garbage
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for detected garbage
            cv2.putText(frame, f'{garbage_model.names[int(class_id)]} {conf:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Crop the detected garbage for dry/wet classification
            cropped_garbage = frame[y1:y2, x1:x2]

            # Classify the cropped garbage as dry or wet
            dry_wet_results = dry_wet_model(cropped_garbage)
            dry_wet_label = 'Unknown'
            dry_wet_confidence = 0.0  # Initialize confidence

            for dw_result in dry_wet_results:
                if len(dw_result.boxes) > 0:
                    dw_class_id = dw_result.boxes[0].cls[0]
                    dry_wet_label = dry_wet_model.names[int(dw_class_id)]  # Get the class name (dry/wet)
                    dry_wet_confidence = dw_result.boxes[0].conf[0]  # Get the confidence score

            # Print the dry/wet classification
            print(f"Garbage detected: {dry_wet_label} with confidence {dry_wet_confidence:.2f}")
            cv2.putText(frame, f"Dry/Wet: {dry_wet_label} ({dry_wet_confidence:.2f})", 
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Display the frame using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=False)  # Allow the loop to continue running
    plt.pause(0.001)  # Pause briefly to allow the plot to update

# Cleanup
cap.release()
cv2.destroyAllWindows()  # This will still not work if OpenCV's GUI is not supported
#----------------------------------------------------------------------------------------------by using both custom model 

import cv2
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load your custom YOLO model
# Load your custom YOLO model
model = YOLO('C:/Users/HP/Downloads/R1-20241015T142216Z-001/R1/runs/detect/train/weights/best.pt')


# Create a small dataset for KNN (bounding box size vs distance)
# Dummy examples (modify based on actual calibration)
bbox_size = np.array([[50, 50], [100, 100], [150, 150], [200, 200], [250, 250]])  # (width, height) of bounding boxes
distance = np.array([3, 2, 1.5, 1, 0.5])  # Corresponding distances in meters

# Initialize KNN regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(bbox_size, distance)  # Train the KNN model

# Open video capture
cap = cv2.VideoCapture(0)

# Set start time
start_time = time.time()

# Define boundary parameters
boundary_distance = 1.0  # 1 meter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)
    
    # Flag to track if any object is out of range
    out_of_range = False
    inside_objects = []  # List to track objects inside the boundary
    closest_outside_distance = float('inf')  # Initialize to a large value
    closest_outside_object = None  # To keep track of the closest outside object

    # Loop through the detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = box.cls[0]

            # Calculate bounding box width and height
            box_width = x2 - x1
            box_height = y2 - y1

            # Predict the distance using KNN (based on bounding box size)
            predicted_distance = knn.predict([[box_width, box_height]])[0]

            # Check if object is within 1 meter
            if predicted_distance <= boundary_distance:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for inside
                cv2.putText(frame, f"Inside 1m ({predicted_distance:.2f}m)", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                inside_objects.append(model.names[int(class_id)])  # Add object name to list
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for outside
                cv2.putText(frame, f"Outside 1m ({predicted_distance:.2f}m)", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                out_of_range = True  # Set flag if any object is out of range
                
                # Update closest outside object if this one is closer
                if predicted_distance < closest_outside_distance:
                    closest_outside_distance = predicted_distance
                    closest_outside_object = model.names[int(class_id)]  # Store the closest object's name

            # Draw label with confidence
            cv2.putText(frame, f'{model.names[int(class_id)]} {conf:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw boundary line for 1 meter (you can adjust the position as needed)
    boundary_y = int(frame_height / 2)  # Middle of the frame as an example
    cv2.line(frame, (0, boundary_y), (frame_width, boundary_y), (0, 255, 0), 2)  # Green for boundary

    # Display the frame using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=False)  # Allow the loop to continue running
    plt.pause(0.001)  # Pause briefly to allow the plot to update

    # If any object was out of range, print a message
    if out_of_range:
        print("Object detected outside 1 meter")

    # Print which objects are inside the 1-meter boundary
    if inside_objects:
        print(f"Objects inside 1 meter: {', '.join(inside_objects)}")

    # Print the closest outside object
    if closest_outside_object is not None:
        print(f"Closest object outside 1 meter: {closest_outside_object} at {closest_outside_distance:.2f}m")

    # Check if 10 seconds have passed
    if time.time() - start_time > 10:
        break

cap.release()


#---------------------------------------------------------------------------------------------- working with simple way of knn 


import cv2
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load your custom YOLO model for garbage detection
garbage_model = YOLO('C:/Users/HP/Downloads/R1-20241015T142216Z-001/R1/runs/detect/train/weights/best.pt')

# Load the dry/wet classification model
dry_wet_model = YOLO('C:/Users/HP/Downloads/Dry-Model-20241010T131307Z-001/Dry-Model/Dataset/runs/detect/train2/weights/best.pt')

# Open video capture
cap = cv2.VideoCapture(0)

# Create a more detailed dataset for KNN (bounding box size vs distance)
bbox_size = np.array([[50, 50], [100, 100], [150, 150], [200, 200], [250, 250]])  # Example values
distance = np.array([3, 2, 1.5, 1, 0.5])  # Corresponding distances in meters

# Initialize KNN regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(bbox_size, distance)  # Train the KNN model

# Set start time
start_time = time.time()

# Define boundary parameters
boundary_distance = 1.0  # 1 meter

# Define the figure and axis for Matplotlib
plt.figure()
plt.ion()  # Enable interactive mode

# Function to draw a curved boundary line centered in the frame
def draw_curved_boundary(frame):
    frame_height, frame_width = frame.shape[:2]
    curve_depth = 30  # Control the maximum depth of the curve above and below the midpoint
    curve_center_y = frame_height // 2  # Midpoint height

    num_points = frame_width
    curve_points = []

    for x in range(num_points):
        t = x / (frame_width - 1)  # Parameter from 0 to 1
        y = curve_center_y + int(curve_depth * np.sin(np.pi * t))
        curve_points.append((x, y))

    for i in range(len(curve_points) - 1):
        cv2.line(frame, curve_points[i], curve_points[i + 1], (0, 255, 0), 2)

    return frame

# Loop to capture frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform garbage detection
    garbage_results = garbage_model(frame)

    garbage_count = 0
    closest_outside_distance = float('inf')
    closest_label = ""

    for result in garbage_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = box.cls[0]

            if conf < 0.5:  # Adjust this threshold as needed
                continue

            garbage_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f'{garbage_model.names[int(class_id)]} {conf:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cropped_garbage = frame[y1:y2, x1:x2]

            dry_wet_results = dry_wet_model(cropped_garbage)
            dry_wet_label = 'Common'
            dry_wet_confidence = 0.0

            for dw_result in dry_wet_results:
                if len(dw_result.boxes) > 0:
                    dw_class_id = dw_result.boxes[0].cls[0]
                    dry_wet_label = dry_wet_model.names[int(dw_class_id)]
                    dry_wet_confidence = dw_result.boxes[0].conf[0]

            print(f"Garbage detected: {dry_wet_label} with confidence {dry_wet_confidence:.2f}")
            cv2.putText(frame, f"Dry/Wet: {dry_wet_label} ({dry_wet_confidence:.2f})", 
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            box_width = x2 - x1
            box_height = y2 - y1
            predicted_distance = knn.predict([[box_width, box_height]])[0]

            if predicted_distance <= boundary_distance:
                cv2.putText(frame, f"Inside 1m ({predicted_distance:.2f}m)", 
                            (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Outside 1m ({predicted_distance:.2f}m)", 
                            (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if predicted_distance < closest_outside_distance:
                    closest_outside_distance = predicted_distance
                    closest_label = dry_wet_label

    frame = draw_curved_boundary(frame)

    cv2.putText(frame, f"Garbage Count: {garbage_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)

    if closest_outside_distance < float('inf'):
        print(f"Closest object outside 1 meter: {closest_label} at {closest_outside_distance:.2f}m")

    if time.time() - start_time > 20:
        break

# Cleanup
cap.release()
plt.ioff()
plt.show()
#---------------------------------------------------------------------------------------------------------working with both model and knn here it is it perfect way!!!.
