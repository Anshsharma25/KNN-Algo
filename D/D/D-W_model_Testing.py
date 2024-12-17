import cv2
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

# Load the dry/wet classification model
dry_wet_model_path = 'Main_DW.pt'  # Update with the correct model path 
dry_wet_model = YOLO(dry_wet_model_path)

# Start capturing video from the laptop's camera
cap = cv2.VideoCapture(0)

# Define boundary parameters
frame_width = 640  # Frame width
frame_height = 480  # Frame height
boundary_y = frame_height // 2  # Horizontal boundary in the middle of the frame
boundary_thickness = 2  # Thickness of the boundary line

# Get the current time
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform dry/wet classification
    results = dry_wet_model(frame)

    # Check if results are available
    if results:
        boxes = results[0].boxes  # Access the first result
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class index

            # Calculate the complementary confidence
            main_conf = round(conf * 100)  # Confidence in %
            comp_conf = 100 - main_conf  # Complementary confidence in %

            # Get class names
            predicted_class = results[0].names[int(cls)]  # 'dry' or 'wet'
            if predicted_class == "wet":
                label = f"wet: {main_conf}% | dry: {comp_conf}%"
            else:
                label = f"dry: {main_conf}% | wet: {comp_conf}%"

            # Calculate the center of the bounding box
            center_y = (y1 + y2) / 2  # Vertical center

            # Check if the object is within the boundary
            if center_y < boundary_y:
                label = f"outside: {label}"
                color = (0, 165, 255)  # Orange for outside
            else:
                label = f"inside: {label}"
                color = (255, 0, 0)  # Blue for inside

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw the straight boundary line
    cv2.line(frame, (0, boundary_y), (frame_width, boundary_y), (0, 255, 0), boundary_thickness)  # Green for boundary

    # Convert frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame using matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')  # Hide axes
    plt.show(block=False)
    plt.pause(0.01)  # Pause to allow the image to display
    plt.clf()  # Clear the current figure

    # Break the loop if 10 seconds have passed
    if time.time() - start_time > 10:  # 10 seconds
        break

# Release the camera
cap.release()
plt.close()  # Close the matplotlib figure
