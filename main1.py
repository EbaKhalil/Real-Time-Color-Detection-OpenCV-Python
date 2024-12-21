import cv2
import numpy as np
import math

# Define color ranges for red and green in HSV color space
red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])
green_lower = np.array([50, 100, 100])
green_upper = np.array([70, 255, 255])
# Define the blue color range in HSV
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize variables for zooming
initial_distance = None
zoom_factor = 1.0
# Initialize a flag to indicate if the two centroid points are combined
combined = False
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the image to get only the red and green colors
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    # Create a mask for blue color range
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply morphological operations to remove noise
    blue_mask = cv2.erode(blue_mask, None, iterations=2)
    blue_mask = cv2.dilate(blue_mask, None, iterations=2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    
    # Initialize the centroid coordinates
    centroid_points = []
    # Find contours in the red and green masks
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find the contour with the largest area and a minimum area threshold
    min_area_threshold = 500  # Adjust this value according to your needs
    max_area = -1
    max_contour = None
    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold and area > max_area:
            max_area = area
            max_contour = contour
    
    if max_contour is not None:
        # Find the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # Calculate the centroid of the bounding rectangle
        centroid_x = int(x + w / 2)
        centroid_y = int(y + h / 2)
        
        # Draw a red line on the frame at the centroid position
        cv2.line(frame, (centroid_x - 50, centroid_y), (centroid_x + 50, centroid_y), (0, 0, 255), 2)
    
    # Check if there are exactly two green contours
    if len(green_contours) == 2:
        # Get the bounding boxes of the two green contours
        green_boxes = [cv2.boundingRect(green_contour) for green_contour in green_contours]

        # Calculate the distance between the centers of the two green contours
        center1 = np.array([green_boxes[0][0] + green_boxes[0][2] // 2, green_boxes[0][1] + green_boxes[0][3] // 2])
        center2 = np.array([green_boxes[1][0] + green_boxes[1][2] // 2, green_boxes[1][1] + green_boxes[1][3] // 2])
        distance = np.linalg.norm(center1 - center2)

        # Zoom in or zoom out based on the distance
        if initial_distance is None:
            initial_distance = distance
        else:
            zoom_factor = initial_distance / distance

    # Resize the frame based on the zoom factor
    resized_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)

    # Check if any contour is detected
    if len(contours) > 0:
        for contour in contours:
            # Calculate the centroid of each contour
            moments = cv2.moments(contour)
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            centroid_points.append((centroid_x, centroid_y))

            # Draw a circle at each centroid
            cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

    # Check if two centroid points are combined
    if len(centroid_points) == 2:
        point1, point2 = centroid_points
        distance = np.linalg.norm(np.array(point1) - np.array(point2))

        # Define a threshold distance for combination
        threshold_distance = 40

        if distance < threshold_distance and not combined:
            # Display a message when the two centroid points are combined
            cv2.putText(frame, "Two points combined!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Save the image
            cv2.imwrite("combined_points.jpg", frame)
            combined = True
        elif distance >= threshold_distance:
            combined = False

    # Show the video stream
    cv2.imshow('frame', resized_frame)

    # If the 'q' key is pressed, stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()