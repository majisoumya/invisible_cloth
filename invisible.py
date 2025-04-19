import cv2
import numpy as np
import time

# Start webcam
cap = cv2.VideoCapture(0)

# Let the camera warm up
print("Capturing background... Please stay out of the frame.")
time.sleep(2)

# Capture the background
for i in range(30):
    ret, background = cap.read()
background = np.flip(background, axis=1)

print("Background captured. You can now wear the yellow cloak!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define yellow color range in HSV
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Clean up the mask
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    yellow_mask = cv2.dilate(yellow_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Inverse mask
    inverse_mask = cv2.bitwise_not(yellow_mask)

    # Extract non-yellow part from current frame
    res1 = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Extract yellow part from background
    res2 = cv2.bitwise_and(background, background, mask=yellow_mask)

    # Combine both results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Show the result
    cv2.imshow("Invisibility Cloak - Yellow", final_output)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
