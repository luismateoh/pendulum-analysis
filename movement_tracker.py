"""This program is used to calculate the speed and acceleration of a pendulum. The code presents an implementation of
motion analysis using OpenCV and numpy in Python. The objective is to analyze the movement of a pendulum captured in
a video. The program uses the following steps: 1. Masking: The video is converted to HSV color space and a mask is
applied to the image to isolate the pendulum. 2. Erosion and Dilation: The mask is eroded and dilated to remove noise
and fill holes. 3. Contour detection: The largest contour is detected and the centroid is calculated. 4. Speed and
acceleration calculation: The speed and acceleration of the pendulum are calculated using the centroid of the
pendulum. 5. Data storage: The data is stored in a CSV file."""

import numpy as np
import cv2
import csv
from math import dist

# Variables
point_1 = (0, 0)
point_2 = (0, 0)
speed_1 = 0
speed_2 = 0
acceleration = 0
time = 0
data = []

# Read video
cap = cv2.VideoCapture('pendulumVideo.mp4')
if not cap.isOpened():
    print("Error opening video stream or file")


def process_image(img_frame):
    # BGR to HSV
    in_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
    # Mask
    lower = np.array([3, 200, 205])
    upper = np.array([9, 240, 235])
    mask = cv2.inRange(in_hsv, lower, upper)
    # Erosion y Dilation
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(mask, kernel, iterations=20)
    erode = cv2.erode(dilate, kernel, iterations=15)

    return erode


def get_centroid(img, binary_frame):
    # find contours in the binary image
    contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest_contour)

    # calculate x,y coordinate of center
    c_x = int(m["m10"] / m["m00"])
    c_y = int(m["m01"] / m["m00"])

    # Paint centroid and contour
    cv2.circle(img, (c_x, c_y), 3, (0, 0, 255), -1)
    cv2.putText(img, "Centroid", (c_x - 30, c_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)

    return img, (c_x, c_y)


def speed_calc(img, point_a, point_b):
    distance = (dist(point_a, point_b) * 0.086)  # conversion to cm/s
    distance = distance * 0.01  # conversion to m/s
    speed = distance / (1 / 30)  # get speed in m/s (1/30 is the frame rate)

    cv2.putText(img, str("Vel:  {:.3f} m/s".format(speed_1)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return img, speed


def acceleration_calc(img, speed_a, speed_b):
    acceleration_a = (speed_a - speed_b) / (1 / 30)
    cv2.putText(img, str("Acc:  {:.3f} m/s^2".format(acceleration)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 0, 255), 2)
    return img, acceleration_a


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        processed_frame = process_image(frame)
        frame_with_centroid, point_1 = get_centroid(frame, processed_frame)
        frame_with_speed, speed_1 = speed_calc(frame_with_centroid, point_1, point_2)
        frame_with_acceleration, acceleration = acceleration_calc(frame_with_speed, speed_1, speed_2)

        # Save data
        data.append(
            {
                "position": point_1,
                "time": round(time, 3),
                "speed": round(speed_1, 3),
                "acceleration": round(acceleration, 3)
            }
        )
        # Update variables
        point_2 = point_1
        speed_2 = speed_1
        time = time + (1 / 30)

        # Rescale
        scale_percent = 0.7  # percent of original size
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent)  # float `width`
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent)  # float `height`
        dim = (width, height)
        res_frame = cv2.resize(frame_with_acceleration, dim)

        cv2.imshow('Frame', res_frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# CSV File
data_names = ['position', 'time', 'speed', 'acceleration']
with open('movement_data.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_names)
    writer.writeheader()
    writer.writerows(data)
