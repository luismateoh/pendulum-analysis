import numpy as np
import cv2
import csv
from math import dist

point_1 = (0, 0)
point_2 = (0, 0)
speed_1 = 0
speed_2 = 0
acceleration = 0
time = 0

data = []


def click_event(event, x, y):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('pendulumVideo.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")


def process_image(img_frame):
    in_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)

    lower = np.array([3, 200, 205])
    upper = np.array([9, 240, 235])

    mask = cv2.inRange(in_hsv, lower, upper)

    # Erosion y Dilation
    kernel = np.ones((3, 3), np.uint8)

    dilate = cv2.dilate(mask, kernel, iterations=12)
    erode = cv2.erode(dilate, kernel, iterations=9)

    return erode


def get_centroid(img):
    global point_2
    global point_1
    point_1 = point_2
    processed_frame = process_image(frame)
    # convert the grayscale image to binary image
    # ret,thresh = cv2.threshold(img,127,255,0)

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    m = cv2.moments(largest_contour)

    # calculate x,y coordinate of center
    c_x = int(m["m10"] / m["m00"])
    c_y = int(m["m01"] / m["m00"])
    cv2.circle(img, (c_x, c_y), 3, (0, 0, 255), -1)
    cv2.putText(img, "Centroid", (c_x - 30, c_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    point_2 = (c_x, c_y)
    return img


def speed_calc(img):
    global speed_1
    global speed_2
    global acceleration

    speed_1 = speed_2
    # https://www.section.io/engineering-education/approximating-the-speed-of-an-object-and-its-distance/
    # Speed is distance / time

    distance = (dist(point_1, point_2) * 0.086)  # conversion to cm/s
    distance = distance * 0.01  # conversion to m/s
    speed_2 = distance / (1 / 30)

    acceleration = abs(speed_1 - speed_2) / (1 / 30)

    cv2.putText(img, str("Vel:  {:.3f} m/s".format(speed_1)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(img, str("Acc:  {:.3f} m/s^2".format(acceleration)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 0, 255), 2)
    # cv2.putText(img, "Centroid", (40, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img


# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:

        image_with_centroid = get_centroid(frame)

        image_with_measures = speed_calc(image_with_centroid)

        data.append(
            {
                "position": point_2,
                "time": round(time, 3),
                "speed": round(speed_1, 3),
                "acceleration": round(acceleration, 3)
            }
        )

        time = time + (1 / 30)

        scale_percent = 0.7  # percent of original size

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent)  # float `width`
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent)  # float `height`
        dim = (width, height)
        res_frame = cv2.resize(image_with_measures, dim)

        cv2.imshow('Frame', res_frame)

        var = cv2.setMouseCallback

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

data_names = ['position', 'time', 'speed', 'acceleration']

with open('movement_data.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_names)
    writer.writeheader()
    writer.writerows(data)
