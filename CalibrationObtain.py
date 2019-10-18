from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import time

camera_resolution = (1296, 972)

image = np.zeros((1296, 972, 3), np.uint8)


def capture_image(event, x, y, flags, param):
    global image
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Captured")
        timestr = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite("cam-calibration-images/IMG_" + timestr + ".png", image)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = camera_resolution
camera.framerate = 25
camera.rotation = 90
rawCapture = PiRGBArray(camera, size=camera_resolution)

i = 1

# allow the camera to warmup
time.sleep(0.1)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", capture_image)

start_time = time.time()


# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    image_show = cv2.resize(image, (360, 240))

    # show the frame
    cv2.imshow("Frame", image_show)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
    #
    # if time.time() - start_time >= 3:
    #     print("saving")
    #     cv2.imwrite("cam-calibration-images/calib" + str(i) + ".png", image)
    #     i += 1
    #     start_time = time.time()
    # if i > 25:
    #     cv2.destroyAllWindows()
    #     break


