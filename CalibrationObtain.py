from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

camera_resolution = (640, 480)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = camera_resolution
camera.framerate = 40
camera.rotation = 90
rawCapture = PiRGBArray(camera, size=camera_resolution)

# allow the camera to warmup
time.sleep(0.1)

start_time = time.time()
i = 1

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
        break

    if time.time() - start_time >= 3:
        print("saving")
        cv2.imwrite("cam-calib-imgs/calib" + str(i) + ".jpg", image)
        i += 1
        start_time = time.time()
