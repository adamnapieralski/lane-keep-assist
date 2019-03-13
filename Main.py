import cv2
from Camera import Camera


camera = Camera()

# camera.calibrateCamera()
camera.loadCameraProperties()
img = cv2.imread('./IMG_20190311_152834.jpg')
camera.undistortImage(img)
