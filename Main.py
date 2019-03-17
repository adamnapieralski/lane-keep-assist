import cv2
from Camera import Camera


camera = Camera()

# camera.calibrateCamera()
camera.loadCameraProperties()
img = cv2.imread('./road_test_1.jpg')
imgUndist = camera.undistortImage(img)
# cv2.namedWindow("w1")
# cv2.imshow("w1", imgUndist)

camera.perspectiveTransform(imgUndist)

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    cv2.destroyAllWindows()