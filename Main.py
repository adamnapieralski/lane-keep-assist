import cv2
from Camera import Camera


camera = Camera()

# camera.calibrateCamera()
camera.loadCameraProperties()

sourceImg = 'samples/video_snap_1.png'
img = cv2.imread(sourceImg)
imgUndist = camera.undistortImage(img)
# cv2.namedWindow("w1")
# cv2.imshow("w1", imgUndist)

imgTrans = camera.perspectiveTransform(imgUndist)
camera.colorTransforms(imgTrans)

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    cv2.destroyAllWindows()