import cv2
import numpy as np
import glob

objPoints = []
imgPoints = []

chessboardPattern = (6, 6)

testImgPath = glob.glob("./cc-test-images-2/*")

cv2.namedWindow("Camera calibration", cv2.WINDOW_AUTOSIZE)

objPt = np.zeros((chessboardPattern[0] * chessboardPattern[1], 3), np.float32)
objPt[:, :2] = np.mgrid[0:chessboardPattern[0], 0:chessboardPattern[1]].T.reshape(-1, 2)

for image in testImgPath:
    calibImg = cv2.imread(image)
    imH, imW = calibImg.shape[:2]
    imgReSize = (int(imW), int(imH))
    calibImg = cv2.resize(calibImg, imgReSize)
    ret, corners = cv2.findChessboardCorners(calibImg, chessboardPattern)
    cv2.drawChessboardCorners(calibImg, chessboardPattern, corners, ret)
    if ret:
        imgPoints.append(corners)
        objPoints.append(objPt)
        cv2.imshow("Camera calibration", calibImg)
    cv2.waitKey(50)

retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, imgReSize, None, None)

optCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgReSize, 1)
print(cameraMatrix)
print(optCameraMatrix)
print(roi)

x, y, w, h = roi

if retval:
    np.savetxt('cameraMatrix.csv', cameraMatrix)
    np.savetxt('distCoeffs.csv', distCoeffs)

camMat = np.genfromtxt('cameraMatrix.csv')
dstCoeffs = np.genfromtxt('distCoeffs.csv')

print(camMat)
print(dstCoeffs)
img = cv2.imread('./pic.jpg')
imH, imW = img.shape[:2]
imgReSize = (int(imW), int(imH))
optCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camMat, dstCoeffs, imgReSize, 1)
print(optCameraMatrix)
print(roi)
x, y, w, h = roi

imgUndist = cv2.undistort(img, camMat, dstCoeffs, None, optCameraMatrix)
cv2.imwrite('result.jpg', img)
imgUndist = imgUndist[y:y+h, x:x+w]
cv2.imwrite('resultCropped.jpg', imgUndist)
# cv2.imshow("Camera Calibration", imgUndist)

cv2.waitKey(2000)

# print(camMat)
# print(dstCoeffs)



cv2.destroyAllWindows()
