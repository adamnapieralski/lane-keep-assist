import cv2
import numpy as np
import glob

class Camera:

    cameraMatrix = []
    cameraDistCoeffs = []

    calibChessboardPattern = (6, 6)
    calibImgsPath = glob.glob("./ccalib-test-images-2/*")

    def calibrateCamera(self):
        objPoints = []
        imgPoints = []
        objPt = np.zeros((self.calibChessboardPattern[0] * self.calibChessboardPattern[1], 3), np.float32)
        objPt[:, :2] = np.mgrid[0:self.calibChessboardPattern[0], 0:self.calibChessboardPattern[1]].T.reshape(-1, 2)

        for image in self.calibImgsPath:
            calibImg = cv2.imread(image)
            imH, imW = calibImg.shape[:2]
            imgReSize = (int(imW), int(imH))
            calibImg = cv2.resize(calibImg, imgReSize)
            ret, corners = cv2.findChessboardCorners(calibImg, self.calibChessboardPattern)
            cv2.drawChessboardCorners(calibImg, self.calibChessboardPattern, corners, ret)
            if ret:
                imgPoints.append(corners)
                objPoints.append(objPt)

        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, imgReSize, None,
                                                                             None)

        if retval:
            optAlpha = 1
            optCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgReSize, optAlpha)
            np.savetxt('cameraOptMatrix.csv', optCameraMatrix)
            np.savetxt('cameraDistCoeffs.csv', distCoeffs)
            np.savetxt('cameraCalibROI.csv', roi)

    def loadCameraProperties(self):
        self.cameraMatrix = np.genfromtxt('cameraMatrix.csv')
        self.cameraDistCoeffs = np.genfromtxt('cameraDistCoeffs.csv')

    def undistortImage(self, img):
        imgUndist = cv2.undistort(img, self.cameraMatrix, self.cameraDistCoeffs)
        cv2.imwrite('IMG_20190311_152834_result.jpg', img)
