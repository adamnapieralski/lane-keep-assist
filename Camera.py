import cv2
import numpy as np
import glob
import os

class Camera:

    cameraMatrix = []
    cameraOptMatrix = []
    cameraDistCoeffs = []
    cameraCalibROI = []

    imageSize = []

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
            self.imageSize = (imW, imH)
            # imgReSize = (int(imW), int(imH))
            #calibImg = cv2.resize(calibImg, imgSize)
            ret, corners = cv2.findChessboardCorners(calibImg, self.calibChessboardPattern)
            cv2.drawChessboardCorners(calibImg, self.calibChessboardPattern, corners, ret)
            if ret:
                imgPoints.append(corners)
                objPoints.append(objPt)
                print(self.calibImgsPath.index(image), ": ", os.path.basename(image))

        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, self.imageSize, None, None)

        if retval:
            optAlpha = 1
            optCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, self.imageSize, optAlpha)
            np.savetxt('cameraMatrix.csv', cameraMatrix)
            np.savetxt('cameraOptMatrix.csv', optCameraMatrix)
            np.savetxt('cameraDistCoeffs.csv', distCoeffs)
            np.savetxt('cameraCalibROI.csv', roi)

    def loadCameraProperties(self):
        self.cameraMatrix = np.genfromtxt('cameraMatrix.csv')
        self.cameraOptMatrix = np.genfromtxt('cameraOptMatrix.csv')
        self.cameraDistCoeffs = np.genfromtxt('cameraDistCoeffs.csv')
        self.cameraCalibROI = np.genfromtxt('cameraCalibROI.csv')

    def undistortImage(self, img):
        imH, imW = img.shape[:2]
        self.imageSize = (imW, imH)
        imgUndist = cv2.undistort(img, self.cameraMatrix, self.cameraDistCoeffs, None, self.cameraOptMatrix)
        cv2.imwrite('road_test_1_result.jpg', imgUndist)
        x, y, w, h = self.cameraCalibROI
        x, y, w, h = int(x), int(y), int(w), int(h)
        imgUndist = imgUndist[y:y + h, x:x + w]
        imgUndist = cv2.resize(imgUndist, self.imageSize)
        cv2.imwrite('road_test_1_result_cropped_scaled.jpg', imgUndist)
        return imgUndist

    def perspectiveTransform(self, img):

        srcPoints = np.array([[self.imageSize[0] / 2 - 725, self.imageSize[1]],
                     [self.imageSize[0] / 2 + 725, self.imageSize[1]],
                     [self.imageSize[0] / 2 + 190, self.imageSize[1] / 2],
                     [self.imageSize[0] / 2 - 190, self.imageSize[1] / 2]], dtype=np.float32)

        dstPoints = np.array([[self.imageSize[0] / 2 - 440, self.imageSize[1]],
                     [self.imageSize[0] / 2 + 440, self.imageSize[1]],
                     [self.imageSize[0] / 2 + 440, self.imageSize[1] / 2],
                     [self.imageSize[0] / 2 - 440, self.imageSize[1] / 2]], dtype=np.float32)

        transMat = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        imgTrans = cv2.warpPerspective(img, transMat, self.imageSize)
        cv2.imwrite('road_test_1_result_cropped_trans.jpg', imgTrans)
        return imgTrans
