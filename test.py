import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from pathlib import Path

data_folder = Path("C:/Users/kbinw/Binwant/01 IGDTUW/8th Sem/Final YR Project/Code 4/Sign_lang_project/")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
trained_model = Classifier(data_folder/"Models/InceptionV2Model.h5")

offset = 20
imgSize = 300
folder = "Data"
counter = 0
labels = ['A','B','C','D','E','F','G','H','I','K', 'L', 'M','N','O','P','Q','R','S','T','U','V','W','X','Y']

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        imgH, imgW, imgC = imgCrop.shape
        if imgH > 0 and imgW > 0 and imgC > 0:
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = trained_model.getPrediction(imgWhite, draw=True)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = trained_model.getPrediction(imgWhite, draw=False)

            #cv2.rectangle(imgOutput, (x - offset+90, y - offset-50), (x-offset+50, y - offset-50+50), (255, 0, 255), cv2.FILLED)

            cv2.putText(imgOutput, labels[index], (x,y-26), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)


            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)




