import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/extra"
counter = 0

while True:
    success, img = cap.read()
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
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap,:] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1

        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        #    raise Exception("Could not write image")
        print(counter)

if cv2.waitKey(10) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

