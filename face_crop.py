import numpy as np
import cv2
import sys
from pathlib import Path
import os

# Get user supplied values
#imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

directory = Path("C:\\Path\to\originals")
savedir = Path("C:\\Path\to\save")
faildir = Path("C:\\Path\for\fails")
assert(directory.is_dir())

for filename in directory.iterdir():
    if filename.is_file() and str(filename).lower().endswith(('.png', '.jpg', '.jpeg')): 

            # Read the image
            image = cv2.imread(str(filename))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image - change values to adjust sensitivity
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(500, 500)
            )

            print("Found {0} faces!".format(len(faces)))

            # Draw a rectangle around the faces
            #for (x, y, w, h) in faces:
             #  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 10)

            if len(faces)==1:
                Pad = int(faces[0][2] * 0.5)
                Voffset = int(faces[0][3] * 0.1)
            elif len(faces)==0:
                savename = faildir.joinpath(filename.stem + filename.suffix)
                print("FAILED!")
                print(savename)
                cv2.imwrite(str(savename),image)
                continue
            else:
                Pad = (int(np.mean([item[3] for item in faces])*0.3))
                Voffset = 0

            BoundLeft=min([item[0] for item in faces])-Pad
            if BoundLeft<=0: BoundLeft=0
            BoundRight=max([item[0]+item[2] for item in faces])+Pad
            if BoundRight>=image.shape[1]: BoundRight=image.shape[1]
            BoundTop=min([item[1] for item in faces])-Pad+Voffset
            if BoundTop<=0: BoundTop=0
            BoundBot=max([item[1]+item[3] for item in faces])+Pad+Voffset
            if BoundBot>=image.shape[0]: BoundBot=image.shape[0]


            BoundWidth = BoundRight - BoundLeft
            BoundHeight = BoundBot - BoundTop
            BoundCentre = BoundLeft + int((BoundWidth)/2)
            BoundMiddle = BoundTop + int((BoundHeight)/2)


            if BoundWidth > BoundHeight:
                BoundTop = BoundMiddle - int(BoundWidth/2)
                BoundBot = BoundMiddle + int(BoundWidth/2)
                if BoundTop<=0:
                    BoundTop = 0
                    BoundBot = BoundWidth
                if BoundBot>=image.shape[0]:
                    BoundBot = image.shape[0]
                    BoundTop = image.shape[0] - BoundHeight
                    
            elif BoundWidth < BoundHeight: 
                BoundLeft = BoundCentre - int(BoundHeight/2)
                BoundRight = BoundCentre + int(BoundHeight/2)
                if BoundLeft<=0:
                    BoundLeft = 0
                    BoundRight = BoundWidth
                if BoundRight>=image.shape[1]:
                    BoundRight = image.shape[1]
                    BoundLeft = image.shape[1] - BoundHeight


            #cv2.rectangle(image, (BoundLeft, BoundTop), (BoundRight, BoundBot), (255, 0, 0), 10)
            #cropped = image[BoundTop:BoundBot,BoundLeft:BoundRight]
            #cropped = cv2.resize(cropped, (1000, 1000))
            cropped = cv2.resize((image[BoundTop:BoundBot,BoundLeft:BoundRight]), (1000, 1000))

            #cv2.imshow("Faces found", image)
            #cv2.imshow("Cropped", cropped)
            #print(cropped.shape)
            #cv2.waitKey(0)

            savename = savedir.joinpath(filename.stem + filename.suffix)
            print(savename)
            cv2.imwrite(str(savename),cropped)
            
            continue
    else:
        continue

    cv2.destroyAllWindows() 
