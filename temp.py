# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2 
import argparse
import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from espeak import espeak
ap=argparse.ArgumentParser()
ap.add_argument("-p","--prototxt",required=True)
ap.add_argument("-m","--model",required=True)
ap.add_argument("-c","--confidence",type=float,default=0.2)
args=vars(ap.parse_args())
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])
vs=VideoStream(src=0).start()
time.sleep(2)
fps=FPS().start()
while True:
    frm=vs.read()
    frm=imutils.resize(frm,width=400)
    (h,w)=frm.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frm,(300,300)),0.007843,(300,300),127.5)
    net.setInput(blob)
    detect=net.forward()
    for i in np.arange(0,detect.shape[2]):
        confidence=detect[0,0,i,2]
        if(confidence>args["confidence"]):
            idx=int(detect[0,0,i,1])
            bound = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startx,starty,endx,endy)= bound.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
            obj="{}".format(CLASSES[idx])
            print("A {} is in front of you".format(label))
            espeak.synth("A {} is in front of you".format(obj))
            if(obj=="person"):
                espeak.synth("Say Hello")
            if(obj=="dog" or obj=="cat"):
                espeak.synth("Move around it")
            #if(obj=="car" or "train" or "bus"):
             #   espeak.synth("Careful while crossing")
            time.sleep(3)
    key = cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break
    fps.update()
fps.stop()
cv2.destroyAllWindows()
vs.stop()


        