import cv2
import urllib
import numpy as np
import time

stream = urllib.urlopen('http://192.168.2.188:8080/video?x.mjpeg')
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
bytes = bytes()
while True:
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        count = 0
        if(len(faces)!=0):
            for (x,y,w,h) in faces:
                cv2.rectangle(i,(x,y),(x+w,y+h),(255,0,0),2)
                count = count+1
        cv2.putText(i,"Person Count="+str(count),(10,255), font, 1,(255,0,0),2)
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 640, 480)
        cv2.imshow('frame',i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
        
