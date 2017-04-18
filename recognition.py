import cv2
import numpy as np
import urllib

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


stream = urllib.urlopen('http://192.168.2.188:8080/video?x.mjpeg')
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
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
        for(x,y,w,h) in faces:
            cv2.rectangle(i,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf<50):
                if(Id==1):
                    Id="Bharath"
                elif(Id==2):
                    Id="Sam"
            else:
                Id="Unknown"
            cv2.cv.PutText(cv2.cv.fromarray(i),str(Id), (x,y+h),font, 255)
        cv2.imshow('im',i) 
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
cv2.destroyAllWindows()