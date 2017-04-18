import cv2
import urllib
import numpy as np

stream = urllib.urlopen('http://192.168.2.188:8080/video?x.mjpeg')
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
bytes = bytes()

Id=raw_input('enter your id')
sampleNum=0
while(True):
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        if(len(faces)!=0):
            for (x,y,w,h) in faces:
                cv2.rectangle(i,(x,y),(x+w,y+h),(255,0,0),2)
        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder
                cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
                cv2.resizeWindow("frame", 640, 480)
                cv2.imshow('frame',i)
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 20
        elif sampleNum>20:
            break
cv2.destroyAllWindows()