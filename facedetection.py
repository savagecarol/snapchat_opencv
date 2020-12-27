import cv2

cap = cv2.VideoCapture(0)

cascade = cv2.CascadeClassifier('face.xml')


while cap.isOpened():
    result , frame = cap.read()
    if result:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces =cascade.detectMultiScale(gray ,1.3 , 5 , 0 , minSize=(120,120) , maxSize=(350,350))
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0)  , 5)
        cv2.imshow("frame" , frame)
        if cv2.waitKey(1)== ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
