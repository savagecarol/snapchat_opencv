import cv2

cap = cv2.VideoCapture(0)
glass = cv2.imread("glass.png")
cigar = cv2.imread("cigar.png")
mustache = cv2.imread("mustache.png")


face_cascade = cv2.CascadeClassifier('face.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5 , 0, (120, 120), (350, 350))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("detect",frame)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

