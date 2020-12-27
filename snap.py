import cv2

cap = cv2.VideoCapture(0)

cascade = cv2.CascadeClassifier('face.xml')

specs_ori = cv2.imread("glass.png" , -1)
cigar_ori = cv2.imread("cigar.png" , -1)
mus_ori = cv2.imread("mustache.png" , -1)

def transparentOverlay(src, overlay , pos = (0,0)  , scale = 1):
    overlay = cv2.resize(overlay , (0,0) ,fx = scale , fy = scale)
    h , w , _ =  overlay.shape ## size of foreground image
    rows , cols , _ = src.shape  ## size of background image
    y , x = pos[0] , pos [1]

    for i in range(h):
        for j in range(w):
            if x + i > rows or y + j >=cols:
                continue
            alpha = float(overlay[i][j][3]/255) ##  read the alpha chanel
            src[x+i][y+j] = alpha * overlay[i][j][:3] + (1-alpha) * src[x+i][y+j]
    return src


while cap.isOpened():
    result , frame = cap.read()
    if result:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces =cascade.detectMultiScale(gray ,1.3 , 5 , 0 , minSize=(120,120) , maxSize=(350,350))
        for (x,y,w,h) in faces:
            if h> 0 and w>0:
                glass_symin = int(y+ 1.5 * h/5)
                glass_symax = int(y + 2.5 * h / 5)
                sh_glass = glass_symax - glass_symin

                cigar_symin = int(y + 4 * h / 6)
                cigar_symax = int(y + 5.5 * h / 6)
                sh_cigar = cigar_symax - cigar_symin

                mus_symin = int(y + 3.5 * h / 6)
                mus_symax = int(y + 5 * h / 6)
                sh_mus = mus_symax - mus_symin

                face_glass_ori = frame [glass_symin:glass_symax , x:x+w]
                cigar_glass_ori = frame[cigar_symin:cigar_symax, x:x + w]
                mus_glass_ori = frame[mus_symin:mus_symax, x:x + w]

                glass = cv2.resize(specs_ori , (w , sh_glass) , interpolation= cv2.INTER_CUBIC)
                cigar = cv2.resize(cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_CUBIC)
                mus = cv2.resize(mus_ori, (w, sh_mus), interpolation=cv2.INTER_CUBIC)

                transparentOverlay(face_glass_ori  , glass)
                transparentOverlay(mus_glass_ori , mus)


        cv2.imshow("frame" , frame)
        if cv2.waitKey(1)== ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
