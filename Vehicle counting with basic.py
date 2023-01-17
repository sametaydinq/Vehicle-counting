import cv2
import numpy as np


# GÖrüntü üzerine contourler uygulayarak araç takibi belirleme
# hata ihtimali çok fazla sadece mantığını görmek amaçlı yaptım
# daha sonra yolo modeli kullanarak sayaç yapacağım



cap = cv2.VideoCapture("video.mp4")

min_width_rect = 80 
min_hight_rect = 80
counter_line_position = 500
algoritma = cv2.bgsegm.createBackgroundSubtractorMOG()



def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 6 # allowable error between pixel
counter= 0



while True:
    ret,frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)

    img_sub = algoritma.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)

    counterSahpe , h= cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    cv2.line(frame1,(50,counter_line_position),(1250,counter_line_position),(255,0,255),3)


    for(i,c) in enumerate(counterSahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width_rect ) and (h>= min_hight_rect)
        if not val_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
        
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(255,0,0),-1)

        for(x,y) in detect:
            if y<(counter_line_position + offset) and  y>(counter_line_position - offset):
                counter+=1
            cv2.line(frame1,(50,counter_line_position),(1250,counter_line_position),(127,55,255),3)
            detect.remove((x,y))
            print("Arac sayisi "+str(counter))

    cv2.putText(frame1,"Arac sayisi :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0))

    cv2.imshow("test",dilatada)
    cv2.imshow("Video Original",frame1)

    if cv2.waitKey(1) == 13:
        break



cv2.destroyAllWindows()
cap.release()



