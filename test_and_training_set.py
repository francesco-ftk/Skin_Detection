import cv2
from PIL import Image
import numpy as np
import math
from sklearn.model_selection import train_test_split

def save_frame_from_video():

    Video=['1','2']
    ni=0
    no=0
    totalFrame=0

    print("Salvataggio frame da video: ")
    for video in Video:

        print("Inizio elaborazione video %s" % video)
        # Conta frame del video input e del video output corrispondente
        count=0
        cap=cv2.VideoCapture('Dataset/Video/%si.avi' % video)
        while(True):
            ret, frame= cap.read()
            if not ret:
                break
            count+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        frameI=count

        count=0
        cap=cv2.VideoCapture('Dataset/Video/%so.avi' % video)
        while(True):
            ret, frame= cap.read()
            if not ret:
                break
            count+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        frameO=count

        if(frameI<=frameO):
            min=frameI
        else:
            min=frameO

        # Salva i frame dei due video come immagini jpg
        cap=cv2.VideoCapture('Dataset/Video/%si.avi' % video)
        count=0
        while(count<min):
            ret, frame= cap.read()
            if not ret:
                break
            ni+=1
            cv2.imwrite("Dataset/frame_input/frame%d.jpg" % ni, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count+=1
        cap.release()

        cap=cv2.VideoCapture('Dataset/Video/%so.avi' % video)
        count=0
        while(count<min):
            ret, frame= cap.read()
            if not ret:
                break
            no+=1
            cv2.imwrite("Dataset/frame_output/frame%d.jpg" % no, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count+=1
        cap.release()

        totalFrame+=min
        print("Fine elaborazione video %s" % video)

    return totalFrame


def create_training_and_test_set(totalFrame):
    print("Creazione Training e Test Set in corso... ")
    count=1
    X=[]
    for i in range(totalFrame):
        im=Image.open("Dataset/frame_input/frame%d.jpg" % count)
        pix=im.load()
        col, row= im.size
        for j in range(col):
            for k in range(row):
                p=pix[j,k]
                p[0]=p[0]/255
                p[1]=p[1]/255
                p[2]=p[2]/255
                # Luminanza
                Y= 0.2126*p[0]+0.7152*p[1]+0.0722*p[2]
                # Satuazione
                S=max(p[0],p[1],p[2])-min(p[0],p[1],p[2])
                # Tonalità
                if(p[0]==p[1]==p[2]):
                    H1=90
                else:
                    H1=np.arccos((p[0]-0.5*p[1]-0.5*p[2])/(float)((p[0]**2+p[1]**2+p[2]**2-p[0]*p[1]-p[0]*p[2]-p[2]*p[1])**0.5))
                    H1=math.degrees(H1)
                if p[2]>p[1]:
                    H=360-H1
                else:
                    H=H1
                X.append((Y,S,H))
        count+=1
    count=1
    y=[]
    for i in range(totalFrame):
        im=Image.open("Dataset/frame_output/frame%d.jpg" % count)
        pix=im.load()
        col, row= im.size
        for j in range(col):
            for k in range(row):
                p=pix[j,k]
                if p[0]>248 and p[1]>248 and p[2]>248: # se bianco
                    y.append(1) # Skin
                else:
                    y.append(0) # Background
        count+=1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Training e Test Set creati")
    return X_train, X_test, y_train, y_test



