 #install mediapipe library for face,finger and other ML solutions for live and streaming media etc
from unittest import result
import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mpHands= mp.solutions.hands#passes parameter
hands = mpHands.Hands()  # hands gesture used and  has various parameters 
mpDraw= mp.solutions.drawing_utils # using for landmarks drawing structure 

# NOW CALCULATE FRAME PER SECOND
pTime=0 #FOR Previous time
cTime=0 # For Current time

while True:
    ret,img=cap.read()
    imgBGR=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgBGR)
    print(results.multi_hand_landmarks)# Landmarks is used for out finger postion check and where they perform.
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):# totall 21 ids and show in terminal in 0 x coordinate.(0 to 20ids show)
                print(id,lm)
            
                #now draw ids in circle landmarks
                
                h,w,c = img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                
                if id==4:# it shows the where is our 21th point are get used for start and ending point.
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)# To draw circle in every finger with there thickness radius and diamter with filled color 
                
            
            
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            
            
    cTime=time.time()
    fps=1/(cTime-pTime)   # this is the formula of frame per second which fps=1(ctime-ptime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) #used to text is media or image with also mention there color,font,thickness size etc.
            
    
    cv2.imshow('Image',img)
    if cv2.waitKey(1)==13:
        break
    
cv2.destroyAllWindows()    



