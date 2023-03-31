import mediapipe as mp
import cv2
import time
import mouse
import sys



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
ring_finger_tip=0
index_finger_mcp=0
index_finger_tip = 0
middle_finger_tip=0
thumb_tip = 0
calibration=0
calibrationText=0
duration=0
stopCount=0
distanceHold1 = 0
distanceHold2 = 0
distanceHold3 = 0
distanceClick1 = 0
distanceClick2 = 0
distanceClick3 = 0
distanceStop1 = 0
distanceStop2 = 0
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55) as hands:
    
    
    while cap.isOpened():
        calibreClick=((distanceClick2+distanceClick3)/2)
        calibreHold=((distanceHold2+distanceHold3)/2)+5
        calibreStop = ((distanceStop1+distanceStop2)/2)+4
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        prev=time.time()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand annotations on the image.
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
        
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
            cur=time.time()
            duration+=(cur-prev)
        if not results.multi_hand_landmarks:
            continue
        prev=time.time()
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        
        for hand_landmarks in results.multi_hand_landmarks:
        
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_mcp= hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            middle_finger_tip= hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            
            rescaled_ring_finger_tip = (int(ring_finger_tip.x * image_width),int(ring_finger_tip.y * image_height))
            rescaled_index_finger_tip = (int(index_finger_tip.x * image_width),int(index_finger_tip.y * image_height))
            rescaled_thumb_tip = (int(thumb_tip.x * image_width), int(thumb_tip.y * image_height))
            rescaled_middle_finger_tip = (int(middle_finger_tip.x * image_width), int(middle_finger_tip.y * image_height))
            distanceStop = ((rescaled_ring_finger_tip[0] - rescaled_thumb_tip[0])**2 + (rescaled_ring_finger_tip[1] - rescaled_thumb_tip[1])**2)**0.5
            distanceHold = ((rescaled_index_finger_tip[0] - rescaled_thumb_tip[0])**2 + (rescaled_index_finger_tip[1] - rescaled_thumb_tip[1])**2)**0.5
            distanceClick= ((rescaled_index_finger_tip[0] - rescaled_middle_finger_tip[0])**2 + (rescaled_index_finger_tip[1] - rescaled_middle_finger_tip[1])**2)**0.5
            
            #x =1980-(int(index_finger_tip.x * image_width)*4)
            
            y=int((index_finger_mcp.y*1700)-400)
            
            x =2000-(int(index_finger_mcp.x * image_width)*5)
            
        
            if(calibration==0):
                
                
                if(int(duration)==4):
                    distanceClick2=distanceClick
                if(int(duration)==5):
                    distanceClick3=distanceClick
                
                    calibration=1
                    calibrationText=1
                
                    duration=0
                duration+=0.02
                break
            if(calibration==1):
                
                
                
                if(int(duration)==4):
                    distanceHold2=distanceHold
                
                if(int(duration)==5):
                    distanceHold3=distanceHold
                
                    calibration=2
                    calibrationText=2
                    duration=0
                duration+=0.02
                break
            if(calibration==2):
                if(int(duration)==4):
                    distanceStop1=distanceStop
                if(int(duration)==5):
                    distanceStop2=distanceStop
                    
                    calibration=3
                duration+=0.02
                break
            else:  
                
                
                mouse.move(x,y,absolute=True,duration=0.02)
                z = ((index_finger_mcp.z)*-1000)
                
                
                
                
                if(distanceClick<calibreClick):
                    mouse.click("left")
                    
                if(distanceHold<calibreHold):
                    mouse.hold("left")
                if(distanceHold>calibreHold+20):
                    mouse.release("left")
                    
                
                if(distanceStop<calibreStop):
                    if(stopCount==10):
                        sys.exit()
                    stopCount+=1
                
                
                #mouse.move(x,y,absolute=True,duration=0.07)
                
            
                
        
            
            
            mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            cur=time.time()
            duration+=(cur-prev)
            duration+=0.02
        
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        image=cv2.flip(image, 1)
        if(calibrationText==0):
        
            font=cv2.FONT_HERSHEY_SIMPLEX
            seconds=int(duration)
            cv2.putText(image,"  Please Combine Index Finger and Middle Finger",org=(30,100),fontFace=font,fontScale=0.7,color=(48,255,0),thickness=2,lineType=cv2.LINE_AA)
            cv2.putText(image,str(seconds),org=(300,130),fontFace=font,fontScale=0.7,color=(48,255,0),thickness=2,lineType=cv2.LINE_AA)
        
        if(calibrationText==1):
        
            font=cv2.FONT_HERSHEY_SIMPLEX
            seconds=int(duration)
            cv2.putText(image,"  Please Combine Thumb and Index Finger",org=(30,100),fontFace=font,fontScale=0.7,color=(48,255,0),thickness=2,lineType=cv2.LINE_AA)
            cv2.putText(image,str(seconds),org=(300,130),fontFace=font,fontScale=0.7,color=(48,255,0),thickness=2,lineType=cv2.LINE_AA)
            if(int(duration)==5):calibrationText=2
        if(calibrationText==2):
        
            font=cv2.FONT_HERSHEY_SIMPLEX
            seconds=int(duration)
            cv2.putText(image,"  Please Combine Thumb and Ring Finger",org=(30,100),fontFace=font,fontScale=0.7,color=(48,255,0),thickness=2,lineType=cv2.LINE_AA)
            cv2.putText(image,str(seconds),org=(300,130),fontFace=font,fontScale=0.7,color=(48,255,0),thickness=2,lineType=cv2.LINE_AA)
            if(int(duration)==5):calibrationText=3
        cv2.imshow('MediaPipe Hands',image )
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()