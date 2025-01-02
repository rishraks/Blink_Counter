import cv2
import mediapipe as mp
import math
import cvzone
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,360)

mp_detect_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

face = mp_face_mesh.FaceMesh(max_num_faces = 1, min_detection_confidence =0.5, min_tracking_confidence = 0.5, refine_landmarks = True)

left_eye_ids = set(sum(mp_face_mesh.FACEMESH_LEFT_EYE, ())) 
print("Left Eye Landmark IDs:", left_eye_ids)

plot_y = LivePlot(640,360,[0.09,0.35])

# Upper = 386
# Lower = 374
# outer = 362
# Inner = 263

threshold_close = 0.2
threshold_open = 0.29
is_blinking = False
count=0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = face.process(frame_rgb)
        h,w,c = frame.shape
        
        if results.multi_face_landmarks:
            for facelms in results.multi_face_landmarks:
                for landmark in left_eye_ids:
                    lms = facelms.landmark[landmark]
                    x,y = int(lms.x*w), int(lms.y*h)
                    cv2.circle(frame, (x,y),1,(0,0,255),-1)
                    
                upper_id = 386
                lower_id = 374
                inner_id = 263
                outer_id = 362
                
                upper_point = facelms.landmark[upper_id]
                upper_pos = int(upper_point.x*w), int(upper_point.y*h)
                
                lower_point = facelms.landmark[lower_id]
                lower_pos = int(lower_point.x*w), int(lower_point.y*h)
                
                inner_point = facelms.landmark[inner_id]
                inner_pos = int(inner_point.x*w), int(inner_point.y*h)
                
                outer_point = facelms.landmark[outer_id]
                outer_pos = int(outer_point.x*w), int(outer_point.y*h)
        
                vertical_distance = math.sqrt((upper_pos[0]-lower_pos[0])**2+(upper_pos[1]-lower_pos[1])**2)
                horizontal_distance = math.sqrt((outer_pos[0]-inner_pos[0])**2+(outer_pos[1]-inner_pos[1])**2)
                
                normalized_distance = (vertical_distance/horizontal_distance)
                
                if normalized_distance<threshold_close and not is_blinking:
                    is_blinking = True
                elif normalized_distance >threshold_open and is_blinking:
                    count+=1
                    is_blinking = False
                    
                cv2.line(frame, upper_pos,lower_pos,(0,200,0),1)
                cv2.line(frame, inner_pos,outer_pos,(200,200,0),1)
                
                
                plot = plot_y.update(normalized_distance)
                stack_frame = cvzone.stackImages([frame,plot],2,1)
                
                
        cv2.putText(stack_frame, f"Blink Count: {count}",(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
        cv2.imshow("Frame",stack_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()