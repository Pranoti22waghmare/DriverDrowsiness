import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound


class DrowsyDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('drowsiness_predicter/shape_predictor_68_face_landmarks.dat')

        self.sleep = 0
        self.drowsy = 0
        self.active = 0
        self.status = ""
        self.color =(0,0,0)

        self.delay = 5000

        self.cap = cv2.VideoCapture(0)

    def compute(self, ptA, ptB):
        dist = np.linalg.norm(ptA-ptB)
        return dist
    
    def blinked(self, a, b, c, d, e, f):
        self.up = self.compute(b, d) + self.compute(c, e)
        self.down = self.compute(a, f)
        self.ratio = self.up / (2.0 * self.down)
        if(self.ratio > 0.25):
            return 2
        elif(self.ratio > 0.21 and self.ratio <= 0.25):
            return 1
        else :
            return 0
        
    def run(self):
        while True:
            _, frame = self.cap.read()
            gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                face_frame = frame.copy()

                cv2.rectangle(face_frame, (x1,y1) , (x2,y2) , (0,255,0) , 2)
                landmarks =  self.predictor(gray,face)
                landmarks = face_utils.shape_to_np(landmarks)
                left_blink = self.blinked(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
                right_blink = self.blinked(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])


                if(left_blink == 0 or right_blink == 0):
                    self.sleep += 1
                    self.drowsy = 0
                    self.active = 0
                    if(self.sleep > 12):
                        self.status = "SLEEPING !!!"
                        self.color = (255,0,0)
                        winsound.Beep(2500, 1000)

                elif(left_blink == 1 or right_blink == 1):
                    self.sleep = 0
                    self.active = 0
                    self.drowsy += 1
                    if(self.drowsy > 12):
                        self.status = "DROWSY !"
                        self.color = (0,0,255)
                        winsound.Beep(2500, 1000)


                else:
                    self.sleep = 0
                    self.drowsy = 0
                    self.active += 1
                    if(self.active > 6):
                        self.status = "Active :)"
                        self.color = (0,255,0)

                cv2.putText(frame, self.status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color,3)

                for n in range(0, 68):
                    (x,y) = landmarks[n]
                    cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
                    
                cv2.imshow("Frame", frame)
                cv2.imshow("Result of detector", face_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        self.cap.release()
        cv2.destroyAllWindows()

            





