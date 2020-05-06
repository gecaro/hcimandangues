import cv2
import dlib
import math 




def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  

def set_landmarks(img_gray, img, face, predictor, detector):
    landmarks = predictor(img_gray, face)
    landmarks_list = []
    for i in range(0, landmarks.num_parts):
        landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))


    # for each landmark, plot and write number

    ##Jump rñhjslvhpdvslghslgslh
    factor_j = 0
    jump = False
    distancia_jump = calculateDistance(landmarks.part(41).x,landmarks.part(41).y,landmarks.part(19).x,landmarks.part(19).y)   


    ##fireball borjborbhrobhrohb
    factor_f = 0
    fireball = False
    distanica_fireball= calculateDistance(landmarks.part(51).x,landmarks.part(51).y,landmarks.part(57).x,landmarks.part(57).y) 

    ##shield
    factor_s = 0
    shield = False
    distancia_shield = calculateDistance(landmarks.part(48).x,landmarks.part(48).y,landmarks.part(54).x,landmarks.part(54).y) 


    if distancia_jump>35.0:
       #print("JUMP");
        jump = True
        factor_j = 1

    elif distanica_fireball>32.0:
       #print("FIREBALL");
        fireball = True
        factor_f = 1

    elif distancia_shield>65.0:
        #print("SHIELD");                                      
        shield = True
        factor_s = 1    


    for landmark_num, xy in enumerate(landmarks_list, start = 1):
        cv2.circle(img, (xy[0], xy[1]), 5, (100*factor_j, 100*factor_s, 100*factor_f), -1)
        cv2.putText(img, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)


def face_detection(img, predictor, detector):
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces in the image
    faces_in_image = detector(img_gray, 0)
    for face in faces_in_image:
        set_landmarks(img_gray, img, face, predictor, detector)

def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_webcam(detector, predictor):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        face_detection(img, predictor, detector)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    # set up the 68 point facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    img = cv2.imread('data/face.jpg', 1)
    #face_detection(img, predictor, detector)
    #show_img(img)
    show_webcam(detector, predictor)


if __name__ == "__main__":
    main()


