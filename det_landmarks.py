import cv2
import dlib


def set_landmarks(img_gray, img, face, predictor, detector):
    landmarks = predictor(img_gray, face)
    landmarks_list = []
    for i in range(0, landmarks.num_parts):
        landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))
    # for each landmark, plot and write number
    for landmark_num, xy in enumerate(landmarks_list, start = 1):
        cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
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


