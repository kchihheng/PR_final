import dlib
import cv2
import argparse
import _pickle as pickle
import numpy as np
from cyvlfeat.hog import hog
from config import FER2013, CKPLUS
from data_loader import landmark_feats
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="fer2013", help="dataset")
args = parser.parse_args()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
THRESH1 = 0.7
THRESH2 = 0.4

if args.dataset == FER2013.name:
    model_name = 'fer2013_model.bin'
    face_width = FER2013.face_width
    face_height = FER2013.face_height
    cell = FER2013.cell
elif args.dataset == CKPLUS.name:
    model_name = 'ckplus_model.bin'
    face_width = CKPLUS.face_width
    face_height = CKPLUS.face_height
    cell = CKPLUS.cell



with open(model_name, 'rb') as f:
    clf = pickle.load(f)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

hog_d = int((face_height / cell) * (face_width / cell) * 31)

cap = cv2.VideoCapture(0)
cv2.namedWindow('webcam')

while(True):
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = clahe.apply(image)
    # detect the face
    detections = detector(image, 1)
    if len(detections) < 1:
        cv2.imshow('webcam', frame)
        key = cv2.waitKey(1)
        if key in [27, ord('Q'), ord('q')]:  # exit on ESC
            break
        continue
    # for all detected face
    for k, d in enumerate(detections):
        # box clipped
        if d.right() >= frame.shape[1]:
            right = frame.shape[1] - 1
        else:
            right = d.right()
        if d.bottom() >= frame.shape[1]:
            bottom = frame.shape[0] - 1
        else:
            bottom = d.bottom()
        if d.top() < 0:
            top = 0
        else:
            top = d.top()
        if d.left() < 0:
            left = 0
        else:
            left = d.left()
        d = dlib.rectangle(left, top, right, bottom)
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        crop = image[d.top():d.top() + h, d.left():d.left() + w]
        frame = cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
        crop = cv2.resize(crop, (face_width, face_height), interpolation=cv2.INTER_CUBIC)
        hog_image = hog(crop, cell)
        landmarks_vectorised = landmark_feats(image, d, predictor, frame.shape[1],frame.shape[0])
        hog_feats = np.reshape(hog_image, [1, hog_d])
        feats = np.concatenate([landmarks_vectorised, hog_feats], axis=1)
        conf = clf.predict_proba(feats)
        tmp_conf = conf[0, :].copy()
        idx = np.argmax(tmp_conf)
        if conf[0,idx] >= THRESH1:
            text = emotions[idx]
        else:
            text = emotions[-1]
        tmp_conf[idx] = -1
        cv2.putText(frame, text, (d.left(), d.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        idx = np.argmax(tmp_conf)
        if conf[0,idx] >= THRESH2:
            text = emotions[idx]
            cv2.putText(frame, text, (d.left(), d.top()-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)


    cv2.imshow('webcam', frame)
    key = cv2.waitKey(1)
    if key in [27, ord('Q'), ord('q')]:  # exit on ESC
        break
cap.release()
cv2.destroyAllWindows()