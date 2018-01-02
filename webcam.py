import dlib
import cv2
import argparse
import _pickle as pickle
import numpy as np
from cyvlfeat.hog import hog
from config import FER2013, CKPLUS
from data_loader import landmark_feats

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="fer2013", help="dataset")
args = parser.parse_args()

if args.dataset == FER2013.name:
    model_name = 'fer2013_model.bin'
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
elif args.dataset == CKPLUS.name:
    model_name = 'ckplus_model.bin'
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
else:
    model_name = 'saved_model.bin'
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


with open(model_name, 'rb') as f:
    clf = pickle.load(f)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face_width = 48
face_height = 48
cell = 3
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
        continue
    # for all detected face
    for k, d in enumerate(detections):
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        crop = image[d.top():d.top() + h, d.left():d.left() + w]
        frame = cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
        crop = cv2.resize(crop, (face_width, face_height), interpolation=cv2.INTER_LINEAR)
        hog_image = hog(crop, cell)
        landmarks_vectorised = landmark_feats(crop, d, predictor, args.dataset)
        hog_feats = np.reshape(hog_image, [1, hog_d])
        feats = np.concatenate([landmarks_vectorised, hog])
        conf = clf.predict_proba(feats)
        label = np.argmax(conf[0, :])
        idx = np.argmax(conf[0, :])
        cv2.putText(frame, emotions[idx], (d.left(), d.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3,
                    cv2.LINE_AA)

    cv2.imshow('webcam', frame)
    key = cv2.waitKey(10)
    if key in [27, ord('Q'), ord('q')]:  # exit on ESC
        break
cap.release()
cv2.destroyAllWindows()