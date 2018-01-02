import cv2
import os
import numpy as np
import dlib
import argparse
import time
import _pickle as pickle
from sklearn import svm
from sklearn import calibration
from config import FER2013, CKPLUS
from data_loader import get_data_fer2013, get_data_ckplus
import pdb

image_height = 48
image_width = 48
OUTPUT_FOLDER_NAME = "fer2013"
SAVE_IMAGES = False
model_name="ckplus_model.bin"
emotions = ['Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral']




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jpg", default="no", help="save images as .jpg files")
    parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
    parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
    parser.add_argument("-i", "--iter", default=10000, type=int, help="Maximum number of evaluations during hyperparameters search")
    parser.add_argument("-c", "--convert", default="no", help="Convert dataset")
    parser.add_argument("-d", "--dataset", default="fer2013", help="dataset")
    parser.add_argument("-emo", "--expressions", default=[0,1,2,3,4,5,6, 7], type=list,
                        help="choose the faciale expression you want to use: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")
    args = parser.parse_args()

    if args.jpg == "yes" or args.jpg == "Yes" or args.jpg == "YES":
        SAVE_IMAGES = True
    if args.train == "yes" or args.train == "Yes" or args.train == "YES":
        train = True
    else:
        train = False
    if args.evaluate == "yes" or args.evaluate == "Yes" or args.evaluate == "YES":
        test = True
    else:
        test = False
    if args.convert == "yes" or args.convert == "Yes" or args.convert == "YES":
        convert = True
    else:
        convert = False

    #setting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    if args.dataset == FER2013.name:
        original_labels = [0, 1, 2, 3, 4, 5, 6]
    elif args.dataset == CKPLUS.name:
        original_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    new_labels = list(set(original_labels) & set(args.expressions))
    if convert:
        if args.dataset == FER2013.name:
            get_data_fer2013(clahe, detector, predictor, new_labels, SAVE_IMAGES)
        elif args.dataset == CKPLUS.name:
            get_data_ckplus(clahe, detector, predictor, new_labels, SAVE_IMAGES)

    if train:
    #for C in [1.5*1e-3, 3e-3, 4.5*1e-3, 6*1e-3, 7.5*1e-3, 9e-3]:
        print("building model...")
        clf = svm.LinearSVC(C=0.1, random_state=0, tol=1e-4, dual=False)
        # CK+: C=0.1 or 0.01
        # fer2013: C=1e-3
        clf = calibration.CalibratedClassifierCV(clf, method='sigmoid', cv=5)
        if args.dataset == FER2013.name:
            OUTPUT_FOLDER_NAME = FER2013.name
            with open(OUTPUT_FOLDER_NAME + '/Training/landmarks_feats.pkl', 'rb') as f:
                feats_data= pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/Training/hog_feats.pkl', 'rb') as f:
                hog_feats = pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/Training/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            feats_data = np.concatenate([feats_data, hog_feats], axis=1)

            with open(OUTPUT_FOLDER_NAME + '/PrivateTest/landmarks_feats.pkl', 'rb') as f:
                feats_data2 = pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/PrivateTest/hog_feats.pkl', 'rb') as f:
                hog_feats2 = pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/PrivateTest/labels.pkl', 'rb') as f:
                labels2 = pickle.load(f)
            feats_data2 = np.concatenate([feats_data2, hog_feats2], axis=1)
        elif args.dataset == CKPLUS.name:
            OUTPUT_FOLDER_NAME = CKPLUS.name
            with open(OUTPUT_FOLDER_NAME + '/landmarks_feats.pkl', 'rb') as f:
                feats_data = pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/hog_feats.pkl', 'rb') as f:
                hog_feats = pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            number = feats_data.shape[0]
            train_num = int(number * CKPLUS.ratio[0])
            val_num = int(number * CKPLUS.ratio[1])
            #pdb.set_trace()
            feats_data2 = np.concatenate(
            [feats_data[train_num:train_num + val_num, :], hog_feats[train_num:train_num + val_num, :]], axis=1)
            labels2 = labels[train_num:train_num + val_num]
            feats_data = np.concatenate([feats_data[0:train_num, :], hog_feats[0:train_num, :]], axis=1)
            labels = labels[0:train_num]

        print("start training...")
        start_time = time.time()
        clf.fit(feats_data, labels)
        training_time = time.time() - start_time
        print("training time = {0:.1f} sec".format(training_time))
        print("saving model...")
        with open(model_name, 'wb') as f:
            pickle.dump(clf, f)
        accuracy = clf.score(feats_data2, labels2)
        print("  - testing accuracy = {0:.1f}".format(accuracy * 100))

    if test:
        print("start evaluation...")
        if not train:
            print("loading pretrained model...")
            if os.path.isfile(model_name):
                with open(model_name, 'rb') as f:
                    clf = pickle.load(f)
            else:
                print("Error: file '{}' not found".format(model_name))
                exit()
        if args.dataset == FER2013.name:
            OUTPUT_FOLDER_NAME = FER2013.name
            with open(OUTPUT_FOLDER_NAME + '/PublicTest/landmarks_feats.pkl', 'rb') as f:
                feats_data= pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/PublicTest/hog_feats.pkl', 'rb') as f:
                hog_feats = pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/PublicTest/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            feats_data = np.concatenate([feats_data, hog_feats], axis=1)
        elif args.dataset == CKPLUS.name:
            OUTPUT_FOLDER_NAME = CKPLUS.name
            with open(OUTPUT_FOLDER_NAME + '/landmarks_feats.pkl', 'rb') as f:
                feats_data = pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/hog_feats.pkl', 'rb') as f:
                hog_feats = pickle.load(f)
            with open(OUTPUT_FOLDER_NAME + '/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            number = feats_data.shape[0]
            trainval_num = int(number * (CKPLUS.ratio[0]+CKPLUS.ratio[1]))
            feats_data = np.concatenate([feats_data[trainval_num:, :], hog_feats[trainval_num:, :]], axis=1)
            labels = labels[trainval_num:]
        conf = clf.predict_proba(feats_data)
        pdb.set_trace()
        accuracy = clf.score(feats_data, labels)
        print("  - validation accuracy = {0:.1f}".format(accuracy * 100))
