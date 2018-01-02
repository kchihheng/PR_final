import cv2
import pandas as pd
import os
import errno
import math
import numpy as np
import dlib
import _pickle as pickle
from glob import glob
from config import FER2013, CKPLUS
from cyvlfeat.hog import hog
#import pdb

def landmark_feats(image, d, predictor, dataset):
    if dataset == 'fer2013':
        image_height = FER2013.height
        image_width = FER2013.width
    elif dataset == 'ckplus':
        image_height = CKPLUS.height
        image_width = CKPLUS.width
    # box clipped
    if d.right() >= image_width:
        right = image_width - 1
    else:
        right = d.right()
    if d.bottom() >= image_height:
        bottom = image_height - 1
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
    rect = dlib.rectangle(left, top, right, bottom)
    # detect landmarks
    shape = predictor(image, rect)  # Draw Facial Landmarks with the predictor class
    # scale factor for the face
    xscale = (rect.right() - rect.left() + 1) / image_width
    yscale = (rect.bottom() - rect.top() + 1) / image_height
    # get normalized landmarks coordinates
    xlist = []
    ylist = []
    for ind in range(0, 68):
        xlist.append(float(shape.part(ind).x) / xscale)
        ylist.append(float(shape.part(ind).y) / yscale)
    # find the static point on the face
    xc = xlist[39] + (xlist[42] - xlist[39]) / 2
    yc = ylist[39] + (ylist[42] - ylist[39]) / 2
    # normalized by static point
    xcentral = [(x - xc) for x in xlist]
    ycentral = [(y - yc) for y in ylist]
    # find the rotate angle
    if ycentral[39] == ycentral[42]:
        rotate = 0
    else:
        rotate = math.atan2(ycentral[42], xcentral[42])
    # store the mean in the buffer
    landmarks_vectorised = [np.mean(xcentral), np.mean(ycentral)]
    # store each point information
    for x, y in zip(xcentral, ycentral):
        landmarks_vectorised.append(x)
        landmarks_vectorised.append(y)
        dist = np.sqrt(x ** 2 + y ** 2)
        landmarks_vectorised.append(dist)
        angle = math.atan2(y, x) + rotate
        if angle > math.pi:
            angle = angle - 2 * math.pi
        if angle < math.pi:
            angle = angle + 2 * math.pi
        landmarks_vectorised.append(angle)
    landmarks_vectorised = np.asarray(landmarks_vectorised)
    landmarks_vectorised = np.reshape(landmarks_vectorised, [1, 274])
    return landmarks_vectorised

# feature extractor
def get_data_fer2013(clahe, detector, predictor, selected_label, save_images):
    OUTPUT_FOLDER_NAME = FER2013.name
    image_height = FER2013.height
    image_width = FER2013.width
    cell = FER2013.cell
    ori_labels = [0, 1, 2, 3, 4, 5, 6]
    #new_labels = list(set(ori_labels) & set(selected_label))

    print("importing csv file")
    data = pd.read_csv(OUTPUT_FOLDER_NAME+'/fer2013.csv')

    for category in data['Usage'].unique():
        print("converting fer2013 set: " + category + "...")
        # create folder
        if not os.path.exists(category):
            try:
                os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)
            except OSError as e:
                if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
                    pass
                else:
                    raise

        # get samples and labels of the actual category
        category_data = data[data['Usage'] == category]
        samples = category_data['pixels'].values
        labels = category_data['emotion'].values

        # initialize
        feats_data = np.zeros([0, 274])
        feats_labels = []
        hog_d = int((image_height/cell) * (image_width/cell) * 31)
        hog_feats = np.zeros([0, hog_d])

        # get images and extract features
        for i in range(len(samples)):
            if labels[i] in selected_label:
                image = np.fromstring(samples[i], dtype=np.uint8, sep=" ").reshape((image_height, image_width))
                # save images
                if save_images:
                    cv2.imwrite(OUTPUT_FOLDER_NAME + '/'+ category + '/' + str(i) + '.jpg', image)
                # image pre-processing
                image = cv2.imread(OUTPUT_FOLDER_NAME + '/'+ category + '/' + str(i) + '.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = clahe.apply(image)
                # detect the face
                detections = detector(image, 1)
                if len(detections) < 1:
                    continue
                # extract HOG features
                hog_image = hog(image, cell)
                #for all detected face
                for k, d in enumerate(detections):
                    landmarks_vectorised = landmark_feats(image, d, predictor, FER2013.name)
                    # concat feature
                    feats_data = np.concatenate([feats_data, landmarks_vectorised], axis=0)
                    hog_feats = np.concatenate([hog_feats, np.reshape(hog_image, [1, hog_d])],axis=0)
                    feats_labels.append(labels[i])
        print("saving features and labels...")
        with open(OUTPUT_FOLDER_NAME + '/'+ category + '/landmarks_feats.pkl', 'wb') as f:
            pickle.dump(feats_data, f)
        with open(OUTPUT_FOLDER_NAME + '/'+ category + '/hog_feats.pkl', 'wb') as f:
            pickle.dump(hog_feats, f)
        with open(OUTPUT_FOLDER_NAME + '/'+ category + '/labels.pkl', 'wb') as f:
            pickle.dump(feats_labels, f)

def get_data_ckplus(clahe, detector, predictor, selected_label, save_images):
    OUTPUT_FOLDER_NAME = CKPLUS.name
    face_height = CKPLUS.face_height
    face_width = CKPLUS.face_width
    cell = CKPLUS.cell
    ori_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    #new_labels = list(set(ori_labels) & set(selected_label))

    # initialize
    feats_data = np.zeros([0, 274])
    feats_labels = []
    hog_d = int((face_height / cell) * (face_width / cell) * 31)
    hog_feats = np.zeros([0, hog_d])
    dirs = os.listdir(OUTPUT_FOLDER_NAME+'/emotion')
    print("converting CK+...")
    for dir in dirs:
        video_dirs = os.listdir(OUTPUT_FOLDER_NAME+'/emotion/'+dir)
        for video_dir in video_dirs:
            txt = glob(OUTPUT_FOLDER_NAME+'/emotion/'+dir+'/'+video_dir+'/*.txt')
            if not txt:
                continue
            name = txt[0].split('.')
            name = name[0].split('/')
            im_name = name[-1].split('_')
            im_name = im_name[0]+'_'+im_name[1]+'_'+im_name[2]
            with open(txt[0],'r') as f:
                label = f.read()
                label = label.split('.')
                label = label[0].split(' ')
                label = int(label[-1])
            image = cv2.imread(name[0]+'/image/'+ name[2] + '/' + name[3] + '/' + im_name + '.png')
            if image.shape[0:1] != (CKPLUS.height, CKPLUS.width):
                image = cv2.resize(image, (CKPLUS.width, CKPLUS.height))
            #pdb.set_trace()
            #print(image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = clahe.apply(image)
            # detect the face
            detections = detector(image, 1)
            if len(detections) < 1:
                continue
            # for all detected face
            for k, d in enumerate(detections):
                w = d.right() - d.left()
                h = d.bottom() - d.top()
                crop = image[d.top():d.top()+h, d.left():d.left()+w]
                crop = cv2.resize(crop, (face_width, face_height), interpolation=cv2.INTER_LINEAR)
                if save_images:
                    cv2.imwrite(OUTPUT_FOLDER_NAME + '/face_image/' + im_name + '.png', crop)
                # extract HOG features
                hog_image = hog(crop, cell)
                landmarks_vectorised = landmark_feats(crop, d, predictor, CKPLUS.name)
                # concat feature
                feats_data = np.concatenate([feats_data, landmarks_vectorised], axis=0)
                hog_feats = np.concatenate([hog_feats, np.reshape(hog_image, [1, hog_d])], axis=0)
                feats_labels.append(label)
    print("saving features and labels...")
    with open(OUTPUT_FOLDER_NAME + '/landmarks_feats.pkl', 'wb') as f:
        pickle.dump(feats_data, f)
    with open(OUTPUT_FOLDER_NAME + '/hog_feats.pkl', 'wb') as f:
        pickle.dump(hog_feats, f)
    with open(OUTPUT_FOLDER_NAME + '/labels.pkl', 'wb') as f:
        pickle.dump(feats_labels, f)