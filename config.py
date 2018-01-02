class fer2013:
    name = 'fer2013'
    train_folder = 'fer2013_features/Training'
    validation_folder = 'fer2013_features/PublicTest'
    test_folder = 'fer2013_features/PrivateTest'
    width = 48
    height = 48
    cell = 3

class ckplus:
    name = 'ckplus'
    image_folder = 'ckplus/image'
    emotion_folder = 'ckplus/emotion'
    width = 640
    height = 480
    face_width = 48
    face_height = 48
    cell = 3
    ratio = [0.8,0.1,0.1]

FER2013 = fer2013()
CKPLUS = ckplus()

