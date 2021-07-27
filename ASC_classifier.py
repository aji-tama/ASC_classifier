import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pandas

class_names = ['birds', 'broken', 'sky']
num_classes = len(class_names)

# load trained model
def create_model():
    model = Sequential([layers.experimental.preprocessing.Rescaling(1./255, input_shape=(256, 256, 3)),
                        layers.Conv2D(16, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Conv2D(32, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Conv2D(64, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Conv2D(128, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Conv2D(256, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        #layers.Conv2D(512, 3, padding='same', activation='relu'),
                        #layers.MaxPooling2D(),
                        layers.Dropout(0.2),
                        layers.Flatten(),
                        layers.Dense(128, activation='relu'),
                        layers.Dense(num_classes)])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

model = create_model()
model.load_weights(tf.train.latest_checkpoint(pathlib.Path.cwd()))

# load all images into a list

Month = '06JUN'     #<========================

pandas.DataFrame({Month:[]}).to_csv("oracle.csv", index=False)
prophecy = pandas.read_csv(pathlib.Path.cwd().joinpath('oracle.csv'))

if Month == '02FEB':
    monthdate = ['01','02','03','04','05','06','07','08','09','10',
                 '11','12','13','14','15','16','17','18','19','20',
                 '21','22','23','24','25','26','27','28']
elif Month == '03MAR' or Month == '05MAY':
    monthdate = ['01','02','03','04','05','06','07','08','09','10',
                 '11','12','13','14','15','16','17','18','19','20',
                 '21','22','23','24','25','26','27','28','29','30','31']
elif Month == '04APR' or Month == '06JUN':
    monthdate = ['01','02','03','04','05','06','07','08','09','10',
                 '11','12','13','14','15','16','17','18','19','20',
                 '21','22','23','24','25','26','27','28','29','30']

for j in range(len(monthdate)):
    print(monthdate[j])
    images = []
    images_name = []
    for img in pathlib.Path.cwd().joinpath('2021',Month,monthdate[j]).iterdir():
        images_name.append(str(img)[-12:-8])
        img = pathlib.Path.cwd().joinpath('2021',Month,monthdate[j],img)
        img = keras.preprocessing.image.load_img(img, target_size=(480, 720))
        img = keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, (256,256), interpolation='bilinear')
        img = tf.expand_dims(img, 0)
        images.append(img)

# stack up images list to pass for prediction
    images = np.vstack(images)
    predict = model.predict(images)
    classes = np.argmax(predict, axis=-1)
    scores = tf.nn.softmax(predict).numpy()
    oracle = []

    for i in range(len(images)):
        if classes[i] == 0:
            oracle.append((images_name[i],class_names[classes[i]],'{:.2f}%'.format(100*max(scores[i]))))
    for i in range(len(images)):
        if classes[i] == 1:
            oracle.append((images_name[i],class_names[classes[i]],'{:.2f}%'.format(100*max(scores[i]))))
    for i in range(len(images)):
        if classes[i] == 2:
            oracle.append((images_name[i],class_names[classes[i]],'{:.2f}%'.format(100*max(scores[i]))))

    prophecy[monthdate[j]] = pandas.Series(oracle)
    prophecy.to_csv(pathlib.Path.cwd().joinpath('oracle.csv'), index=False)
