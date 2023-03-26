import random
import pandas as pd
import numpy as np
import os
import tensorflow
import cv2
from tensorflow import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Activation, GlobalAveragePooling2D,Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
from keras.applications import InceptionResNetV2
from keras.applications.mobilenet import preprocess_input
from keras.models import Sequential, Model
from pathlib import Path
import matplotlib.pyplot as plt

data_folder = Path("C:/Users/kbinw/Binwant/01 IGDTUW/8th Sem/Final YR Project/Code 4/Sign_lang_project/")

img_width, img_height = 224, 224
img_folder = data_folder/"Data"

n_epoch = 20
batch_sz = 16
input_shape = (img_width, img_height, 3)
input_tensor = Input(shape=(224, 224, 3))

# imports the mobilenet model and discards the last 1000 neuron layer
base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)

model = Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(1024,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(24,activation='softmax'))

for layer in base_model.layers:
    layer.trainable = False

model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', patience=5 ,
                   restore_best_weights=True, verbose=1)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc', tensorflow.keras.metrics.AUC(name='auc'), tensorflow.keras.metrics.AUC(name='roc')])

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   validation_split=0.2,
                                   featurewise_center=False,  # set input mean to 0 over the dataset
                                   samplewise_center=False,  # set each sample mean to 0
                                   featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                   samplewise_std_normalization=False,  # divide each input by its std
                                   zca_whitening=False,  # apply ZCA whitening
                                   rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                                   zoom_range=0.1,  # Randomly zoom image
                                   width_shift_range=0.1,
                                   # randomly shift images horizontally (fraction of total width)
                                   height_shift_range=0.1,
                                   # randomly shift images vertically (fraction of total height)
                                   horizontal_flip=False,  # randomly flip images
                                   vertical_flip=False)

train_generator = train_datagen.flow_from_directory(img_folder,
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=batch_sz,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory(img_folder,
                                                         target_size=(224, 224),
                                                         color_mode='rgb',
                                                         batch_size=batch_sz,
                                                         class_mode='categorical',
                                                         shuffle=False,
                                                         subset='validation')

step_size_train = train_generator.n//train_generator.batch_size

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples//validation_generator.batch_size,
                    steps_per_epoch=step_size_train,
                    epochs=n_epoch)

# predictions = model.predict_generator(test_generator, test_generator.samples//batch_sz+1)
# pred = np.argmax(predictions, axis=1)
# cm = confusion_matrix(test_generator.classes, pred)
#
# print('Confusion Matrix')
# print(cm)
# print('Classification Report')
# target_names = ['0', '1', '2', '3', '4']
# print(classification_report(test_generator.classes, pred, target_names=target_names))
#
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
#
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Save the trained model for testing and classification in real-time
model.save(data_folder/"Models/InceptionV2Model.h5")
