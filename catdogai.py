import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D,BatchNormalization,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator



train_dir = "D:/ai/catdogrecog/dataset/training_set/"
test_dir = "D:/ai/catdogrecog/dataset/test_set/"

train2_dir = "D:/ai/catdogrecog/cat and dog/train/"
test2_dir = "D:/ai/catdogrecog/cat and dog/validation/"

validation_dog_dir = "D:/ai/catdogrecog/cat and dog/validation/Dog/"
validation_cat_dir = "D:/ai/catdogrecog/cat and dog/validation/Cat/"
dog_images = os.listdir(validation_dog_dir)
cat_images = os.listdir(validation_cat_dir)
dog_images = random.sample(dog_images,5)
cat_images = random.sample(cat_images,5)

batch = 256
image_size = (64,64)


#圖片數據增強
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
    horizontal_flip=True
)                  


test_datagen = ImageDataGenerator(
    rescale=1./255
)

#數據集生成器
train_generator = train_datagen.flow_from_directory(
    train2_dir,
    target_size=image_size,
    batch_size=batch,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test2_dir,
    target_size=image_size,
    batch_size=batch,
    class_mode='binary'
)



model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(image_size[0],image_size[1],3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),
    
    Conv2D(128,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2), 
    Dropout(0.2),

    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.2),
    Dense(128,activation='relu'),
    Dropout(0.2),
    Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)


model.save("catdog.keras")

image_path = [os.path.join(validation_dog_dir, img)for img in dog_images]+[os.path.join(validation_cat_dir,img) for img in cat_images]

plt.figure(figsize=(10,10))
for i, image_path in enumerate(image_path,1):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(image_size[0],image_size[1]))
    img = img / 255.0
    prediction = model.predict(np.expand_dims(img, axis=0))[0]
    if prediction > 0.5:
        label = "Dog"
    else:
        label = "Cat"
    plt.subplot(5,5,i)
    plt.imshow(img)
    plt.title(label)
    plt.axis("off")
plt.show()