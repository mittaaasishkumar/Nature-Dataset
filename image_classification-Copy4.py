#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import random




# In[5]:


train_path ='/home/nihith/Downloads/11/archive (1)/seg_train/seg_train'
validation_path ='/home/nihith/Downloads/11/archive (1)/seg_test/seg_test'
prediction_path = '/home/nihith/Downloads/11/archive (1)/seg_pred'


# In[6]:


#Pre Processing the data

image_size =(150,150)
batch_size = 64
train_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                 shear_range=0.2,
                                                                 zoom_range=0.2,
                                                                 horizontal_flip=True,
                                                                 )

train_batches = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training')


validation_batches= train_datagen.flow_from_directory(
     validation_path,
     target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    )

prediction_batches= train_datagen.flow_from_directory(
     prediction_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,)

num_of_train_samples = train_batches.samples
num_of_test_samples = validation_batches.samples
num_of_prediction_samples = prediction_batches.samples

print('Number of training samples:',num_of_train_samples)
print('Number of testing samples:',num_of_test_samples)
print('Number of prediction samples:',num_of_prediction_samples)


# In[7]:


# splitting the dataa into input and classes
x_train, y_train = next(train_batches)
x_valid, y_valid = next(validation_batches)
x_pred = next(prediction_batches)

#specifing the classes and displaying the data set
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
n=6  #number of classes
fig = plt.figure(figsize=(10, 5))
for i in range(len(class_names)):
  var = fig.add_subplot(2,3,1+i, xticks=[], yticks=[])
  class_path =os.path.join(train_path, class_names[i])
  img_names = os.listdir(class_path)
  img_path = os.path.join(class_path, img_names[0])
  image = plt.imread(img_path)
  var.set_title(class_names[i])
  plt.imshow(image)
plt.show()


# In[5]:


#converts categorial data types to numericals
y_train = tf.keras.utils.to_categorical(y_train, n)
y_test = tf.keras.utils.to_categorical(y_valid, n)


# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
no_of_classes = 6

# building the cnn model
model = Sequential()
model.add(Conv2D( 32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D( 32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(no_of_classes))
model.add(Activation('softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# In[7]:


import visualkeras
visualkeras.layered_view(model, scale_xy=10, legend=True)


# In[8]:


#summary of the model
print(model.summary())


# In[9]:


epochs = 10
training = model.fit(train_batches,epochs= epochs,validation_data=validation_batches,shuffle=True)



# In[10]:


import matplotlib.pyplot as plt

plt.figure(figsize=[20, 8])

# Summarize history for accuracy
plt.subplot(1, 2, 1)
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('Model Accuracy', size=25, pad=20)
plt.ylabel('Accuracy', size=15)
plt.xlabel('Epoch', size=15)
plt.legend(['train', 'test'], loc='upper left')

# Summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('Model Loss', size=25, pad=20)
plt.ylabel('Loss', size=15)
plt.xlabel('Epoch', size=15)
plt.legend(['train', 'test'], loc='upper left')

plt.show()



# In[28]:


predictions = model.predict(x_pred)

plt.figure(figsize=[10, 10])

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

n = random.randint(1,16)
plt.subplot(3, 2, 1)
plt.imshow(x_pred[n].reshape(150, 150, -1), cmap=plt.cm.binary)  
plt.title("Predicted value: " + str(class_names[np.argmax(predictions[n], axis=0)]), size=20)
plt.grid(False)

m = random.randint(10,20)
plt.subplot(3, 2, 2)
plt.imshow(x_pred[m].reshape(150, 150, -1), cmap=plt.cm.binary)  
plt.title("Predicted value: " + str(class_names[np.argmax(predictions[m], axis=0)]), size=20)
plt.grid(False)

o = random.randint(1,25)
plt.subplot(3, 2, 3)
plt.imshow(x_pred[o].reshape(150, 150, -1), cmap=plt.cm.binary)  
plt.title("Predicted value: " + str(class_names[np.argmax(predictions[o], axis=0)]), size=20)
plt.grid(False)

p = random.randint(1,28)
plt.subplot(3, 2, 4)
plt.imshow(x_pred[p].reshape(150, 150, -1), cmap=plt.cm.binary)  
plt.title("Predicted value: " + str(class_names[np.argmax(predictions[p], axis=0)]), size=20)
plt.grid(False)


plt.suptitle("Predictions of Dataset", size=30, color="#6166B3")

plt.show()


# In[ ]:




