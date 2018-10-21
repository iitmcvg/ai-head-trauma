""" Contains the code for resnet_models_1/2/3 as explained in resnet_models.md """
import os
import time
import numpy as np
from resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten

from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

# Create test and train dataset
if os.path.exists('X.npy') and os.path.exists('y.npy'):
    # Load X and y created using dataset_preparation.py
    X = np.load('X.npy')
    y = np.load('y.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1067, random_state=42)
    
    X_train = X_train/255.
    X_test = X_test/255.
    
    # Stack the test and train images 3 times to match input dimension of resnet50 i.e. 224x224x3
    X_train = np.stack([X_train]*3, axis= -1)
    X_test = np.stack([X_test]*3, axis= -1)
else:
    print('First run dataset_preparation.py')

# Binary classification task; patient is operated on([1 0]) or not([0 1])
num_classes = 2

###########################################################################################################################
# resnet_model_1
# Replace dense output layer of 1000 classes with dense output layer of 2 classes

image_input = Input(shape=(224, 224, 3))

# Load standard ResNet50 model
model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

# Modify softmax dense layer
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)

# Create custom resnet model
resnet_model1 = Model(inputs=image_input,outputs= out)
resnet_model1.summary()

# Freeze the imagenet weights of all layers except newly modified layer
for layer in resnet_model1.layers[:-1]:
	layer.trainable = False

# Create checkpoint
checkpoint = ModelCheckpoint('./temp/resnet_model_1_weights.{epoch:02d}-{val_acc:.2f}.hdf5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')
# Compile the model
resnet_model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train the model
t=time.time()
hist = resnet_model1.fit(X_train, y_train, batch_size=32, epochs=6, verbose=1, validation_data=(X_test, y_test), callbacks = [checkpoint])
print('Training time: %s' % (time.time()-t))

# Evaluate the model
(loss, accuracy) = resnet_model1.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

###########################################################################################################################
# resnet_model_2
# Fine tune the resnet 50. Changed the Fully connected layers completely and added global pooling average, dropout
# and dense layers. Froze weights of all convolutional layers

image_input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
model.summary()
last_layer = model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)

# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-3')(x)
x = Dropout(0.5)(x)

# a softmax layer for 2 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# Create custom resnet model
resnet_model2 = Model(inputs=image_input, outputs=out)
resnet_model2.summary()

# Freeze the imagenet weights of all layers except newly modified layer
for layer in resnet_model2.layers[:-5]:
	layer.trainable = False
    
#print(resnet_model2.layers[-5].name)
#print(resnet_model2.layers[-5].trainable)

# Create checkpoint
checkpoint = ModelCheckpoint('./temp/resnet_model_2_weights.{epoch:02d}-{val_acc:.2f}.hdf5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')

# Compile the model
resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train the model
t=time.time()
hist = resnet_model2.fit(X_train, y_train, batch_size=32, epochs=6, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint])
print('Training time: %s' % (t - time.time()))

###########################################################################################################################
# resnet_model_3
# Fine tune the resnet 50 to retrain last convolutional layer as well.

image_input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
model.summary()

last_layer = model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 2 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
resnet_model3 = Model(inputs=image_input, outputs=out)
resnet_model3.summary()

#resnet_model3.load_weights('./temp/resnet_model_2_weights.{epoch:02d}-{val_acc:.2f}.hdf5')

# Freeze the imagenet weights of all layers except newly modified layer and the last conv layer
for layer in resnet_model3.layers[:-11]:
	layer.trainable = False


# Create checkpoint
checkpoint = ModelCheckpoint('./temp/resnet_model_3_weights.{epoch:02d}-{val_acc:.2f}.hdf5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')

# Compile the model
resnet_model3.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# Train the model
t=time.time()
hist = resnet_model3.fit(X_train, y_train, batch_size=32, epochs=6, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint])
print('Training time: %s' % (t - time.time()))

###########################################################################################################################
# visualizing losses and accuracy
import matplotlib.pyplot as plt

# Number of epochs used to train the model
NUM_EPOCHS = 6
# Model name (for saving plots)
MODEL_NAME = 'resnet_model_X'

train_loss = hist.history['loss']
val_loss   = hist.history['val_loss']
train_acc  = hist.history['acc']
val_acc    = hist.history['val_acc']

xc=range(NUM_EPOCHS)

## Train_loss vs Val_loss
a = plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
#a.savefig('{}_loss.png'.format(MODEL_NAME))

## Train_acc vs Val_acc
a = plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
#a.savefig('{}_acc.png'.format(MODEL_NAME))