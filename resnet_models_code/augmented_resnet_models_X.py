""" Contains the code for augmented_resnet_models_1/2/3 and also custom_model_1 as explained in resnet_models.md """
import os
import numpy as np
from resnet50 import ResNet50
from keras.layers import Dense, Dropout, Flatten, Conv2D

from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# Create test and train dataset
if os.path.exists('X.npy') and os.path.exists('y.npy'):
    # Load X and y created using dataset_preparation.py
    X = np.load('X.npy')
    y = np.load('y.npy')
    y.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Stack the test and train images 3 times to match input dimension of resnet50 i.e. 224x224x3
    X_train = np.stack([X_train]*3, axis= -1)
    X_test = np.stack([X_test]*3, axis= -1)
else:
    print('First run dataset_preparation.py')
    
# Instantiate the ImageDataGenerator object. The transformations include horizontal/vertical shifts, upto 45 degree rotations and horizontal flips.
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,# randomly flip images
    rescale = 1./255)  

datagen.fit(X_train) 

# Binary classification task; patient is operated on([1 0]) or not([0 1])
num_classes = 2

image_input = Input(shape=(224, 224, 3))

###########################################################################################################################
# augmented_resnet_model_1
#Replaced dense output layer of 1000 classes with a dense layer of 2 classes. Also, added a dropout layer between flatten and output layer.

# Load standard ResNet50 model
model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

# Modify softmax dense layer and add dropout layer
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)

# Create custom resnet model
augmented_model = Model(inputs=image_input,outputs= out)
augmented_model.summary()

# Freeze the imagenet weights of all layers except newly modified layer
for layer in augmented_model.layers[:-1]:
	layer.trainable = False

#augmented_model.load_weights('weights.01-0.74-LRe-4-D0.5.hdf5')

# Create checkpoint
augmented_checkpoint = ModelCheckpoint('./temp/augmented_resnet_model_1_weights.{epoch:02d}-{val_acc:.2f}.hdf5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')

## User-defined inputs
NUM_EPOCH = 10
LR = 1e-4

# Compile the model
adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
augmented_model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['accuracy'])

# Train the model
augmented_model_details = augmented_model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32),
                    steps_per_epoch = len(X_train) / 32, # number of samples per gradient update
                    epochs = NUM_EPOCH, # number of iterations
                    validation_data= (X_test, y_test),
                    callbacks=[augmented_checkpoint],
                    verbose=1)

###########################################################################################################################
# augmented_resnet_model_2
# Everything same as augmented_resnet_model_1 but also added a dense layer(512) in between the flatten and dropout layer.

# Load standard ResNet50 model
model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

# Modify the FCN
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)

# Create custom resnet model
augmented_model = Model(inputs=image_input,outputs= out)
augmented_model.summary()

#augmented_model.load_weights('augmented2_weights.08-val_acc0.72.hdf5')

# Freeze the imagenet weights of all layers except newly modified layer
for layer in augmented_model.layers[:-4]:
	layer.trainable = False

#augmented_model.layers[-4].name
#augmented_model.layers[-4].trainable

# Create checkpoint
augmented_checkpoint = ModelCheckpoint('./temp/augmented_resnet_model_2_weights.{epoch:02d}-{val_acc:.2f}.hdf5',  # model filename
                             monitor='val_acc', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')

## User-defined inputs
NUM_EPOCH = 10

# Compile the model
augmented_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

# Train the model
augmented_model_details = augmented_model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32),
                    steps_per_epoch = len(X_train) / 32, # number of samples per gradient update
                    epochs = NUM_EPOCH, # number of iterations
                    validation_data= (X_test, y_test),
                    callbacks=[augmented_checkpoint],
                    verbose=1)


# Evaluate the model
(loss, accuracy) = augmented_model.evaluate(X_test, y_test, batch_size=32, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

###########################################################################################################################
# augmented_resnet_model_3
#Used weights of model_2 and retrained the last conv layers (retrained last 15 layers in new architecture
# which had 3 conv layers)

# Load standard ResNet50 model
model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

# Modify FCN (same as augmented_resnet_model_2)
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)

# Create custom resnet model
augmented_model = Model(inputs=image_input,outputs= out)
augmented_model.summary()

#augmented_model.load_weights('./temp/augmented_resnet_model_2_weights.{epoch:02d}-{val_acc:.2f}.hdf5')
#augmented_model.load_weights('./temp/augmented3_weights.18-val_acc0.75.hdf5')

# Freeze the imagenet weights of all layers except newly modified layer
for layer in augmented_model.layers[:-15]:
	layer.trainable = False

# Create checkpoint
augmented_checkpoint = ModelCheckpoint('./temp/augmented_resnet_model_3_weights.{epoch:02d}-val_acc{val_acc:.2f}.hdf5',  # model filename
                             monitor='val_acc', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')

## User-defined inputs
NUM_EPOCH = 10
LR = 1e-4

# Compile the model
adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
augmented_model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['accuracy'])

# Train the model
augmented_model_details = augmented_model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32),
                    steps_per_epoch = len(X_train) / 32, # number of samples per gradient update
                    epochs = NUM_EPOCH, # number of iterations
                    validation_data= (X_test, y_test),
                    verbose=1)

###########################################################################################################################
# custom_model_1
# Model: 5x5, max pool, 3x3, mp, 3x3,mp, 3x3, mp, fc
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.summary()

# Create checkpoint
augmented_checkpoint = ModelCheckpoint('./temp/custom_model_aug_8_weights.{epoch:02d}-val_acc{val_acc:.2f}.hdf5',  # model filename
                             monitor='val_acc', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# User-defined inputs
NUM_EPOCH = 20

# Train the model
augmented_model_details = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32),
                    steps_per_epoch = len(X_train) / 32, # number of samples per gradient update
                    epochs = NUM_EPOCH, # number of iterations
                    validation_data= (X_test, y_test),
                    callbacks=[augmented_checkpoint],
                    verbose=1)

############################################################################################
# visualizing losses and accuracy
import matplotlib.pyplot as plt

# Number of epochs used to train the model
NUM_EPOCHS = 10
# Model name (for saving plots)
MODEL_NAME = 'augmented_resnet_model_X'

train_loss = augmented_checkpoint.history['loss']
val_loss   = augmented_checkpoint.history['val_loss']
train_acc  = augmented_checkpoint.history['acc']
val_acc    = augmented_checkpoint.history['val_acc']

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