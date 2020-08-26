#building convolution NN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers

#initialize CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))

#Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

#Convolultion layer
classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))

#Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

#Flattenning
classifier.add(Flatten())

#Full connection to neural network
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(50,activation='softmax'))

#Compiliing CNN
sgd = optimizers.SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Fit CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_data = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)


training_set = train_data.flow_from_directory(
            'dataset/images/train',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')

test_set = test_data.flow_from_directory(
        'dataset/images/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=50,
        epochs=35,
        validation_data=test_set)

classifier.save('leaf_pred.h5')

