
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128, activation='relu', init='uniform'))
#classifier.add(Dense(output_dim=128, activation='relu', init='uniform'))
classifier.add(Dense(output_dim=1, activation='sigmoid', init='uniform'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

fit = False

from keras.models import load_model
from keras.preprocessing import image
if fit == True:
    classifier.fit_generator(
            training_set,
            samples_per_epoch = 8000,
            epochs=5,
            validation_data=test_set,
            nb_val_samples =2000)
    classifier.save('cat_vs_dog.h5')
else:
    classifier = load_model('cat_vs_dog.h5')
    print('loaded the model')
    
'''def catOrDog(y_pred):
    objects = ('cat')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, y_pred, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('cat')
    plt.show() '''

img = image.load_img('img/d1.jpeg', target_size=(64,64))
#plt.imshow(img)
X = image.img_to_array(img)
X = np.expand_dims(X, axis = 0)
X /= 255

y_pred = classifier.predict(X)
#catOrDog(y_pred[0])
print(y_pred[0])

