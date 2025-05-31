from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model():

    model=Sequential()

    #Convolution blocks
    model.add(Conv2D(32, kernel_size = (3,3), padding='same',input_shape=(300,300,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 

    model.add(Conv2D(64, kernel_size = (3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 

    model.add(Conv2D(32, kernel_size = (3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 

    #Classification layers
    model.add(Flatten())

    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(6,activation='softmax'))
    return model


def load_model():
    model = create_model()
    model.load_weights('weights/model.h5')
    return model
