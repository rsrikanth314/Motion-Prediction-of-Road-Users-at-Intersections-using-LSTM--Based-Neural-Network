from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Flatten
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LeakyReLU
from keras import callbacks
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def LSTM_CNN_Dense_model(X_train, X_test, Y_train, Y_test):
    
    # Manipulating input data in order to input the data to the LSTM Encoder, the data needs to be in the form of a 3 dimensional tensor in order to solve the problem of Multi-variate Multi output Multi-step time series forecasting.
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    Y_train = Y_train.transpose(0, 2, 1)
    Y_test = Y_test.transpose(0, 2, 1)
    
    
    
    # LSTM autoencoder network with cnn and dense layers
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(20, 4), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(RepeatVector(15))
    model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv1D(64, kernel_size=3, activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv1D(128, kernel_size=3, activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                 
    model.add(Dense(500, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(250, activation='relu'))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(3)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()


    #Initializing earlystopping callback
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 20, restore_best_weights = True)
    
    #Fit and Train the model
    history = model.fit(X_train, Y_train, epochs = 200, batch_size=64, validation_data =(X_test, Y_test), callbacks =[earlystopping])
    
    # save the trained model
    filepath = 'LSTM_CNN_Dense_training_ID_27_testing_ID_28.h5'
    model.save(filepath)
    
    # load trained model
    model = load_model('LSTM_CNN_Dense_training_ID_27_testing_ID_28.h5')
 

    # Training and validation loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0,0.0007)
    plt.xlim(0,108)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('LSTM_CNN_Dense_Loss_Curve.pdf')

    return model