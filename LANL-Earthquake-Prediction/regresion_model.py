import tensorflow as tf
from tensorflow import keras
from etl.feature_import import feature_h5 as f5
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import pandas as pd

if __name__ == "__main__":

    path_h5 = "H:\\kaggle\\LANL-Earthquake-Prediction\\train_obspy_full.h5"
    model_h5 = "H:\\kaggle\\LANL-Earthquake-Prediction\\model_cwt.h5"
    dh5 = f5(path_h5)

    label_x = 'f80-90'
    label_y = 'ttf'
    colx = [i for i in range(0,100)]

    x = dh5.feature_cwtf_load()
    y = dh5.y_load()
    x = np.log10(x.values)
    y = y[label_y].values

    y_rounded = np.round(y).astype('str').reshape(-1,1)

    enc = OneHotEncoder(handle_unknown='ignore')
    y_class = enc.fit(y_rounded).transform(y_rounded).toarray()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    model = tf.keras.Sequential([
        keras.layers.Dense(X_train.shape[1], activation=tf.nn.relu, input_shape=[X_train.shape[1]]),
        keras.layers.Dense(X_train.shape[1], activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.0001)

    optimizer2 = tf.keras.optimizers.SGD(lr=0.0001, momentum=1e-6)

    model.compile(optimizer='adadelta',
                  loss='mse',
                  metrics=['mae', 'mse'])

    model.fit(X_train, y_train, epochs=2000, validation_data=[X_test, y_test])

    y_val_mae = model.history.history['val_mean_absolute_error']
    y_train_mae = model.history.history['mean_absolute_error']
    x_epoch = [i for i in range(1,2001)]

    plt.plot(x_epoch, y_val_mae, x_epoch, y_train_mae)
    plt.show()
    model.save(model_h5)

    # plt.scatter(X_test[:,5],y_test)
    # plt.xlabel(label_x)
    # plt.ylabel(label_y)
    # plt.show()

