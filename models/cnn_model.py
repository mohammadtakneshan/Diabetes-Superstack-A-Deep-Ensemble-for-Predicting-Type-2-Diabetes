# models/cnn_model.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape, Input
from keras.callbacks import EarlyStopping

def load_data():
    from numpy import loadtxt
    dataset = loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    X = PowerTransformer().fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X, y

def reshape_for_cnn(X):
    # Reshape to (samples, 4, 2, 1) to simulate 2D feature grid
    return X.reshape((X.shape[0], 4, 2, 1))

def build_cnn_model():
    model = Sequential([
        Input(shape=(4, 2, 1)),
        Conv2D(32, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(2, 1), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn():
    X, y = load_data()
    X, y = SMOTE(random_state=42).fit_resample(X, y)
    X = reshape_for_cnn(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = build_cnn_model()
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[es], verbose=1)
    return model

if __name__ == "__main__":
    model = train_cnn()
    model.save("models/cnn_feature_image_model.h5")
