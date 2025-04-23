# models/tuned_nn.py

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping
import optuna
from features.domain_features import load_with_features

def load_data():
    X, y = load_with_features()
    X = PowerTransformer().fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X, y


def objective(trial):
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(trial.suggest_int("units_1", 32, 128), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float("dropout_1", 0.2, 0.5)))
    model.add(Dense(trial.suggest_int("units_2", 16, 64), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float("dropout_2", 0.1, 0.4)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)),
                  loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
                        callbacks=[es], verbose=0)
    return max(history.history['val_accuracy'])

def run_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print("Best Parameters:", study.best_params)
    return study.best_params

def train_final_model(params):
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(params['units_1'], activation='relu'),
        BatchNormalization(),
        Dropout(params['dropout_1']),
        Dense(params['units_2'], activation='relu'),
        BatchNormalization(),
        Dropout(params['dropout_2']),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=params['batch_size'],
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=1)
    return model

if __name__ == "__main__":
    best_params = run_optuna()
    model = train_final_model(best_params)
    model.save("models/optuna_best_nn.h5")
