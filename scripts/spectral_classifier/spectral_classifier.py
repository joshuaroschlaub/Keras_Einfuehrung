import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
import csv
import tensorflow as tf
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer, GlobalAveragePooling1D
from tensorflow.keras.metrics import Accuracy

from tensorflow import keras
from tensorflow.keras import layers

import plots
import benchmark

########## Input ##########

data_path = 'F:\\data\\'
samples_per_class = 1000

########## Program ##########

# Listen mit den flux Werten, Labels und Wellenlängen erstellen
data = np.load(data_path + "data.npy")
labels = np.load(data_path + "labels.npy")
wavelengths = np.load(data_path + "wavelengths.npy")

# Liste die Galaxie-Nummer speichert
numbers = range(4*samples_per_class)

# Datensatz mischen
z = list(zip(data, labels, numbers))
random.shuffle(z)
data_shuffled, labels_shuffled, numbers_shuffled = zip(*z)

split_index = int(len(data_shuffled)*0.9)

# Trainings- und Testdatensatz erstellen
data_training = np.asarray(data_shuffled[:split_index])
data_test = np.asarray(data_shuffled[split_index:])

labels_training = np.asarray(labels_shuffled[:split_index])
labels_test = np.asarray(labels_shuffled[split_index:])

numbers_training = numbers_shuffled[:split_index]
numbers_test = numbers_shuffled[split_index:]

# Daten in Form für Convolutional Network bringen
input_shape = (3522,1)
data_training_r = np.reshape(data_training, newshape=(len(data_training), input_shape[0], input_shape[1]))
data_test_r  = np.reshape(data_test, newshape=(len(data_test), input_shape[0], input_shape[1]))

# Netzwerk erstellen
model = Sequential([
    Conv1D(filters=64, kernel_size=80, strides=10, activation='relu', input_shape=(3522,1)),
    MaxPooling1D(3),
    Dropout(0.35),
    Conv1D(filters=128, kernel_size=40, strides=10, activation='relu'),
    MaxPooling1D(3),
    Dropout(0.35),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dropout(0.35),
    Dense(units=4, activation='softmax')
])

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_train = tf.keras.utils.normalize(data_training_r, axis=1)
x_test = tf.keras.utils.normalize(data_test_r, axis=1)

y_train = labels_training
y_test = labels_test

history = model.fit(x_train, y_train,
                    epochs=75, validation_split=0.1,
                    shuffle=True, batch_size=200,
                    verbose=1)

# Auswertung
plots.plot_accuracy("accuracy_graph_v2.png", history)
plots.plot_loss("loss_graph_v2.png", history)

# Benchmark
benchmark.benchmark_all(data_training, labels_training, data_test, labels_test)