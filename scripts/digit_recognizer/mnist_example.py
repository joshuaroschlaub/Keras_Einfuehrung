##### Importieren aller benötigten Module #####


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Für die bessere Lesbarkeit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.metrics import Accuracy
print("Module importiert.")

##### Code #####


# Laden der 28x28 großen Bilder aus dem MNIST Datensatz
# und abspeichern in Numpy arrays.
mnist = tf.keras.datasets.mnist 
(train_samples, train_labels) = mnist.load_data()[0]
(test_samples, test_labels) = mnist.load_data()[1]
print("Daten geladen.")

# Normieren der Trainings- und Testdaten auf Werte zwischen 0 und 1.
train_samples = tf.keras.utils.normalize(train_samples, axis=1)
test_samples = tf.keras.utils.normalize(test_samples, axis=1)
print("Daten normiert.")

# Erstellung des Modells.
model = Sequential([
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
])
print("Modell erstellt.")

# Kompilierung des Modells mit Optimizer, Verlustfunktion und
# zu verwendender Metrik.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Modell compiliert.")

# Trainieren des Modells mit den MNIST Trainingsdaten.
history = model.fit(train_samples, train_labels, epochs=5,
                    validation_split=0.1, batch_size=30, shuffle=True)
print("Modell trainiert")


# Kurve der Treffer-Genauigkeit des Modells mit Trainings- und Testdaten
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()
plt.savefig("/work/baw0284/results/accuracy_graph.png")

# Kurve der Loss-Function des Modells mit Trainings- und Testdaten
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()
plt.savefig("/work/baw0284/results/loss_graph.png")