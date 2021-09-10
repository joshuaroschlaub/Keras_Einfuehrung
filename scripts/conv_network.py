########## Importieren aller benötigten Module ##########


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Für die bessere Lesbarkeit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.metrics import Accuracy
print("Module importiert.")

########## Code ##########


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

# Daten in dreidimensionale Form (28,28,1) bringen.
input_shape = (28, 28, 1)
train_samples = train_samples.reshape(len(train_samples), input_shape[0], 
                input_shape[1], input_shape[2])
test_samples  = test_samples.reshape(len(test_samples), input_shape[0],
                input_shape[1], input_shape[2])

# Erstellung des Modells.
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)), # stride
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(units=128, activation='relu'), # Droput, weniger neuronen
    Dropout(0.5),
    Dense(units=10, activation='softmax')
])
print("Modell erstellt.")

# Kompilierung des Modells mit Optimizer, Verlustfunktion und
# zu verwendender Metrik.
model.compile(optimizer='Adadelta',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trainieren des Modells mit den MNIST Trainingsdaten.
history = model.fit(train_samples, train_labels, epochs=60,
                validation_split=0.1, batch_size=128, shuffle=True)
print("Modell trainiert")

# Modell speichern.
model.save('/work/baw0284/networks/conv_network.h5')

# Kurve der Treffer-Genauigkeit des Modells mit Trainings- und Testdaten
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()
plt.savefig("/work/baw0284/results/conv_network_acc.png")

# Kurve der Loss-Function des Modells mit Trainings- und Testdaten
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()
plt.savefig("/work/baw0284/results/conv_network_loss.png")