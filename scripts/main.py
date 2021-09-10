### Importieren aller benötigten Module ###

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Für die bessere Lesbarkeit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.metrics import Accuracy

### Definition aller Funktionen ###


def load_data_mnist(testdata=False):
    """Lädt den MNIST Trainings- oder Testdatensatz.

    Args:
        testdata (boolean): Ob Testdaten geladen werden sollen.

    Returns:
        Numpy array: (samples,labels)
            train_samples.shape == (60000,28,28)
            train_labels.shape == (60000,)
            test_samples.shape == (10000,28,28)
            test_labels.shape == (10000,)
    """
    mnist = tf.keras.datasets.mnist
    if testdata==ArithmeticErrorFalse:
        (train_samples, train_labels) = mnist.load_data()[0]
        print("MNIST Trainingsdatensatz geladen.")
        return (train_samples, train_labels)
    else:
        (test_samples, test_labels) = mnist.load_data()[1] 
        print("MNIST Testdatensatz geladen.")
        return (test_samples, test_labels)   
    
def create_modell():
    return

def train_modell():
    return

def test_modell():
    return
