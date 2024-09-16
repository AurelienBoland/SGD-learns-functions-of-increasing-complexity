import numpy as np
from pydantic import BaseModel, ConfigDict, SkipValidation
from tensorflow.keras.datasets import mnist, cifar10

CIFAR10_ANIMALS_INDICES = [2, 3, 5, 7] # 4 of the 6 animals in CIFAR10
CIFAR10_VEHICLES_INDICES = [0, 1, 8, 9]

class Dataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    X_train: SkipValidation[np.ndarray]
    y_train: SkipValidation[np.array]
    X_test: SkipValidation[np.ndarray]
    y_test: SkipValidation[np.array]

def load_dataset(name):
    if name == "mnist":
        return mnist_dataset()
    elif name == "cifar10_animals":
        return cifar10_animals_dataset()
    elif name == "cifar10_first":
        return cifar10_first_dataset()
    elif name == "mnist_multiclass":
        return mnist_multiclass_dataset()   
    else:
        raise ValueError("Dataset name not recognized.")

def mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32")/255
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = X_test.astype("float32")/255
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = np.array(y_train > 4, dtype="int")
    y_test = np.array(y_test > 4, dtype="int")
    dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return dataset

def mnist_multiclass_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32")/255
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = X_test.astype("float32")/255
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = y_train
    y_test = y_test
    dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return dataset

def cifar10_animals_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32")/255
    X_test = X_test.astype("float32")/255
    X_train = X_train[np.isin(y_train, CIFAR10_ANIMALS_INDICES + CIFAR10_VEHICLES_INDICES)]
    X_test = X_test[np.isin(y_test, CIFAR10_ANIMALS_INDICES + CIFAR10_VEHICLES_INDICES)]
    y_train = y_train[np.isin(y_train, CIFAR10_ANIMALS_INDICES + CIFAR10_VEHICLES_INDICES)]
    y_test = y_test[np.isin(y_test, CIFAR10_ANIMALS_INDICES + CIFAR10_VEHICLES_INDICES)]
    y_train = np.array(np.isin(y_train, CIFAR10_ANIMALS_INDICES), dtype="int")
    y_test = np.array(np.isin(y_test, CIFAR10_ANIMALS_INDICES), dtype="int")
    dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return dataset

def cifar10_first_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32")/255
    X_test = X_test.astype("float32")/255
    y_train = np.array(y_train > 4, dtype="int")
    y_test = np.array(y_test > 4, dtype="int")
    dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return dataset


