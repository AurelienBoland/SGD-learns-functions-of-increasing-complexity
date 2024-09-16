from pathlib import Path
from typing import List, Tuple
import numpy as np
import tensorflow as tf

from sgd_complexity.utils import est_density

class EstimatedDensity(tf.keras.callbacks.Callback):
    def __init__(self, data:Tuple[tf.Tensor,tf.Tensor], submodels_predictions:List[np.array], n_classes:int=2):
        super().__init__()
        self.densities = []
        self.X = data[0] #tf.convert_to_tensor(data[0],dtype=tf.float32) #evaluating the model on a numpy array multiple times create a memory leak see https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
        self.y = data[1].numpy()
        self.submodels_predictions = submodels_predictions
        self.n_classes = n_classes
    def compute_densities(self):
        #y_pred = np.zeros(len(self.y))
        y_pred = self.model.predict(self.X)
        if self.n_classes == 2:
            y_pred = np.array(y_pred > 0.5, dtype="int")
        else:
            y_pred = np.argmax(y_pred, axis=-1)
        if self.submodels_predictions is None:
            self.densities.append(est_density(y_pred, self.y, n_classes=self.n_classes))
        else:
            densities_epoch = []
            for y_pred_sub in self.submodels_predictions:
                densities_epoch.append(est_density(y_pred, y_pred_sub, self.y, n_classes=self.n_classes))
            self.densities.append(densities_epoch)
    def on_train_begin(self, logs=None):
        self.compute_densities()
    def on_epoch_end(self, epoch, logs=None):
        self.compute_densities()

class SaveIntermediateModels(tf.keras.callbacks.Callback):
    def __init__(self, path:str, period:int):
        super().__init__()
        self.path = path
        self.period = period
    
    def on_train_begin(self, logs=None):
        self.model.save(Path(self.path) / f"epoch_0.keras")
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.period == 0:
            self.model.save(Path(self.path) / f"epoch_{epoch+1}.keras")

class SaveIntermediatePredictions(tf.keras.callbacks.Callback):
    def __init__(self, path:str, data:Tuple[tf.Tensor,tf.Tensor]):
        super().__init__()
        self.path = path
        self.X = data[0]
        self.y = data[1].numpy()
    
    def on_train_begin(self, logs=None):
        y_pred = self.model.predict(self.X)
        if self.n_classes == 2:
            y_pred = np.array(y_pred > 0.5, dtype="int")
        else:
            y_pred = np.argmax(y_pred, axis=-1)
        np.save(Path(self.path) / f"epoch_0.npy",y_pred)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        if self.n_classes == 2:
            y_pred = np.array(y_pred > 0.5, dtype="int")
        else:
            y_pred = np.argmax(y_pred, axis=-1)
        np.save(Path(self.path) / f"epoch_{epoch+1}.npy",y_pred)

            