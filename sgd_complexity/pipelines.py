from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import tempfile

from sgd_complexity.utils import I_XY, mu_metric, est_density
from sgd_complexity import datasets
from sgd_complexity import models
from sgd_complexity.tensorflow.callbacks import EstimatedDensity, SaveIntermediatePredictions

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class ExperimentPipeline:
    def __init__(self,config) -> None:
        quick_config_scan(config)
        self.config = config
        self.results = None
    
    def run(self,mutiple_train=False,iterations_mutiple_train=5,additional_callbacks=[],save_path_submodels=None) -> None:
        logging.info("Starting the experiment...")
        config = self.config
        if "n_classes" in config.keys():
            n_classes = config["n_classes"]
        else:
            n_classes = 2
        if "submodels_training_strategy" not in config.keys() or config["submodels_training_strategy"] == "ground truth":
            pass 
        elif config["submodels_training_strategy"] == "predictions at infinity":
            if mutiple_train:
                raise NotImplementedError("The submodels_training_strategy 'predictions at infinity' is not implemented for multiple train")
            self._run_with_infinity_strategy(additional_callbacks,save_path_submodels, n_classes)
            return #dirty trick to avoid the rest of the function
        else:
            raise ValueError("The submodels_training_strategy is not valid")
        
        data = datasets.load_dataset(config["dataset"])
        data = convert_to_tensor(data)

        # Load the submodels
        logging.info("Loading submodels...")	
        submodels = []
        submodels_fit_kwargs = []

        for submodel_config in config["submodels"]:
            submodels_fit_kwargs.append({} if "fit_kwargs" not in submodel_config.keys() else submodel_config["fit_kwargs"])
            if 'model_kwargs' not in submodel_config.keys():
                submodels.append(models.load_model(submodel_config["name"]))
            else:
                submodels.append(models.load_model(submodel_config["name"],**submodel_config["model_kwargs"]))

        # Train and evaluate the submodels	
        logging.info("Training and evaluating submodels")
        submodels_predictions = []
        submodels_I = [] 
        for submodel, submodel_fit_kwargs,i in zip(submodels, submodels_fit_kwargs,range(len(submodels))):
            logging.info(f"Training submodel")
            submodel_fit_kwargs["epochs"] = 10 if "epochs" not in submodel_fit_kwargs.keys() else submodel_fit_kwargs["epochs"]
            submodel_fit_kwargs["batch_size"] = 32 if "batch_size" not in submodel_fit_kwargs.keys() else submodel_fit_kwargs["batch_size"]
            submodel.fit(data.X_train, data.y_train, **submodel_fit_kwargs)
            logging.info(f"Evaluating submodel")
            y_pred_sub = submodel.predict(data.X_test)
            if n_classes == 2:
                y_pred_sub = np.array(y_pred_sub > 0.5, dtype="int")
            else:
                y_pred_sub = np.argmax(y_pred_sub, axis=-1)
            submodels_predictions.append(y_pred_sub)  
            density = est_density(y_pred_sub, data.y_test.numpy(), n_classes=n_classes)
            submodels_I.append(I_XY(density))
            if save_path_submodels is not None:
                submodel.save(Path(save_path_submodels) / f"submodel_{i}.keras")

        logging.info("Submodels training and evaluation completed.")
        if not mutiple_train:

            # Load the main model
            logging.info("Training the main model")
            if 'model_kwargs' not in config["model"].keys():
                model = models.load_model(config["model"]["name"])
            else:
                model = models.load_model(config["model"]["name"],**config["model"]["model_kwargs"])
            
            # Train the main model
            est_callback_train = EstimatedDensity(data = (data.X_train, data.y_train), submodels_predictions=None, n_classes=n_classes)
            est_callback_test = EstimatedDensity(data = (data.X_test, data.y_test), submodels_predictions=submodels_predictions, n_classes=n_classes)

            model_fit_kwargs = {} if "fit_kwargs" not in config["model"].keys() else config["model"]["fit_kwargs"]
            model_fit_kwargs["epochs"] = 20 if "epochs" not in model_fit_kwargs.keys() else model_fit_kwargs["epochs"]
            model_fit_kwargs["batch_size"] = 32 if "batch_size" not in model_fit_kwargs.keys() else model_fit_kwargs["batch_size"]
            history = model.fit(data.X_train, data.y_train, callbacks=[est_callback_train,est_callback_test] + additional_callbacks, **model_fit_kwargs)

            logging.info("Main model training completed.")
            logging.info("Computing the metrics")
            # compute the metrics
            I_train, I_test, mu_test = compute_paper_metrics(est_callback_train.densities,est_callback_test.densities)
            self.results = {
                "I_train": I_train,
                "I_test": I_test,
                "mu_test": mu_test,
                "densities_train": est_callback_train.densities,
                "densities_test": est_callback_test.densities,
                "main model history": history,
                "submodels_I": submodels_I,
            }
        else:
            logging.info("Training the main model for multiple iterations")
            results = []
            for i in range(iterations_mutiple_train):
                logging.info(f"Iteration {i+1}")
                logging.info("Training the main model")
                # Load the main model
                if 'model_kwargs' not in config["model"].keys():
                    model = models.load_model(config["model"]["name"])
                else:
                    model = models.load_model(config["model"]["name"],**config["model"]["model_kwargs"])
                
                # Train the main model
                est_callback_train = EstimatedDensity(data = (data.X_train, data.y_train), submodels_predictions=None, n_classes=n_classes)
                est_callback_test = EstimatedDensity(data = (data.X_test, data.y_test), submodels_predictions=submodels_predictions, n_classes=n_classes)

                model_fit_kwargs = {} if "fit_kwargs" not in config["model"].keys() else config["model"]["fit_kwargs"]
                model_fit_kwargs["epochs"] = 20 if "epochs" not in model_fit_kwargs.keys() else model_fit_kwargs["epochs"]
                model_fit_kwargs["batch_size"] = 32 if "batch_size" not in model_fit_kwargs.keys() else model_fit_kwargs["batch_size"]
                history = model.fit(data.X_train, data.y_train, callbacks=[est_callback_train,est_callback_test] + additional_callbacks, **model_fit_kwargs)
                logging.info("Main model training completed.")
                logging.info("Computing the metrics")
                # Plot the metrics
                I_train, I_test, mu_test = compute_paper_metrics(est_callback_train.densities,est_callback_test.densities)
                results.append({
                    "I_train": I_train,
                    "I_test": I_test,
                    "mu_test": mu_test,
                    "densities_train": est_callback_train.densities,
                    "densities_test": est_callback_test.densities,
                    "main model history": history,
                    "submodels_I": submodels_I,
                })
            self.results = results

    def _run_with_infinity_strategy(self,additional_callbacks,save_path_submodels, n_classes) -> None:
        with tempfile.TemporaryDirectory() as traindirname, tempfile.TemporaryDirectory() as testdirname:
            logging.info("Starting the experiment...")
            config = self.config
            data = datasets.load_dataset(config["dataset"])
            data = convert_to_tensor(data)

            # Load the submodels
            logging.info("Loading submodels...")	
            submodels = []
            submodels_fit_kwargs = []

            for submodel_config in config["submodels"]:
                submodels_fit_kwargs.append({} if "fit_kwargs" not in submodel_config.keys() else submodel_config["fit_kwargs"])
                if 'model_kwargs' not in submodel_config.keys():
                    submodels.append(models.load_model(submodel_config["name"]))
                else:
                    submodels.append(models.load_model(submodel_config["name"],**submodel_config["model_kwargs"]))

            if 'model_kwargs' not in config["model"].keys():
                model = models.load_model(config["model"]["name"])
            else:
                model = models.load_model(config["model"]["name"],**config["model"]["model_kwargs"])
            
            logging.info("Training the main model")
            model_fit_kwargs = {} if "fit_kwargs" not in config["model"].keys() else config["model"]["fit_kwargs"]
            model_fit_kwargs["epochs"] = 20 if "epochs" not in model_fit_kwargs.keys() else model_fit_kwargs["epochs"]
            model_fit_kwargs["batch_size"] = 32 if "batch_size" not in model_fit_kwargs.keys() else model_fit_kwargs["batch_size"]
            
            save_intermediate_predictions_train = SaveIntermediatePredictions(traindirname, (data.X_train, data.y_train))
            save_intermediate_predictions_test = SaveIntermediatePredictions(testdirname, (data.X_test, data.y_test))
            history = model.fit(data.X_train, data.y_train, callbacks=[save_intermediate_predictions_train,save_intermediate_predictions_test] + additional_callbacks, **model_fit_kwargs)
            X_train = data.X_train
            y_train = model.predict(data.X_train)
            if n_classes == 2:
                y_train = tf.convert_to_tensor(y_train > 0.5, dtype=tf.uint8)
            else:
                y_train = tf.convert_to_tensor(np.argmax(y_train, axis=-1), dtype=tf.uint8)

            # Train and evaluate the submodels	
            logging.info("Training and evaluating submodels")
            submodels_predictions = []
            submodels_I = [] 
            for submodel, submodel_fit_kwargs,i in zip(submodels, submodels_fit_kwargs,range(len(submodels))):
                logging.info(f"Training submodel")
                submodel_fit_kwargs["epochs"] = 10 if "epochs" not in submodel_fit_kwargs.keys() else submodel_fit_kwargs["epochs"]
                submodel_fit_kwargs["batch_size"] = 32 if "batch_size" not in submodel_fit_kwargs.keys() else submodel_fit_kwargs["batch_size"]
                submodel.fit(X_train, y_train, **submodel_fit_kwargs)
                logging.info(f"Evaluating submodel")
                y_pred_sub = submodel.predict(data.X_test)
                if n_classes == 2:
                    y_pred_sub = np.array(y_pred_sub > 0.5, dtype="int")
                else:
                    y_pred_sub = np.argmax(y_pred_sub, axis=-1)
                submodels_predictions.append(y_pred_sub)  
                density = est_density(y_pred_sub, data.y_test.numpy(), n_classes=n_classes)
                submodels_I.append(I_XY(density))
                if save_path_submodels is not None:
                    submodel.save(Path(save_path_submodels) / f"submodel_{i}.keras")

            logging.info("Submodels training and evaluation completed.")

            est_densities_train = []
            est_densities_test = []

            main_model_predictions = np.load(Path(traindirname) / f"epoch_0.npy")
            est_densities_train.append(est_density(main_model_predictions, data.y_train.numpy(), n_classes=n_classes))

            main_model_predictions = np.load(Path(testdirname) / f"epoch_0.npy")
            densities_epoch = []
            for submodel_predictions in submodels_predictions:
                densities_epoch.append(est_density(main_model_predictions, submodel_predictions, data.y_test.numpy(), n_classes=n_classes))
            est_densities_test.append(densities_epoch)	

            for i in range(1,model_fit_kwargs["epochs"]+1):
                main_model_predictions = np.load(Path(traindirname) / f"epoch_{i}.npy")
                est_densities_train.append(est_density(main_model_predictions, data.y_train.numpy(), n_classes=n_classes))

                main_model_predictions = np.load(Path(testdirname) / f"epoch_{i}.npy")
                densities_epoch = []
                for submodel_predictions in submodels_predictions:
                    densities_epoch.append(est_density(main_model_predictions, submodel_predictions, data.y_test.numpy(), n_classes=n_classes))
                est_densities_test.append(densities_epoch)
            
            I_train, I_test, mu_test = compute_paper_metrics(est_densities_train,est_densities_test)
            self.results = {
                "I_train": I_train,
                "I_test": I_test,
                "mu_test": mu_test,
                "densities_train": est_densities_train,
                "densities_test": est_densities_test,
                "main model history": history,
                "submodels_I": submodels_I,
            }            
    
    def plot_results(self,save=False):
        if self.results is None:
            raise Exception("You need to run the experiment first")
        config = self.config
        results = self.results
        if type(results) is dict: # single train
            plot_paper_metrics(results,config["plot"]["submodels_names"],save=True)
        else:
            plot_paper_metrics_with_ci(results,config["plot"]["submodels_names"],save=True)
        

    def save_results(self,path):
        if self.results is None:
            raise Exception("You need to run the experiment first")
        # save results with pickle
        with open(Path(path), 'wb') as f:
            pickle.dump(self.results, f)

    def load_results(self,path):
        with open(Path(path), 'rb') as f:
            self.results = pickle.load(f)

# utils
def quick_config_scan(config):
    assert set(["dataset","model","submodels","plot"]).issubset(config), "The config is not valid"
    assert len(config["submodels"]) == len(config["plot"]["submodels_names"]), "The config is not valid"

def convert_to_tensor(data):
    data.X_train = tf.convert_to_tensor(data.X_train,dtype=tf.float32)
    data.X_test = tf.convert_to_tensor(data.X_test,dtype=tf.float32)
    data.y_train = tf.convert_to_tensor(data.y_train,dtype=tf.uint8)
    data.y_test = tf.convert_to_tensor(data.y_test,dtype=tf.uint8)
    return data

def compute_paper_metrics(densities_train,densities_test):
    I_train = []
    I_test = []
    mu_test = []
    for i in range(len(densities_train)):
        I_train.append(I_XY(densities_train[i]))
        I_test.append(I_XY(densities_test[i][0], idx=(0,2)))
        mu_test_epoch = []
        for j in range(len(densities_test[0])):
            mu_test_epoch.append(mu_metric(densities_test[i][j]))
        mu_test.append(mu_test_epoch)
    return np.array(I_train), np.array(I_test), np.array(mu_test)

def plot_paper_metrics(results,submodels_names,save=False):
    I_train, I_test, mu_test, submodels_I = results["I_train"], results["I_test"], results["mu_test"], results["submodels_I"]
    epochs = np.arange(len(I_train))
    plt.plot(epochs, I_train, label="I(F;Y_train)")
    plt.plot(epochs, I_test, label="I(F;Y_test)")
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(submodels_names))))
    for j, submodel_name in enumerate(submodels_names):
        c = next(colors)
        plt.plot(epochs, mu_test[:,j], label="mu "+submodel_name, color=c)
        plt.hlines(submodels_I[j], 0, max(epochs), linestyles='dashed', color=c)
        plt.text(max(0,max(epochs)-10), submodels_I[j], f"best {submodel_name}", color=c)
    plt.xlabel("epochs")
    plt.legend()
    if save:
        plt.savefig("paper_metrics.png")
    plt.show()

def plot_paper_metrics_with_ci(results,submodels_names,save=False):
    ## plot with confidence interval of 1 std
    I_train = np.array([result["I_train"] for result in results])
    I_test = np.array([result["I_test"] for result in results])
    mu_test = np.array([result["mu_test"] for result in results])
    submodels_I = results[0]["submodels_I"]

    I_train_mean = np.mean(I_train,axis=0)
    I_train_std = np.std(I_train,axis=0)
    I_test_mean = np.mean(I_test,axis=0)
    I_test_std = np.std(I_test,axis=0)
    mu_test_mean = np.mean(mu_test,axis=0)
    mu_test_std = np.std(mu_test,axis=0)

    epochs = np.arange(len(I_train_mean))
    plt.plot(epochs, I_train_mean, label="I(F;Y_train)")
    plt.fill_between(epochs, I_train_mean-I_train_std, I_train_mean+I_train_std, alpha=0.5)
    plt.plot(epochs, I_test_mean, label="I(F;Y_test)")
    plt.fill_between(epochs, I_test_mean-I_test_std, I_test_mean+I_test_std, alpha=0.5)
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(submodels_names))))
    for j, submodel_name in enumerate(submodels_names):
        c = next(colors)
        plt.plot(epochs, mu_test_mean[:,j], label="mu "+submodel_name, color=c)
        plt.fill_between(epochs, mu_test_mean[:,j]-mu_test_std[:,j], mu_test_mean[:,j]+mu_test_std[:,j], alpha=0.5, color=c)
        plt.hlines(submodels_I[j], 0, max(epochs), linestyles='dashed', color=c)
        plt.text(max(0,max(epochs)-10), submodels_I[j], f"best {submodel_name}", color=c)
    plt.xlabel("epochs")
    plt.legend()
    if save:
        plt.savefig("paper_metrics.png")
    plt.show()


