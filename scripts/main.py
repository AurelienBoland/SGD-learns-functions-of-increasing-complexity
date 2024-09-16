import argparse
import yaml
from sgd_complexity.pipelines import ExperimentPipeline
from sgd_complexity.tensorflow.callbacks import SaveIntermediateModels
# compute execution time
import time

parser = argparse.ArgumentParser()
parser.add_argument("config", help="path to config file")
parser.add_argument("--multiple_train", help="train the model multiple times", action="store_true")
parser.add_argument("--iterations_multiple_train", help="number of iterations", type=int, default=5)
parser.add_argument("--save_path", help="path to save the results", default=None)
parser.add_argument("--models_path", help="path to save the intermediate models", default=None)
parser.add_argument("--save_period", help="period to save the intermediate models", type=int, default=5)

args = parser.parse_args()
# Load yaml config file into a dictionary
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

pipeline = ExperimentPipeline(config)
start = time.time()
if args.models_path is not None:
    additional_callbacks = [SaveIntermediateModels(args.models_path,args.save_period)]
    pipeline.run(mutiple_train=args.multiple_train,iterations_mutiple_train=args.iterations_multiple_train,additional_callbacks=additional_callbacks,save_path_submodels=args.models_path)
else: 
    pipeline.run(mutiple_train=args.multiple_train,iterations_mutiple_train=args.iterations_multiple_train)
print("Execution time: ", time.time()-start)
pipeline.plot_results(save=True)
if args.save_path is not None:
    pipeline.save_results(path=args.save_path)












