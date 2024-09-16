Replication of experiments from Nakkiran, Preetum, et al. "Sgd on neural networks learns functions of increasing complexity." arXiv preprint arXiv:1905.11604 (2019).

The code extends the experiments of the original paper to multiclass and partially linear networks (as investigated in the paper from Pinson, Hannah, et al. "Exploring the development of complexity over depth and time in deep neural networks." High-dimensional Learning Dynamics 2024: The Emergence of Structure and Reasoning. )
## Getting started
### Install requirements (ideally in a virtual environment)
```
pip install -r requirements.txt
```
### Install custom package
```
pip install -e .
```
remark: the `-e` is not mandatory but it is adviced if you want to work on the custom package
### Run demo script
```
python scripts/main.py experiments/demo.yml
```
You can also work inside a jupyter notebook as done in `notebooks/demo.ipynb`. Other experiment configurations can be found in the folder `experiments` (such as the one that correponds to the paper)
## Features
- [x] Train a compare to linear classfier
- [x] Train a compare to mutiple submodels
- [x] Define your experiment in a YAML file
- [x] Run an experirement mutiple times to get a confidence interval
- [ ] Option for training a submodel on $F_{T_0}$ or  $F_{\infty}$ predictions (see paper for more details)
- [ ] Experiment tracking mechanism (Should we simply use notebooks, should we pickle experiments results or should we use a dedicated framework like MLflow ?)

## Note
1. Graphviz should be installed to plot keras models (`sudo apt install graphviz`)  