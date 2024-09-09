# Code for Achterhold, Jan and Stueckler, Joerg: "Explore the Context: Optimal Data Collection for Context-Conditional Dynamics Models"
This repository contains code corresponding to:

Achterhold, Jan and Stueckler, Joerg: \
**Explore the Context: Optimal Data Collection for Context-Conditional Dynamics Models**\
24th International Conference on
Artificial Intelligence and Statistics (AISTATS), 2021

Project page: https://explorethecontext.is.tue.mpg.de/ \
Full paper: https://arxiv.org/abs/2102.11394

If you use the code, data or models provided in this repository for your research, please cite our paper as:
```
@inproceedings{achterhold2021_explorethecontext,
  title = {Explore the Context: Optimal Data Collection for Context-Conditional Dynamics Models},
  author = {Achterhold, Jan and Stueckler, Joerg},
  booktitle = {Accepted for publication at the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year = {2021},
  note = {CoRR abs/2102.11394},
  url = {https://arxiv.org/abs/2102.11394}
}
```

## To reproduce the results in the paper, proceed as follows:

### 1. Install environment
Set-up a Python 3.6.9 environment and install requirements via `pip install -r requirements.txt`.  
We used `CUDA 10.2, CuDNN 8.0.2` for our experiments.  
Please install `torch-scatter` according to https://github.com/rusty1s/pytorch_scatter#installation.  
If you also use PyTorch 1.6.0 with CUDA 10.2, run
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
```

All python commands of the form `python -m ...` should be executed from the root directory of this repository.  
**Important note:** Training data will be stored in `data/`, trained models in `experiments/`.  
Please be aware that data and experiments take up several GB of disk space,  
so you may symlink these directories to somewhere
where you can store large amounts of data.

### 2. Generate training data
You can either generate the training data yourself or download it from \
https://keeper.mpdl.mpg.de/f/b0780b562ffb4367898f/?dl=1 (you can verify the checksum in `sha1sums`). \
Extract the downloaded tar archive into the `./data/` directory of this repository (you may have to create this first). \
To generate the data yourself, run
```
python -m context_exploration.data.data_generation
```


### 3. Train models
You can either train the models yourself or download pre-trained (and calibrated) models from \
https://keeper.mpdl.mpg.de/f/e0463fd6b68c45cbaf0b/?dl=1 (you can verify the checksum in `sha1sums`). \
Extract the downloaded tar archive into the `./experiments/` directory of this repository (you may have to create this first). \
To generate the data yourself, first create the experiments directory 
```
mkdir -p experiments/train_model
```
in the root directory of this repository.
Then, run all commands in `jobs/jobs_training_cr.txt`.

### 4. Run calibration (not necessary for already downloaded models)
If you trained the models yourself, you have to run the calibration experiments.
For this, run
```
cd jobs; python generate_jobs_calibration.py
```
Then run all commands in `jobs/jobs_calibration_cr.txt`.

### 5. Run swingup experiment (not necessary for already downloaded models)
To run swingup trials on the calibrated, _learned_ pendulum model, run
```
python -m context_exploration.evaluation.pendulum_swingup
```
The nominal return on a _ground-truth_ model is ≈-215, as given by
```
python -m context_exploration.data.pendulum_cem
```

### 6. Evaluate calibration using notebooks
All plots are generated from the calibration results in the following notebooks
```
cd context_exploration/evaluation/notebooks; jupyter notebook
```
To generate the ablation plots, generate calibration jobs with
` WITH_ABLATIONS = True ` in `jobs/generate_jobs_calibration.py`.


## Errata
May 2024: We have discovered an error in the implementation, such that, in contrast to what is reported in (Achterhold & Stueckler, 2021), not the models with the minimal validation loss are used for the final evaluation, but those after training has finished (after a fixed number of steps). We have fixed the error in the [bugfix_modelselection](https://github.com/EmbodiedVision/explorethecontext/tree/bugfix_modelselection) branch, including notebooks with updated results. Qualitatively, we observe that the two variants yield similar results.



## License

See [LICENSE.md](LICENSE.md).

For license information on 3rd-party software we use in this project, see [3RD_PARTY_LICENSES.md](3RD_PARTY_LICENSES.md).
