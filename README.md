# PtrNet_TSP_Solver

# Google colab - 
* use the TSP_train.ipynb for training.
* use the TSP_run.ipynb for inference

# How to train the model -
* The dataset can be downloaded by running the `download_dataset.py` file.
* Specify the hyperparameters and the type of model you need to train along with the save locations in `hparams.py`
* Run `train.py` to train the model
* To train 5-20 model run `train_5-20.py`

# How to test the model - 
* The hyperparameters are provided in the `hparams.py` file.
* run the `test.py` file for testing and creating metric csv's

# How to generate result metrics - 
* Use the `metrics.ipynb` file to generate metrics.
* Modify the notebook to create metrics for 5 point model and 10 point model.
