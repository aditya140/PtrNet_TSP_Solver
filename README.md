# PtrNet_TSP_Solver

# Google colab

* use the `TSP_train.ipynb` for training and testing.
* use the `TSP_run.ipynb` for inference.


# If you can not use colab
Preferred mode of running is Colab since it has a lot of libraries preinstalled
## How to train the model -
* Replicate the python libraries in the colab environment (use the `colab.txt` as requirements for pip)
* Run `pip install -r requirements.txt` to install  dependencies
* The dataset can be downloaded by running the `download_dataset.py` file.
* Specify the hyperparameters and the type of model you need to train along with the save locations in `hparams.py`
* Run `train.py` to train the model
* To train 5-20 model run `train_5-20.py`

## How to test the model - 
* The hyperparameters are provided in the `hparams.py` file.
* run the `test.py` file for testing and creating metric csv's

## How to generate result metrics - 
* Copy all the CSV's for all tests in Results file and rename them as per convention in `metrics.ipynb`
* Test results are already in the Folder (use them if you do not want to train the model)
* Use the `Results/metrics.ipynb` file to generate metrics.
* Modify the notebook to create metrics for 5 point model and 10 point model.
