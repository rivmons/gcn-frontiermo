# Graph Convolutional Neural Network-Enabled Frontier Molecular Orbital Prediction: A Case Study with Neurotransmitters and Antidepressants 

A repository that enables the training of a GCN-ANN based regressor that is able to predict HOMO and LUMO values with comprehensible, substructure-based, differentiable fingerprints.

## steps to run:

#### make sure you have all the necessary python libraries installed with `pip install -r requirements.txt`

1) **use `python createModels.py ...` to populate the src folder with training scripts**
* `python createModels.py -hl/--homolumo <homo/lumo>`
* hl = {homo, lumo} to designate the energies you want to train on; the resulting directory will be named the same
* currently, the set of hyperparameters used to train are sampled from sets of individual hyperparameters (i.e. learning rate, etc.)
    * these can be easily changed in createModels.py to reflect the type of hyperparameters to explore and the number of models to train
</br><br/>

2) **use `sbatch slurmtrain.sh` if using slurm**
* fill in slurm configuration according to resources available, etc.
* on the last line, make sure to fill in the path to the log file as well as path to the scripts. Use `$SLURM_ARRAY_TASK_ID` to vary names of log and model based on slurm array
* **otherwise, run the scripts in `<directory>/trainingJobs` manually**
<br/><br/>

3) **individual model results**
* the `<homo/lumo>` folder will be populated with a `logs` folder for logging while training each model
* additionally, there will be subfolder of form `<model{i}>` that include results after training each model
    * these folders include:
        * `testset.txt`: the test set sampled in a .txt file
        * `rloss.png`: a graph showing tracked metrics while training
        * `checkpoint.pth`: the saved model architecture and parameters
        * `loss.txt`: text-based file that saves training and validation losses 
<br/><br/>

4) **ensemble training (optional)**
* an ensemble using a variable number of previously trained models can be trained using `ensemble_train.py`
* `python ensemble_train.py -n/--num_models <number of individual models to ensemble> -hl/--homolumo <homo/lumo>`
    * hl = {homo, lumo}; gives directory name and energies to train on
    * num_models is the number of models you want to incorporate from the trained model directory in the ensemble
* results in a directory named `ensemble_<homo/lumo>` which contains:
    * `testset.txt`: the test set sampled in a .txt file
    * `rloss.png`: a graph showing tracked metrics while training
    * `checkpoint.pth`: the saved model architecture and parameters
    * `loss.txt`: text-based file that saves training and validation losses 
<br/><br/>

5) **calculate metrics**
* `reg_eval.py` can be used to obtain metrics for each model and/or ensemble
    * `python reg_eval.py -e/--ensemble <0/1> -m/--models <list of model numbers to eval> -d/--directory <directory of model(s)>`
        * e = {0, 1}; required; if evaluating an ensemble, 1, else 0
        * m = string of model numbers that you want to evaluate; only possible for non-ensembles
            * ex) `-m 1,2,4,6,7`
        * d = directory of models or ensemble
    * outputs:
        * prints out list of metrics 
        * `<directory>_<mn>.png` graph of predicted vs true energies
        * `<directory>_<mn>_preds.csv` .csv of predicted vs true energies

6) **substructure analysis for test sets**
* `evaluate.py` contains code for generating most relevant substructures for the prediction of a certain energy across test set
* `python evaluate.py -hl/--homolumo <homo/lumo> -mn/--model <model number>`
    * hl = {homo, lumo}: the directory name and energy type being predicted
    * mn = the model number you want predictions for
* these substructures will be generated for a certain model's test dataset which is outputted to `testset.txt`
* only currently implemented for individual models 
* outputs a directory named `substructure_activations`:
    * prints log of what is represented in `activations.txt`
    * `fp_<ith index>_<nth highest activation>.png`: image showing molecule with nth highest activation at ith fingerprint index
    * `activations.txt`: log for results including activation, atom, smile, radius, homo/lumo energies
* **you can change how many of the best substructures you want per index on line 145**

#### Acknowledgements to [Xuhan Liu](https://github.com/XuhanLiu/NGFP) for some of the code in this repository