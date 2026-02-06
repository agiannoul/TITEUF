# Running on Your Own Dataset

## Loading Your Own Dataset

To create a compatible dataset, you need to initialize the `Dataset` class from the `OnlineADEngine.data.dataset` module.
which requires the following parameters:

-  **data**: a pandas dataframe
-  **datetime_column**: the column name of the datetime column in data
-  **source_column**: the column name of the source column in data
-  **event_indicator**: the column name of the event indicator column in data
-  **label_column**: a column with time to event labels (if available)
-  **train_sources**: float or list of source names for training eg. 0.6 or ['source1','source2']
-  **val_sources**: float or list of source names for validation eg. 0.2 or ['source1','source2']
-  **test_sources**: float or list of source names for test eg. 0.2 or ['source1','source2']

```python
handler = Dataset(data, datetime_column, event_indicator=event_indicator, source_column=source_column,
                      train_sources=train_sources, val_sources=val_sources, test_sources=test_sources)
```

Given the we can create RUL or SA datasets variants using the following methods 
(dataset contain the training and validation sets, while test_dataset contains the training and test set):
```python
dataset,test_dataset=handler.get_SA_dataset(keep_identifiers=to_keep_identifiers[method_name])
dataset,test_dataset=get_rul_dataset(keep_identifiers=to_keep_identifiers[method_name])
```

## Running SA method on Your Own Dataset
We have to define the parameter space of methods to use, lets say we want to use CoxPH and RandomSurvivalForest methods:

```python
param_space_configurations={
    "RSF":[{
        'n_estimators': [20,30,50],
        'min_samples_split': [10,15],
        'min_samples_leaf': [10,15],
        'max_features': ['sqrt'],
        'n_jobs': [4],
        'random_state': [42],
        'verbose': [1]
    }],
    "CoxPH":[{
        'alpha': [0.1],
        'ties': ['breslow', 'efron'],
        'n_iter': [100,150],
        'tol': [1e-8],
        'verbose': [1]
    }],
}
```

Run for **CoxPH** method on your own dataset:
- optimization_param:  which evaluation metric to optimise
- maximize: if we want to maximize (True) or minimize (False) the metric
- MAX_RUNS: the number of optimisation steps
- datasetname: a name for your dataset to be logged in mlflow
- keep_identifiers: whether to keep source identifiers in the dataset (needed for some methods like RDSM)

```python
from models.CoxModel import CoxPH
method_class = CoxPH
method_name = "CoxPH"
dataset, test_dataset = load_dataset_surv(keep_identifiers=False)
datasetname="YourDatasetName"
param_space_dict_per_method = param_space_configurations["CoxPH"]

run_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name, optimization_param="IBS",maximize=False,MAX_RUNS=20,datasetname=datasetname)
```
Run for **RSF** method on your own dataset:

```python
from models.RSF import RSF
method_class = RSF
method_name = "RSF"
dataset, test_dataset = load_dataset_surv(keep_identifiers=False)
datasetname="YourDatasetName"
param_space_dict_per_method = param_space_configurations["RSF"]

run_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name, optimization_param="IBS",maximize=False,MAX_RUNS=20,datasetname=datasetname)
```
