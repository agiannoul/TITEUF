# Adding your own methods


The framework support two types of methods, RUL models that trained on data with a real number target (the time-to-event),
and SA models that are trained on data with labels being time-to-event and censoring indicator, and predict Individual Survival Distribution (ISD).

## Implementing RUL methods

We will demonstrate how to add a new RUL method by an example. Suppose we want to add XGBoost regressor.

The class has to implement the `SupervisedMethodInterface` (from `OnlineADEngine.method.supervised_method.SupervisedMethodInterface`), which requires the following methods to be implemented:

```python
class XGBoostRUL(SupervisedMethodInterface):
```

* `__init__(self, params: dict)`: constructor that takes parameters of the model

Please note that the constructor must take `event_preferences` as a named argument, 
and save_model must be class property (ignore for now). we coud explicitly 
So the signature should be like this:


```python
def __init__(self, event_preferences: EventPreferences,save_model=False, *args, **kwargs):
    super().__init__(event_preferences=event_preferences)
    self.model_per_source = {}
    self.initial_args = args
    self.initial_kwargs = kwargs
    self.save_model=save_model
```

* Fitting the model

In fit method we train the model on the training data. In our framework we account
for cases where someone wants to train different models for different data sources 
(or a single model for a group of sources). This is handled by the framework in background. 
So we have to keep a dictionary of models, where the key is the source id.
Here instead of XGBoost we can use any other regression model from sklearn or other libraries.


```python
import xgboost as xgb

 def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame,
            anomaly_ranges: list[list]) -> None:
        """
        This method is used to fit a model in supervised way (training), where the data are passed in form
        of Dataframes along with their respected source and labels.

        :param historic_data: a list of Dataframes (used to fit a semi-supervised model). The `historic_data` list parameter elements should be copied if a corresponding method needs to store them for future processing
        :param historic_sources: a list with strings (names) of the different sources
        :param event_data: event data that are produced from the different sources
        :param anomaly_ranges: labels. It is a list of lists, where each inner list corresponds to a source and contains the labels for the data in that source.
        :return: None.
        """
        for current_historic_data, current_historic_source, labels in zip(historic_data, historic_sources,
                                                                          anomaly_ranges):
            print(current_historic_data.shape)
            self.model_per_source[current_historic_source] = xgb.XGBRegressor(*self.initial_args,
                                                                               **self.initial_kwargs)
            self.model_per_source[current_historic_source].fit(current_historic_data, labels)

```

* Predicting RUL values

After training, the system will call the `predict` method to get predictions on the target data (on the correspoding model).
For RUL models this method need to return a list of floats (the predicted RUL values):

```python
    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        predictions= self.model_per_source[source].predict(target_data)[:].tolist()
        predictions= [x for x in predictions]
        return predictions
```

* Logging model parameters

Since the framework supports hyperparameter optimization and logging of model parameters,
we need to implement the `get_params` method that returns a dictionary of model parameters,
and __str__ method that returns a string representation of the method.

```python
    def get_params(self) -> dict:
        return {
            **(xgb.XGBRegressor(novelty=False, *(self.initial_args), **(self.initial_kwargs)).get_params()),
        }


    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "XGBOOST"
```

Refer to other examples in models folder (sklearn_wraper.py, CatBoost_W.py,sktime_wrapper.py)

## Implementing SA method

We will demonstrate how to add a new SA method by an example. 
Suppose we want to add CoxPH model using its sksurv implementation.

The class has to implement the `SupervisedMethodInterface` (from `OnlineADEngine.method.supervised_method.SupervisedMethodInterface`), which requires the following methods to be implemented:

```python
class CoxPH(SupervisedMethodInterface):
```

* `__init__(self, params: dict)`: constructor that takes parameters of the model

Please note that the constructor must take `event_preferences` as a named argument, 
and save_model must be class property (ignore for now). we coud explicitly 
So the signature should be like this:

The only extra parameter we have to implement here is `avail_times_per_source`,
which is used to store the available times for each source (needed for ISD prediction).


```python
def __init__(self, event_preferences: EventPreferences,save_model=False, *args, **kwargs):
    super().__init__(event_preferences=event_preferences)
    self.model_per_source = {}
    self.initial_args = args
    self.initial_kwargs = kwargs
    self.save_model=save_model
    self.avail_times_per_source = {}
```

* Fitting the model

In fit method we train the model on the training data. In our framework we account
for cases where someone wants to train different models for different data sources 
(or a single model for a group of sources). This is handled by the framework in background. 
So we have to keep a dictionary of models, where the key is the source id.
Here instead of CoxPH we can use other SA models as well.

Labels are in form of tuples (time-to-event, event indicator), depending on the model in use,
and its implementation, we might need to convert them to a specific format. 

For each model we train we also keep the unique available times in the training data for each source,
which will be used during ISD prediction.

```python
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame,
            anomaly_ranges: list[list]) -> None:
        """
        This method is used to fit a anomaly detection model in supervised way (training), where the data are passed in form
        of Dataframes along with their respected source and labels.

        :param historic_data: a list of Dataframes (used to fit a semi-supervised model). The `historic_data` list parameter elements should be copied if a corresponding method needs to store them for future processing
        :param historic_sources: a list with strings (names) of the different sources
        :param event_data: event data that are produced from the different sources
        :return: None.
        """

        for current_historic_data, current_historic_source, labels in zip(historic_data, historic_sources,
                                                                          anomaly_ranges):
            print(current_historic_data.shape)
            from sksurv.util import Surv
            ydf=pd.DataFrame({'event': [lb[1] for lb in labels],'RUL': [lb[0] for lb in labels]})
            y = Surv.from_dataframe("event", "RUL", ydf)

            # RandomSurvivalForest(n_estimators=100, min_samples_split=6, min_samples_leaf=5, verbose=1, n_jobs=4)
            self.model_per_source[current_historic_source] = CoxPHSurvivalAnalysis(*self.initial_args,**self.initial_kwargs)
            self.model_per_source[current_historic_source].fit(current_historic_data, y)
            self.avail_times_per_source[current_historic_source]=np.unique([ty for ty in y['RUL']])

```

* Predicting ISD values

After training, the system will call the `predict` method to get predictions 
on the target data (on the corresponding model).
For SA models we return a 3D numpy array of shape (n, 2, T), 
where n is the number of samples in target_data,
T is the number of unique available times in training data for the source,
and the second dimension contains two arrays:

a) The first array is the predicted survival probabilities for each time in T

b) The second array is the corresponding time values in T

```python
    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame):
        predictions = self.model_per_source[source].predict_survival_function(target_data, True)
        n, T = predictions.shape
        # Repeat the time array for every curve → shape (n, T)
        times_tiled = np.tile(self.avail_times_per_source[source], (n, 1))
        # Stack into (n, 2, T)
        result = np.stack([predictions, times_tiled], axis=1)
        return result
```

* Logging model parameters

Since the framework supports hyperparameter optimization and logging of model parameters,
we need to implement the `get_params` method that returns a dictionary of model parameters,
and __str__ method that returns a string representation of the method.

```python
    def get_params(self) -> dict:
        params = {}
        for i, arg in enumerate(self.initial_args):
            params[f"arg{i}"] = arg
        # include keyword args normally
        params.update(self.initial_kwargs)

        return params


    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "CoxPH"
```


Refer to other examples in models folder (DeepHit.py, GradientBoosting.py)
