# AutoML

Auto machine learning project with focus on predict time series using simple usage and high-level understanding over time series prediction methods.

## Contents

- [Install](#install)
- [How to use](#how-to-use)
- [Implementation details](#implementation-details)
    - [Algorithms](#algorithms)
    - [Data preparation](#data-preparation)
    - [Evaluation](#evaluation)

## Install

AutoML is supported on Python 3.6+. The recommended way to install AutoML is via setup.py.

```bash
python setup.py install
```

## How to use

The general use follows this structure:

```python
from automl import AutoML

ml = AutoML('/path/data.csv')
```

After the class perform its intern operations you can interact with AutoML:

```python
# predict the value for a date
ml.predict('02-03-2016')

# append new data to the historical set
ml.add_new_data('/path/new_data.csv', append=True)
```

## Implementation details

### Algorithms

Here follows the list of all the algorithms used inside the AutoML class:

- LightGBM

    The LightGBM is an algorithm based on [Gradient Boosting](https://projecteuclid.org/euclid.aos/1013203451), which is an ensemble of models that produces an improved model by fitting a new predictor with the residual error made by the previous predictor. He is composed of two techniques: the Gradient Based One-Sided Sampling, which excludes small proportions of data that have small gradients; and the Exclusive Feature Bundling, that bundle mutually exclusive features to reduce its number. These techniques aim to make the model more efficient and faster. More information can be found in his [paper](https://dl.acm.org/doi/10.5555/3294996.3295074).

### Data preparation

Once the data is read by the AutoML class, it pass through some operations to be consumable by the algorithms.

First, we look for NaN values on the data and fill them with two optional methods. One is the forward fill, where the NaN value is filled with the last valid data and the other is the back fill, in which the NaN value is filled with the next valid value.

After filling NaN values, we apply the Partial Autocorrelation Function ([PACF](https://en.wikipedia.org/wiki/Partial_autocorrelation_function)) in order to find the best number of time lags. This function calculates the correlation between each past lag of an observation, which describes the direct relationship between an observation and its lag. With the best number of lags, we insert them as features of each input. When defined the past lags every predict will need the same length of lags as input.

By the end, the data is prepared to be consumable to the algorithms, like data types and shape.

### Evaluation

During the evaluation process, all the algorithms are comparer with some metric values. Here we will describe the used metrics.

- Weighted Quantile Loss (wQL)

    The Weighted Quantile Loss metric measures the accuracy of predictions
    at a specified quantile. It is described by the following equation:

    <div style="text-align:center">
    <img src="https://docs.aws.amazon.com/forecast/latest/dg/images/metrics-quantile-loss.png" alt="wql" width="550"/>
    </div>

    Where:

    τ - a quantile in the set {0.01, 0.02, ..., 0.99}.

    qi,t(τ) - the τ-quantile that the model predicts.

    yi,t - the observed value at point (i,t).

- Weighted Absolute Percentage Error (WAPE)

    The Weighted Absolute Percentage Error metric measures the overall
    deviation of forecasted values from observed values. It is described by the following equation:

    <div style="text-align:center">
    <img src="https://docs.aws.amazon.com/forecast/latest/dg/images/WAPE.png" alt="wql" width="230"/>
    </div>

    Where:

    yi,t - the observed value at point (i,t)

    ŷi,t - the predicted value at point (i,t)

- Root Mean Square Error (RMSE)

    The Root Mean Squared Error have the same objective as the WAPE, which is measure the overall
    deviation of forecasted values from observed values. But the RMSE is more sensitve to outliers because of the squared value of the residuals. It is described by the following equation:

    <div style="text-align:center">
    <img src="https://docs.aws.amazon.com/forecast/latest/dg/images/metrics-rmse.png" alt="wql" width="230"/>
    </div>

    Where:

    yi,t - the observed value at point (i,t)

    ŷi,t - the predicted value at point (i,t)

    nT - the number of data points in a testing set