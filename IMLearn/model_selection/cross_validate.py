from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    m = X.shape[0]
    remainder = m % cv
    ceiling = m // cv + 1
    floor = m // cv
    ranges = [i * ceiling for i in range(remainder + 1)]
    ranges += [remainder * ceiling + i * floor for i in range(1, cv - remainder + 1)]

    train_score = validation_score = 0
    for i in range(cv):
        start = ranges[i]
        end = ranges[i+1]
        train_set = np.concatenate((X.take(indices=range(0,start), axis=0), X.take(indices=range(end,m), axis=0)))
        train_response = np.concatenate((y.take(indices=range(0,start), axis=0), y.take(indices=range(end,m), axis=0)))
        validation_set = X.take(indices=range(start,end), axis=0)
        validation_response = y.take(indices=range(start,end), axis=0)

        estimator_i = estimator.fit(train_set, train_response)
        y_validation_pred = estimator_i.predict(validation_set)
        y_train_pred = estimator_i.predict(train_set)

        validation_score += scoring(y_validation_pred, validation_response)
        train_score += scoring(y_train_pred, train_response)

    return validation_score/cv, train_score/cv