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
    remainder = m % cv
    ceiling = m // cv + 1
    floor = m // cv
    ranges = [i * ceiling for i in range(remainder + 1)]
    ranges += [remainder * ceiling + i * floor for i in range(1, cv - remainder + 1)]

    train_score = validation_score = 0
    for i in range(cv):
        start = ranges[i], end = ranges[i+1]
        train_set = X[0:start, :]
        response = y[end:m]

        estimator_i.fit(train_set, response)
        y_pred = estimator_i.predict(X[start:end, :])

        error_i = scoring(y[start:end], y_pred)
        train_score += error_i[0]
        validation_score += error_i[1]

    return train_score/cv, validation_score/cv

if __name__ == '__main__':
    pass
    # m = 96
    # cv = 7
    # remainder = m%cv
    # ceiling = m//cv + 1
    # floor = m // cv
    # ranges = [i*ceiling for i in range(remainder+1)]
    # ranges += [remainder*ceiling + i*floor for i in range(1,cv-remainder+1)]
    #
    # print(ranges)
    #
    # for i in range(cv):
    #     start = ranges[i]
    #     end = ranges[i + 1]
    #     print("level ", i, "\n")
    #     print(0, start, "\n")
    #     print(end, m, "\n")
    #     print(start, end, end-start,  " fold size" ,"\n------------\n")

    # X = np.arange(12)
    # y = np.arange(10,19)
    # folds = np.array_split(X, 3)
    # responses = np.array_split(y, 3)
    # for a in folds:
    #     print(a)
    # for b in responses:
    #     print(b)
