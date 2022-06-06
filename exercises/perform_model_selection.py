from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = pd.DataFrame(np.random.uniform(-1.2,2,1500), columns=['x'])
    f = lambda x : (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    eps = pd.Series(np.random.normal(0, 10, 1500))
    y_clean = f(X)
    X_train, y_train, X_test, y_test = split_train_test(X, y_clean.add(eps, axis=0), 2/3)

    X_train['y'] = y_clean.loc[y_train.index]
    X_train['set'] = "Train"
    X_test['y'] = y_clean.loc[y_test.index]
    X_test['set'] = "Test"
    plot_data = pd.concat([X_train, X_test])
    px.scatter(plot_data, x='x', y='y', color='set',
                title = "clean samples divided into train/data sets", labels={"set":"Set"}).show()
    X_train = X_train.drop(columns=['y','set'])
    X_test = X_test.drop(columns=['y', 'set'])

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    errors = pd.DataFrame(columns=['validation_error', 'train_error'])
    for k in range(11):
        PolyFit_k = PolynomialFitting(k)
        validation_score, train_score = cross_validate(PolyFit_k, X_train.to_numpy(), y_train.to_numpy(), scoring=mean_square_error)
        errors.loc[len(errors.index)] = [validation_score, train_score]
    fig = go.Figure([go.Scatter(name="train error",
                            x=errors.index, y=errors['train_error'], mode="markers",
                            marker=dict(color="red", size=5)),
                     go.Scatter(name="validation error",
                                x=errors.index, y=errors['validation_error'], mode="markers",
                                marker=dict(color="blue", size=5))],
                            layout=go.Layout(
                            title="cross-validation error for different polynomial degrees",
                            scene=dict(xaxis=dict(title="degree"),
                                     yaxis=dict(title="error"))))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = errors['validation_error'].idxmin()
    min_validation_error = np.min(errors['validation_error'])
    PolyFit = PolynomialFitting(k)
    PolyFit.fit(X_train, y_train)
    test_error = mean_square_error(PolyFit.predict(X_test.to_numpy()), y_test.to_numpy())
    print("best degree is ", k, "\n")
    print("test error = ", test_error, "\n")
    print("validation error = ", min_validation_error)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100,5)
