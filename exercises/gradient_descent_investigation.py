import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from plotly.subplots import make_subplots
from IMLearn.model_selection import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    pass


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = [L1, L2]
    module_names = ["L1", "L2"]
    losses = []

    def callback_(w, val, grad, t, eta, delta):
        values.append(val)
        weights.append(w)

    for module in modules:
        for eta in etas:
            values = []
            weights = []
            GD = GradientDescent(FixedLR(eta), callback=callback_)
            f = module(init.copy())
            GD.fit(f, None, None)
            losses.append(pd.Series(values))
            plot_descent_path(module, np.asarray(weights)).show()

    rows = len(modules)
    cols = len(etas)
    fig = make_subplots(rows=rows, cols=cols, start_cell="bottom-left")
    for i in range(rows*cols):
        fig.add_trace(go.Line(x=losses[i].index, y=losses[i], mode="markers",
                                 marker=dict(color="Blue", colorscale="Bluered"), showlegend=False), row=i // cols + 1,
                      col=i % cols + 1)
        fig.update_xaxes(title_text=f"iterations ({module_names[i//cols]} with eta={etas[i % cols]})", row=i // cols + 1, col=i % cols + 1)

    fig.update_yaxes(title_text="loss")
    fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):

    def callback_(w, val, grad, t, eta, delta):
        values.append(val)
        weights.append(w)

    losses = []
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    for gamma in gammas:
        values = []
        weights = []
        GD = GradientDescent(ExponentialLR(eta, gamma), callback=callback_)
        f = L1(init.copy())
        GD.fit(f, None, None)
        losses.append(pd.Series(values))
        if gamma == .95:
            weights_95 = weights

    rows = 1
    cols = len(gammas)
    fig = make_subplots(rows=rows, cols=cols, start_cell="bottom-left")
    for i in range(rows * cols):
        fig.add_trace(go.Line(x=losses[i].index, y=losses[i], mode="markers",
                              marker=dict(color="Blue", colorscale="Bluered"), showlegend=False), row=i // cols + 1,
                      col=i % cols + 1)
        fig.update_xaxes(title_text=f"iterations (L1 with eta={eta}, gamma={gammas[i % cols]})",
                         row=i // cols + 1, col=i % cols + 1)

    fig.update_yaxes(title_text="loss")
    fig.show()


    # Plot descent path for gamma=0.95
    plot_descent_path(L1, np.asarray(weights_95)).show()

    weights = []
    GD = GradientDescent(ExponentialLR(eta, gamma), callback=callback_)
    f = L2(init.copy())
    GD.fit(f, None, None)
    losses.append(pd.Series(values))
    plot_descent_path(L2, np.asarray(weights)).show()


def load_data(path: str = "C:/Users/User/GitRepositories/IML.HUJI/datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve, auc
    from utils import custom
    GD = GradientDescent()
    f = LogisticRegression(solver=GD)
    f.fit(X_train.to_numpy(), y_train.to_numpy())
    y_prob = f.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_alpha = thresholds[np.argmax(tpr-fpr)]

    c = [custom[0], custom[-1]]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    print("best cutoff is", best_alpha)
    f.alpha_ = best_alpha
    print("using this cutoff, the test error is", f.loss(X_test, y_test))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    GD = GradientDescent(max_iter=2000, learning_rate=FixedLR(1e-4))

    scores = np.empty((1,0))
    for lam in lambdas:
        g = LogisticRegression(solver=GD, penalty="l1", lam=lam)
        validation_score, train_score = cross_validate(g, X_train, y_train, scoring=misclassification_error)
        scores = np.append(scores, validation_score)

    best_lam = lambdas[np.argmax(scores)]
    g = LogisticRegression(solver=GD, penalty="l1", lam=best_lam)
    g.fit(X_train, y_train)
    print("best lambda is", best_lam)
    print("test error for best lambda is", g.loss(X_test, y_test))

if __name__ == '__main__':
    np.random.seed(0)
    #compare_fixed_learning_rates()
    #compare_exponential_decay_rates()
    fit_logistic_regression()
