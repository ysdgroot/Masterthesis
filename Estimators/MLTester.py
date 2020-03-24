from sklearn.model_selection import cross_val_score
# from keras.models import Sequential


def time_to_fit(regression_model, training_data: list, solutions_training_data: list):
    """
    To look how fast the training process goes for the model

    :param regression_model: Regression Model from sklearn or the Keras
    :param training_data: A 2D-list or pandas dataframework.
    :param solutions_training_data: The real values of each training_data
    :return: tuple,
        First the fitted model
        Second the total time consumed to fit the model.
    """
    import time

    # if type(regression_model) == Sequential:
    #     print("Het is een neuraal netwerk")

    start = time.time()
    model_fitted = regression_model.fit(training_data, solutions_training_data)
    end = time.time()

    total_time = end - start
    return model_fitted, total_time


def performance_model(fitted_model, test_data: list, solutions_test_data: list):
    """
    Test the performance of the fitted model, for other data than the training data, namely test data.
    The scores are calculated for different measurements.

    :param fitted_model: the (already) fitted regression model. (Sklearn object or Keras)
    :param test_data: a numpy.array of shape (n_datapoints, n_features)
    :param solutions_test_data: a numpy.array of shape (n_datapoints,)
                                these values represents the true solutions of the test data
    :return: 1) score: The R^2 of the model (For regression models)
            2) list with values of the 'mean_squared_error', 'root_mean_squared_error',
                'absolute_error' and the 'max_error'
    """

    from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
    import math

    pred_y = fitted_model.predict(test_data)

    score = fitted_model.score(test_data, solutions_test_data)

    mse = mean_squared_error(solutions_test_data, pred_y)
    rmse = math.sqrt(mse)
    absolute_error = mean_absolute_error(solutions_test_data, pred_y)
    max_er = max_error(solutions_test_data, pred_y)

    return score, [mse, rmse, absolute_error, max_er]


import modelsaver
import Estimators.preprocessing_data as prep
from sklearn.metrics import mean_squared_error

model = modelsaver.get_model("RF_randomsearch_3")
X_test, y_test = prep.DataManager("BS").get_test_data()
del X_test["strike_price_percent"]

# print(model.cv_results_)

pred = model.predict(X_test)

print(mean_squared_error(y_test, pred))
