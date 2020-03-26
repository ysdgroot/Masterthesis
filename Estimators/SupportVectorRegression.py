from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_squared_error
from Estimators import preprocessing_data as prep
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import modelsaver

########################################################
# ---------------- PARAMETERS --------------------------#
########################################################
# SVR, linearSVR, NUSVR
# linearSVR -> larger datasets (notes on scikit-learn)
kernels = ["rbf", "laplacian", "poly", "sigmoid"]

# gamma = ["scale", "auto"]  # kernel coefficient for 'rbf', 'poly' and 'sigmoid'
gamma = [0.01, 0.001, 0.0001]
degree = [2, 3, 4, 5]
dual = False  # n_samples > n_features

distributions = dict(C=uniform(loc=1, scale=50),
                     kernel=["rbf", "linear", "poly", "sigmoid"],
                     degree=[1, 2, 3, 4, 5],
                     gamma=["scale", "auto"],
                     shrinking=[True, False])

# todo: Nystroem gebruiken met een lineaire SVR, anders zal het veel te lang duren om alles te doen.

# param_grid = [{'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf'], 'tol':[0.01]},
#               {'C': [1, 10, 100, 1000], 'degree':[2, 3, 4, 5], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['ploy'], 'tol':[0.01]},
#               {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['sigmoid'], 'tol':[0.01]}]


########################################################################################################################

datamanager = prep.DataManager(column_fitting="opt_exact_standard")
# datamanager = prep.DataManager(column_fitting="opt_asianmean")
X, y = datamanager.get_training_data()
del X["strike_price_percent"]

feature_map = Nystroem(kernel="laplacian",
                       gamma=0.01,
                       n_components=1000,
                       random_state=2)

transformed_data = feature_map.fit_transform(X, y)
lin_svr = LinearSVR(C=100,
                    verbose=1,
                    max_iter=10000)

lin_svr.fit(transformed_data, y)
print(f"Score = {mean_squared_error(y, lin_svr.predict(transformed_data))}")

X_test, y_test = datamanager.get_test_data()
del X_test["strike_price_percent"]

transformed_data_test = feature_map.transform(X_test)

print(f"Score test= {mean_squared_error(y_test, lin_svr.predict(transformed_data_test))}")
# svr = SVR(verbose=1)

# search = svr.fit(X, y)


# search = GridSearchCV(svr, param_grid, cv=5, n_jobs=7)
# search.fit(X, y)

# modelsaver.save_model(search, "SVR_gridsearch")

# print(search.cv_results_)
# print(search.best_params_)
# print(search.best_score_)


# pred = search.predict(X_test)
# mse = mean_squared_error(y_test, pred)
# print(f'MSE test (best) {mse}')

# for kernel in kernels:
#     print(f"Kernel: {kernel}")
#
#     start = time.perf_counter()
#     model_svr = LinearSVR(C=3.0,
#                           dual=False,
#                           cache_size=400,
#                           coef0=0.0,
#                           degree=6,
#                           epsilon=0.1,
#                           gamma='scale',
#                           kernel=kernel,
#                           max_iter=-1,
#                           shrinking=True,
#                           tol=0.0001).fit(X, y)
#     end = time.perf_counter()
#
#     print('Time: ' + str(end - start))
#
#     score_eval_train = model_svr.score(X, y)
#     print(f"Evaluation {score_eval_train}")
#
#     prediction = model_svr.predict(X_test)
#     print(mean_squared_error(prediction, y_test))
