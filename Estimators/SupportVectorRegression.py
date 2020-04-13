from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_squared_error
from Estimators import preprocessing_data as prep
from scipy.stats import uniform
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import modelsaver

########################################################
# ---------------- PARAMETERS --------------------------#
########################################################
# linearSVR -> larger datasets (notes on scikit-learn), with scaling it is faster
kernels = ["rbf", "poly", "sigmoid"]

distributions = dict(C=uniform(loc=1, scale=50),
                     kernel=["rbf", "linear", "poly", "sigmoid"],
                     degree=[2, 3, 4],
                     gamma=["scale", "auto"],
                     epsilon=uniform(0.01, 5))


# param_grid = [{'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf'], 'tol':[0.01]},
#               {'C': [1, 10, 100, 1000], 'degree':[2, 3, 4, 5], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['ploy'], 'tol':[0.01]},
#               {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['sigmoid'], 'tol':[0.01]}]


########################################################################################################################

def cv_svr_models(model, column_fitting, random_state):
    datamanager = prep.DataManager(model=model, column_fitting=column_fitting)
    X, y = datamanager.get_training_data()

    # het SVR gaat veel sneller en presteert veel beter als de data wordt herschaald
    # het werd ook aangeraden!
    scaler = preprocessing.StandardScaler().fit(X, y)
    X = scaler.transform(X)

    svr = SVR(cache_size=500)
    clf = RandomizedSearchCV(svr, distributions, random_state=random_state, cv=3, n_iter=50, verbose=10, n_jobs=7,
                             scoring=['neg_mean_squared_error', 'r2'],
                             refit=False)

    performance = clf.fit(X, y)

    modelsaver.save_model(performance, f"SVR-random_search_{model}_{column_fitting}_scaled_random{random_state}")


def main_cv():
    models = ["BS", "VG", "H"]
    columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    start_random_state = 257

    for i, model in enumerate(models):
        for j, option in enumerate(columns_fitting):
            print(f"Start cv for {model}-{option}")
            if not model == "BS" and not model == "VG" and not (model == "H" and option == "opt_standard"):
                cv_svr_models(model, option, random_state=start_random_state + 10 * i + j * 2)

    print("End")


if __name__ == '__main__':
    print("Start")
    main_cv()
