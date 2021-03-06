import timeit
import numpy as np
import math


def stockpaths_bs_naive(maturity=10,
                        steps_per_maturity=100,
                        interest_rate=0.01,
                        volatility=0.1,
                        start_price=100,
                        n_paths=10000):
    # the amount of timesteps needed for the stockprice
    number_of_steps = maturity * steps_per_maturity
    # length for each step
    dt = 1 / steps_per_maturity

    # make list for all the paths
    full_path_list = []

    for j in range(n_paths):
        # each path starts with the start_price
        one_path_list = [start_price]
        # this value will change continuously for the changes
        previous_price = start_price
        # generate a list with all the random values
        random_values = np.random.normal(0, np.sqrt(dt), number_of_steps)
        for index, i in enumerate(range(number_of_steps)):
            # the change for the stock price
            next_price = previous_price * math.exp((interest_rate - 0.5 * volatility ** 2) * dt +
                                                   volatility * random_values[index])
            one_path_list.append(next_price)
            previous_price = next_price
        # append the list prices of 1 path to list with all the paths
        full_path_list.append(one_path_list)

    return full_path_list


def stockpaths_bs_vector(maturity=10,
                         steps_per_maturity=100,
                         interest_rate=0.01,
                         volatility=0.1,
                         start_price=100,
                         n_paths=10000):
    # the amount of timesteps needed for the stockprice
    number_of_steps = maturity * steps_per_maturity
    # length for each step
    dt = 1 / steps_per_maturity

    # calculates the ln(S(t)) process in Euler method.
    # ln(S_t) = ln(S_{t-1}) + (r - 0.5 sigma^2) dt + sigma * N(0,dt)
    # Because the next (ln) value depends on the sum of the previous ln(S), so we take the 'cumsum'.
    # Add the end, we take the exponential and multiply this with S_0
    ln_stock_process = np.cumsum((interest_rate - 0.5 * volatility ** 2) * dt +
                                 volatility * np.random.normal(0, np.sqrt(dt),
                                                               (n_paths, number_of_steps)), 1)

    # adding 0 column, to get 1 when we take the exponential (so the fist valus is the start_price)
    ln_stock_process = np.append(np.zeros((n_paths, 1)), ln_stock_process, axis=1)

    # because we simulated the ln(S_t) we need top take the exponential and multiply this value with the start_price
    stock_process = start_price * np.exp(ln_stock_process)

    return stock_process


# ----------------------------------------------------------------------------------------------------------------------
def variance_process_func(n_paths: int,
                          maturity: int = 1,
                          steps_per_maturity: int = 100,
                          skewness=-0.1,
                          kurtosis=0.7,
                          volatility=0.2):
    number_of_steps = maturity * steps_per_maturity
    size_increments = 1 / steps_per_maturity

    # declaration of values for the gamma process. (Just for readability reason)
    mu_plus = (np.sqrt(skewness ** 2 + 2 * volatility ** 2 / kurtosis) + skewness) / 2
    mu_min = (np.sqrt(skewness ** 2 + 2 * volatility ** 2 / kurtosis) - skewness) / 2

    # Variance Gamma process is the difference of 2 Gamma processes
    gamma_process_plus = np.random.gamma(size_increments / kurtosis, kurtosis * mu_plus,
                                         (n_paths, number_of_steps))
    gamma_process_min = np.random.gamma(size_increments / kurtosis, kurtosis * mu_min,
                                        (n_paths, number_of_steps))

    return np.cumsum(gamma_process_plus - gamma_process_min, axis=1)


def variance_process_brownian_motion(amount_paths: int,
                                     maturity: int = 1,
                                     steps_per_maturity: int = 100,
                                     skewness=-0.1,
                                     kurtosis=0.7,
                                     volatility=0.2):
    number_of_steps = maturity * steps_per_maturity
    size_increments = 1 / steps_per_maturity

    # Variance Gamma process which is based on a Brownian motion
    gamma_process = np.random.gamma(size_increments / kurtosis, kurtosis, (amount_paths, number_of_steps))
    brownian_motion = np.random.randn(amount_paths, number_of_steps)

    return np.cumsum(skewness * gamma_process + volatility * np.sqrt(gamma_process) * brownian_motion,
                     axis=1)


def stockpaths_vg_diff_gamma(maturity=10,
                             steps_per_maturity=100,
                             interest_rate=0.01,
                             skewness=-0.1,
                             kurtosis=0.7,
                             volatility=0.2,
                             start_price=100,
                             amount_paths=10000):
    number_of_evaluations = steps_per_maturity * maturity
    dt = 1 / steps_per_maturity

    # omega based on the article
    omega = np.log(1 - skewness * kurtosis - kurtosis * volatility ** 2 / 2) / kurtosis

    # the process based on the variance gamma stockmodel, each increment or decrement for each time_step
    variance_process = variance_process_func(amount_paths,
                                             steps_per_maturity=steps_per_maturity,
                                             maturity=maturity)

    # Start with the 0 on position 0 (so S_t=0 = S0)
    constant_rate_stock = np.cumsum(np.append(0, np.full(number_of_evaluations - 1,
                                                         (interest_rate + omega) * dt)))

    # The stock price on time t, based on the variance gamma
    total_exponent = np.add(variance_process, constant_rate_stock)

    # Adding 0 in the first column, so the first column (first value of the paths) will be the start price
    first_column = np.zeros((amount_paths, 1))
    total_exponent = np.append(first_column, total_exponent, axis=1)

    return start_price * np.exp(total_exponent)


def stockpaths_vg_brownianmotion(maturity=10,
                                 steps_per_maturity=100,
                                 interest_rate=0.01,
                                 skewness=-0.1,
                                 kurtosis=0.7,
                                 volatility=0.2,
                                 start_price=100,
                                 amount_paths=10000):
    number_of_evaluations = steps_per_maturity * maturity
    dt = 1 / steps_per_maturity

    # omega based on the article
    omega = np.log(1 - skewness * kurtosis - kurtosis * volatility ** 2 / 2) / kurtosis

    variance_process = variance_process_brownian_motion(amount_paths,
                                                        steps_per_maturity=steps_per_maturity,
                                                        maturity=maturity)

    # Start with the 0 on position 0 (so S_t=0 = S0)
    constant_rate_stock = np.cumsum(np.append(0, np.full(number_of_evaluations - 1,
                                                         (interest_rate + omega) * dt)))

    # The stock price on time t, based on the variance gamma
    total_exponent = np.add(variance_process, constant_rate_stock)

    # Adding 0 in the first column, so the first column (first value of the paths) will be the start price
    first_column = np.zeros((amount_paths, 1))
    total_exponent = np.append(first_column, total_exponent, axis=1)

    return start_price * np.exp(total_exponent)


# ----------------------------------------------------------------------------------------------------------------------

def get_brownian_motions_with_correlation(n_paths: int,
                                          correlation: float,
                                          maturity: int,
                                          steps_per_maturity: int = 100):

    dt = 1 / steps_per_maturity
    number_of_steps = maturity * steps_per_maturity

    # brownian_motions = np.random.randn(2, number_of_steps)
    brownian_motions = np.random.randn(n_paths, 2, number_of_steps)

    # declare the matrix to get the correlation
    matrix_pro = np.array([correlation, np.sqrt(1 - correlation ** 2)])

    # do dot-product between matrixes to get the correlated process of the first element of the 'brownian_motions'
    correlated_process = np.array(list(map(lambda x: matrix_pro.dot(x), brownian_motions)))

    # put everything in the correct form of matrix and the correct values(jumps) of the process.
    total_process_paths = np.array([[brownian_motions[i, 0], correlated_process[i]] for i in range(n_paths)]) \
                          * np.sqrt(dt)

    return total_process_paths



def stockpaths_h_naive(maturity=10,
                       steps_per_maturity=100,
                       interest_rate=0.01,
                       correlation=-0.5,
                       long_variance=0.3,
                       start_volatility=0.2,
                       rate_revert_to_long=2,
                       volatility_of_volatility=0.6,
                       start_price=100,
                       n_paths=10000):
    dt = 1 / steps_per_maturity
    number_of_steps = maturity * steps_per_maturity

    all_stock_prices = []
    all_volatilities = []

    all_processes = get_brownian_motions_with_correlation(n_paths,
                                                          correlation,
                                                          maturity,
                                                          steps_per_maturity=steps_per_maturity)

    brownianmotion_stock_processes = all_processes[:, 0, :]
    brownianmotion_volatility_processes = all_processes[:, 1, :]

    for j in range(n_paths):
        stock_prices = [start_price]
        volatilities = [start_volatility]
        bm_stock = brownianmotion_stock_processes[j]
        bm_volatility = brownianmotion_volatility_processes[j]

        for i in range(number_of_steps):
            last_price = stock_prices[-1]
            last_vol = volatilities[-1]
            not_negative_vol = max(last_vol, 0)

            S = last_price * math.exp((interest_rate - 0.5 * not_negative_vol) * dt +
                                      math.sqrt(not_negative_vol) * bm_stock[i])
            dnu = rate_revert_to_long * (long_variance - not_negative_vol) * dt + \
                  volatility_of_volatility * \
                  np.sqrt(not_negative_vol) * bm_volatility[i]

            # adding the next stock prices and volatilities
            stock_prices.append(S)
            volatilities.append(volatilities[-1] + dnu)
        all_stock_prices.append(stock_prices)
        all_volatilities.append(volatilities)

    return np.array(all_stock_prices), np.array(all_volatilities)


def stockpaths_h_partvector(maturity=10,
                            steps_per_maturity=100,
                            interest_rate=0.01,
                            correlation=-0.5,
                            long_variance=0.3,
                            start_volatility=0.2,
                            rate_revert_to_long=2,
                            volatility_of_volatility=0.6,
                            start_price=100,
                            n_paths=10000):
    dt = 1 / steps_per_maturity
    number_of_steps = maturity * steps_per_maturity

    all_volatilities = []

    all_processes = get_brownian_motions_with_correlation(n_paths,
                                                          correlation,
                                                          maturity,
                                                          steps_per_maturity=steps_per_maturity)

    brownianmotion_stock_processes = all_processes[:, 0, :]
    brownianmotion_volatility_processes = all_processes[:, 1, :]

    # start of process of the volatility
    for j in range(n_paths):
        volatilities = [start_volatility]
        bm_volatility = brownianmotion_volatility_processes[j]

        # we don't need the last element of the volatilities (Euler method)
        for i in range(number_of_steps - 1):
            last_vol = volatilities[-1]
            not_negative_vol = max(last_vol, 0)

            dnu = rate_revert_to_long * (long_variance - not_negative_vol) * dt + \
                  volatility_of_volatility * math.sqrt(not_negative_vol) * bm_volatility[i]
            volatilities.append(last_vol + dnu)
        all_volatilities.append(volatilities)
    # change it to a numpy array.
    all_volatilities = np.array(all_volatilities)

    # Because of the Euler scheme, it is possible to get negative values for Volatitlity.
    # This cause problems because the square root is taken for the volatility.
    all_volatilities[all_volatilities < 0] = 0

    drift = (interest_rate - 0.5 * all_volatilities) * dt
    vol = np.sqrt(all_volatilities) * brownianmotion_stock_processes

    # the ln(S_t) = ln(S_{t-1}) + (r - 0.5 * sigma)dt + sqrt(sigma)* N(0, dt)
    # this is the reason we take the 'cumsum'
    total_process_ln = np.cumsum(drift + vol, axis=1)

    # adding 0, to get 1 when we take the exponential
    total_process_ln = np.append(np.zeros((n_paths, 1)), total_process_ln, axis=1)

    total_process = np.exp(total_process_ln) * start_price

    return total_process, all_volatilities


def stockpaths_h_fastest(maturity=10,
                         steps_per_maturity=100,
                         interest_rate=0.01,
                         correlation=-0.5,
                         long_variance=0.3,
                         start_volatility=0.2,
                         rate_revert_to_long=2,
                         volatility_of_volatility=0.6,
                         start_price=100,
                         n_paths=10000):
    dt = 1 / steps_per_maturity
    number_of_steps = maturity * steps_per_maturity

    # get the full process of the correlated Brownian motions
    all_processes = get_brownian_motions_with_correlation(n_paths,
                                                          correlation,
                                                          maturity,
                                                          steps_per_maturity=steps_per_maturity)
    # set names for the different processes.
    brownianmotion_stock_processes = all_processes[:, 0, :]
    brownianmotion_volatility_processes = all_processes[:, 1, :]

    # make matrix for all the volatilities
    all_volatilities = np.zeros((n_paths, number_of_steps))
    # first position set to start volatility
    all_volatilities[:, 0] = start_volatility

    # function for each iteration per timestep
    def func_vol(position, step_size, volatilities, weiner_volatility):
        # get last volatility's of each path
        last_vol = volatilities[:, position]
        # set negative values to 0
        not_negative_vol = last_vol.copy()
        not_negative_vol[not_negative_vol < 0] = 0

        dnu = rate_revert_to_long * (long_variance - not_negative_vol) * step_size + \
              volatility_of_volatility * np.sqrt(not_negative_vol) * weiner_volatility[:, position]

        return dnu

    # lambda expression to get array of the volatility's for each path
    array_f = lambda pos, step_size, vola, process: np.asarray(func_vol(pos, step_size, vola, process))

    # Calculate simultaneously (all paths) the next step.
    for i in range(number_of_steps - 1):
        all_volatilities[:, i + 1] = all_volatilities[:, i] + array_f(i, dt, all_volatilities,
                                                                      brownianmotion_volatility_processes)

    # Because of the Euler scheme, it is possible to get negative values for Volatility.
    # This cause problems because the square root is taken for the volatility.
    all_volatilities[all_volatilities < 0] = 0

    # start of the Stock stockmodel process
    drift = (interest_rate - 0.5 * all_volatilities) * dt
    vol = np.sqrt(all_volatilities) * brownianmotion_stock_processes

    # the ln(S_t) = ln(S_{t-1}) + (r - 0.5 * sigma)dt + sqrt(sigma)* N(0, dt)
    # this is the reason we take the 'cumsum'
    total_process_ln = np.cumsum(drift + vol, axis=1)

    # adding 0, to get 1 when we take the exponential the first element will be the start_price
    total_process_ln = np.append(np.zeros((n_paths, 1)), total_process_ln, axis=1)

    total_process = np.exp(total_process_ln) * start_price

    return total_process

# ----------------------------------------------------------------------------------------------------------------------


speed_bs_vector = timeit.timeit(stockpaths_bs_vector, number=100)
print(f"Speed BS vector: {speed_bs_vector}")
speed_bs_naive = timeit.timeit(stockpaths_bs_naive, number=100)
print(f"Speed BS naive: {speed_bs_naive}")

speed_vg_diff = timeit.timeit(stockpaths_vg_diff_gamma, number=100)
print(f"Speed VG difference: {speed_vg_diff}")
speed_vg_brownian = timeit.timeit(stockpaths_vg_brownianmotion, number=100)
print(f"Speed VG Brownian: {speed_vg_brownian}")

speed_h_naive = timeit.timeit(stockpaths_h_naive, number=100)
print(f"Speed H naive: {speed_h_naive}")
speed_h_partvector = timeit.timeit(stockpaths_h_partvector, number=100)
print(f"Speed H partvector: {speed_h_partvector}")
speed_h_fastest = timeit.timeit(stockpaths_h_fastest, number=100)
print(f"Speed H fastest: {speed_h_fastest}")

