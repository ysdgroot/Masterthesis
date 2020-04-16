import Estimators.preprocessing_data as prep
import pandas as pd


def difference_exact_vs_simulation():
    """
    This function is to check the difference between the simulations of the data and the exact theoretic value.
    In this case it means the difference of the option price based on the Black-Scholes model.
    """

    def get_n_smallest_elements(dataframe_column, amount):
        # returns the list of the 'amount' smallest elements from column in a Dataframe
        list_data = dataframe_column.tolist()
        list_data.sort()
        return list_data[0:amount]

    def get_n_greatest_elements(dataframe_column, amount):
        # returns the list of the 'amount' largest elements from column in a Dataframe
        list_data = dataframe_column.tolist()
        list_data.sort(reverse=True)
        return list_data[0:amount]

    datamanager = prep.DataManager("BS")
    file_name_v2 = datamanager.get_correct_file(first_version=False, test=False, model="BS")
    df_v2 = pd.read_csv(file_name_v2, comment='#', header=0)

    diff_v2 = df_v2["opt_exact_standard"] - df_v2["opt_standard"]
    min_diff_v2 = get_n_smallest_elements(diff_v2, 10)
    max_diff_v2 = get_n_greatest_elements(diff_v2, 10)

    print("Second version; data is sampled with maturity between 1 and 25")
    print("Results = Theoretic - Simulation")

    print("Biggest diffence, where simulation estimates to large")
    for index, diff in enumerate(min_diff_v2):
        print(f"{index}: {diff}")

    print("Biggest diffence, where simulation estimates to small")
    for index, diff in enumerate(max_diff_v2):
        print(f"{index}: {diff}")

    # -----------------------------------------------------------------------------------------------------------------
    file_name_v1 = datamanager.get_correct_file(first_version=True, test=False, model="BS")
    df_v1 = pd.read_csv(file_name_v1, comment='#', header=0)

    diff_v1 = df_v1["opt_exact_standard"] - df_v1["opt_standard"]
    max_diff_v1 = get_n_greatest_elements(diff_v1, 10)
    min_diff_v1 = get_n_smallest_elements(diff_v1, 10)
    print("First version; data is sampled with maturity between 1 and 60")
    print("Results = Theoretic - Simulation")

    print("Biggest diffence, where simulation estimates to large")
    for index, diff in enumerate(min_diff_v1):
        print(f"{index}: {diff}")

    print("Biggest diffence, where simulation estimates to small")
    for index, diff in enumerate(max_diff_v1):
        print(f"{index}: {diff}")

    # -----------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    difference_exact_vs_simulation()
