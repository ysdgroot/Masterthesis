import Estimators.preprocessing_data as prep
import matplotlib.pyplot as plt
import pandas as pd


# todo: schrijven van commentaar
# het doel van deze file is het algemeen beschrijven van de gegeneeerde data
def difference_exact_vs_simulation():
    def get_n_smallest_elements(dataframe_column, amount):
        list_data = dataframe_column.tolist()
        list_data.sort()
        return list_data[0:amount]

    def get_n_greatest_elements(dataframe_column, amount):
        list_data = dataframe_column.tolist()
        list_data.sort(reverse=True)
        return list_data[0:amount]

    datamanager = prep.DataManager("BS")

    file_name_v1 = datamanager.get_correct_file(first_version=False, test=False, model="BS")

    df_v1 = pd.read_csv(file_name_v1, comment='#', header=0)

    diff_v1 = df_v1["opt_exact_standard"] - df_v1["opt_standard"]
    min_diff_v1 = get_n_smallest_elements(diff_v1, 1000)
    for index, diff in enumerate(min_diff_v1):
        print(f"{index}: {diff}")

    file_name_v1_vg = datamanager.get_correct_file(first_version=True, test=False, model="VG")
    df_v1_vg = pd.read_csv(file_name_v1_vg, comment='#', header=0)

    max_prices = get_n_greatest_elements(df_v1_vg["opt_standard"], 50)
    print("Maximum pricing")
    for index, diff in enumerate(max_prices):
        print(f"{index}: {diff}")

    # full_dataframe_train = datamanager.get_full_dataframe(test_data=False)
    # full_dataframe_test = datamanager.get_full_dataframe(test_data=True)
    #
    # difference_train = full_dataframe_train["opt_exact_standard"] - full_dataframe_train["opt_standard"]
    # difference_test = full_dataframe_test["opt_exact_standard"] - full_dataframe_test["opt_standard"]
    #
    # min_diff_train = get_n_smallest_elements(difference_train, 30)
    # print(min_diff_train)
    #
    # # plt.hist(difference_train)
    # # plt.show()
    #
    # min_diff_test = get_n_smallest_elements(difference_test, 20)
    # print(min_diff_test)

    # plt.hist(difference_test)
    # plt.show()


if __name__ == '__main__':
    difference_exact_vs_simulation()
