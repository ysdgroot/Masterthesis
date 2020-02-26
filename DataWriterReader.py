import csv

# todo dit bestand wordt niet gebruikt
def write_to_file(name_file, data, model_stock="BS"):
    if model_stock == 'BS':
        print("BlackScholes model")
    with open(name_file, 'w', newline='') as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerow(["interest_rate", "volatility", "maturity", "stock_price", "strike_price", "option_plain",
                              "option_EuroAsian", "option_LB_min", "option_LB_max", "option_LB_min_float",
                              "option_LB_max_float"])

    raise NotImplementedError