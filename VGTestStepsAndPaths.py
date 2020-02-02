from ModelsStock.BlackScholes import BlackScholes
from OptionModels.PlainVanilla import PlainVanilla
import time
import csv
import math
import numpy as np

# Testing paths
time_steps_per_maturities = [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(15000, 20001, 1000)]
add_header = False
write_comment_info = False

# todo instellen van andere parameters
file_name = 'Test-steps and accuracy-VG-v1.csv'
maturity = 10
interest_rate = 0.001
volatitlity = 0.1
start_price = 100
strike_price = 100
