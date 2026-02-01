import numpy as np
import pandas as pd

# TRAINING DATA

# Importing data
fd1_train = pd.read_csv("train_FD001.txt", sep = r"\s+", header = None) # train
fd1_test = pd.read_csv("test_FD001.txt", sep = r"\s+", header = None) # test

cols = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Assigning columns names

fd1_train.columns = cols
fd1_test.columns = cols 

fd1_train["RUL1"] = fd1_train.groupby("unit_id")["cycle"].transform("max") - fd1_train["cycle"] # Calculating RUL (target) - train

# Id, cycle and RUL are ints

fd1_train[["unit_id", "cycle","RUL1"]] = fd1_train[["unit_id","cycle","RUL1"]].astype("int64") 
fd1_test[["unit_id", "cycle"]] = fd1_test[["unit_id","cycle"]].astype("int64")

# All the other parameters are floats
fd1_train.iloc[:, 2:26] = fd1_train.iloc[:, 2:26].astype("float64") 
fd1_test.iloc[:, 2:26] = fd1_test.iloc[:, 2:26].astype("float64")

# This verifies that the dataset doesn't have any NaNs
print(f"We found {fd1_train.isna().sum().sum()} missing values for train")
print(f"We found {fd1_test.isna().sum().sum()} missing values for test") 

print(fd1_train)
print(fd1_test)

fd1_train.to_csv("training_df_01.csv")
fd1_test.to_csv("testing_df_01.csv")