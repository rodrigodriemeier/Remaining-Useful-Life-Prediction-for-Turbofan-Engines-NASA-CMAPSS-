import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

tr1 = pd.read_csv("fd001_train_features.csv")
te1 = pd.read_csv("fd001_test_features.csv")

sep_motors = tr1["unit_id"].unique()
val_units = sep_motors[:20]
train_units = sep_motors[20:]

train = tr1[tr1["unit_id"].isin(train_units)]
val = tr1[tr1["unit_id"].isin(val_units)]

X_train = train.drop(columns=["unit_id", "cycle", "RUL1"])
y_train = train["RUL1"]

X_val = val.drop(columns=["unit_id", "cycle", "RUL1"])
y_val = val["RUL1"]

model = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=25,
    max_features=0.5,
    min_samples_leaf=5,
    random_state=0,
    n_jobs=-1
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
print(mean_absolute_error(y_val, val_pred))

rul_test = np.loadtxt("RUL_FD001.txt")

te_last = te1.loc[te1.groupby("unit_id")["cycle"].idxmax()].sort_values("unit_id")
X_test = te_last.drop(columns=["unit_id", "cycle"])

test_pred = model.predict(X_test)
print(mean_absolute_error(rul_test, test_pred))
