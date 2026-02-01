import pandas as pd
import numpy as np

w = 10
min_p = 1  

tr1 = pd.read_csv("training_df_01.csv")
te1 = pd.read_csv("testing_df_01.csv")

def rolling_slope(y):
    w_ = len(y)
    t = np.arange(w_, dtype=float)
    t_mean = (w_ - 1) / 2.0
    var_t = np.sum((t - t_mean) ** 2)
    y_mean = np.mean(y)
    cov_ty = np.sum((t - t_mean) * (y - y_mean))
    return cov_ty / var_t if var_t > 0 else 0.0

def add_rolling_features(df, w, min_p):
    sensor_cols = list(df.columns[6:27])
    n = np.arange(1, 22)

    g = df.groupby("unit_id", group_keys=False)

    mean_s = g[sensor_cols].rolling(window=w, min_periods=min_p).mean().reset_index(level=0, drop=True)
    std_s  = g[sensor_cols].rolling(window=w, min_periods=min_p).std(ddof=0).reset_index(level=0, drop=True)

    mean_s.columns = [f"mean_S{i}" for i in n]
    std_s.columns  = [f"std_S{i}" for i in n]

    slope_s = pd.DataFrame(index=df.index)
    i = 1
    for c in sensor_cols:
        slope_s[f"slope_S{i}"] = (
            g[c]
            .rolling(window=w, min_periods=min_p)
            .apply(rolling_slope, raw=True)
            .reset_index(level=0, drop=True)
        )
        i += 1

    df_out = pd.concat([df, mean_s, slope_s, std_s], axis=1)

    if min_p > 1:
        idx_drop = []
        for uid, idxs in df_out.groupby("unit_id").groups.items():
            idx_drop.extend(list(idxs[:min_p-1]))
        df_out = df_out.drop(index=idx_drop)

    return df_out

tr1_feat = add_rolling_features(tr1, w, min_p)
te1_feat = add_rolling_features(te1, w, min_p)

tr1_feat.to_csv("fd001_train_features.csv", index=False)
te1_feat.to_csv("fd001_test_features.csv", index=False)
