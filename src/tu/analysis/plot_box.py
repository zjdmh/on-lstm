import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import pandas as pd
import seaborn as sns

def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new


def two_dim_lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :int(x.shape[1] / 2)] = x[:, int(x.shape[1] / 2):]
    x_new[:, int(x.shape[1] / 2):] = x[:, :int(x.shape[1] / 2)]
    return x_new


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


def density_calc(x, y, radius):
    """
    散点密度计算（以便给散点图中的散点密度进行颜色渲染）
    :param x:
    :param y:
    :param radius:
    :return:  数据密度
    """
    res = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        res[i] = np.sum((x > (x[i] - radius)) & (x < (x[i] + radius))
                        & (y > (y[i] - radius)) & (y < (y[i] + radius)))
    return res


mask = np.load("D:\\Outcome\\Soil Moisture\\Mask with 1 spatial resolution.npy")
mask = two_dim_lon_transform(mask)

# out_path = '/data/jinxiaochun/test_LandBench/LandBench/1/LandBench-seed9995-epoch1000-r9/meta-LSTM/focast_time 0/'
out_path = 'D:/Outcome/Soil Moisture/'
y_test = np.load(out_path + 'observations.npy')
y_test = lon_transform(y_test)
mask[-int(mask.shape[0] / 5.4):, :] = 0
min_map = np.min(y_test, axis=0)
max_map = np.max(y_test, axis=0)
mask[min_map == max_map] = 0
# LSTM------------------------------------------------------------------------
y_pred_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\KGE_LSTMEncoderDecoder.npy")
r2_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\r2_LSTMEncoderDecoder.npy")
ubrmse_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\urmse_LSTMEncoderDecoder.npy")
bias_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\bias_LSTMEncoderDecoder.npy")
y_pred_LSTMEncoderDecoder = lon_transform(y_pred_LSTMEncoderDecoder)
KGE_LSTMEncoderDecoder = two_dim_lon_transform(KGE_LSTMEncoderDecoder)
r2_LSTMEncoderDecoder = two_dim_lon_transform(r2_LSTMEncoderDecoder)
ubrmse_LSTMEncoderDecoder = two_dim_lon_transform(ubrmse_LSTMEncoderDecoder)
bias_LSTMEncoderDecoder = two_dim_lon_transform(bias_LSTMEncoderDecoder)
# lat_LSTM-------------------------------------------------------------------------
y_pred_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\KGE_lat_LSTMEncoderDecoder.npy")
r2_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\r2_lat_LSTMEncoderDecoder.npy")
ubrmse_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\urmse_lat_LSTMEncoderDecoder.npy")
bias_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\bias_lat_LSTMEncoderDecoder.npy")
y_pred_lat_LSTMEncoderDecoder = lon_transform(y_pred_lat_LSTMEncoderDecoder)
KGE_lat_LSTMEncoderDecoder = two_dim_lon_transform(KGE_lat_LSTMEncoderDecoder)
r2_lat_LSTMEncoderDecoder = two_dim_lon_transform(r2_lat_LSTMEncoderDecoder)
ubrmse_lat_LSTMEncoderDecoder = two_dim_lon_transform(ubrmse_lat_LSTMEncoderDecoder)
bias_lat_LSTMEncoderDecoder = two_dim_lon_transform(bias_lat_LSTMEncoderDecoder)
# FTLSTM-------------------------------------------------------------------------
y_pred_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\KGE_FTLSTMEncoderDecoder.npy")
r2_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\r2_FTLSTMEncoderDecoder.npy")
ubrmse_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\urmse_FTLSTMEncoderDecoder.npy")
bias_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\bias_FTLSTMEncoderDecoder.npy")
y_FTLSTMEncoderDecoder = lon_transform(y_pred_FTLSTMEncoderDecoder)
KGE_FTLSTMEncoderDecoder = two_dim_lon_transform(KGE_FTLSTMEncoderDecoder)
r2_FTLSTMEncoderDecoder = two_dim_lon_transform(r2_FTLSTMEncoderDecoder)
ubrmse_FTLSTMEncoderDecoder = two_dim_lon_transform(ubrmse_FTLSTMEncoderDecoder)
bias_FTLSTMEncoderDecoder = two_dim_lon_transform(bias_FTLSTMEncoderDecoder)
# lat_FTLSTM-------------------------------------------------------------------------
y_pred_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\KGE_lat_FTLSTMEncoderDecoder.npy")
r2_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\r2_lat_FTLSTMEncoderDecoder.npy")
ubrmse_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\urmse_lat_FTLSTMEncoderDecoder.npy")
bias_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\bias_lat_FTLSTMEncoderDecoder.npy")
y_pred_lat_FTLSTMEncoderDecoder = lon_transform(y_pred_lat_FTLSTMEncoderDecoder)
KGE_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(KGE_lat_FTLSTMEncoderDecoder)
r2_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(r2_lat_FTLSTMEncoderDecoder)
ubrmse_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(ubrmse_lat_FTLSTMEncoderDecoder)
bias_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(bias_lat_FTLSTMEncoderDecoder)

plt.rcParams['font.family'] = 'Times New Roman'

# LSTM------------------------------------------------------------------------
y_pred_LSTMEncoderDecoder = y_pred_LSTMEncoderDecoder[:,mask==1]
KGE_LSTMEncoderDecoder = KGE_LSTMEncoderDecoder[mask==1]
r2_LSTMEncoderDecoder = r2_LSTMEncoderDecoder[mask==1]
ubrmse_LSTMEncoderDecoder = ubrmse_LSTMEncoderDecoder[mask==1]
bias_LSTMEncoderDecoder = bias_LSTMEncoderDecoder[mask==1]
# lat_LSTM-------------------------------------------------------------------------
y_pred_lat_LSTMEncoderDecoder = y_pred_lat_LSTMEncoderDecoder[:,mask==1]
KGE_lat_LSTMEncoderDecoder = KGE_lat_LSTMEncoderDecoder[mask==1]
r2_lat_LSTMEncoderDecoder = r2_lat_LSTMEncoderDecoder[mask==1]
ubrmse_lat_LSTMEncoderDecoder = ubrmse_lat_LSTMEncoderDecoder[mask==1]
bias_lat_LSTMEncoderDecoder = bias_lat_LSTMEncoderDecoder[mask==1]
# FTLSTM-------------------------------------------------------------------------
y_pred_FTLSTMEncoderDecoder = y_pred_FTLSTMEncoderDecoder[:,mask==1]
KGE_FTLSTMEncoderDecoder = KGE_FTLSTMEncoderDecoder[mask==1]
r2_FTLSTMEncoderDecoder = r2_FTLSTMEncoderDecoder[mask==1]
ubrmse_FTLSTMEncoderDecoder = ubrmse_FTLSTMEncoderDecoder[mask==1]
bias_FTLSTMEncoderDecoder = bias_FTLSTMEncoderDecoder[mask==1]
# lat_FTLSTM-------------------------------------------------------------------------
y_pred_lat_FTLSTMEncoderDecoder = y_pred_lat_FTLSTMEncoderDecoder[:,mask==1]
KGE_lat_FTLSTMEncoderDecoder = KGE_lat_FTLSTMEncoderDecoder[mask==1]
r2_lat_FTLSTMEncoderDecoder = r2_lat_FTLSTMEncoderDecoder[mask==1]
ubrmse_lat_FTLSTMEncoderDecoder = ubrmse_lat_FTLSTMEncoderDecoder[mask==1]
bias_lat_FTLSTMEncoderDecoder = bias_lat_FTLSTMEncoderDecoder[mask==1]


# ------------------------------------------------------------
# bias
# ------------------------------------------------------------
data1 = bias_LSTMEncoderDecoder
data2 = bias_lat_LSTMEncoderDecoder
data3 = bias_FTLSTMEncoderDecoder
data4 = bias_lat_FTLSTMEncoderDecoder
# 将数据放入列表中
df1 = pd.DataFrame({'EDLSTM': data1, 'lat_EDLSTM': data2, 'FTEDLSTM': data3,'lat_FTEDLSTM': data4})# 绘制箱型图
# 绘制箱型图
# 绘制箱型图
sns.boxplot(data=df1,showfliers=False)
plt.ylabel('Bias')
# 显示图形
plt.show()

# ------------------------------------------------------------
# KGE
# ------------------------------------------------------------
data1 = KGE_LSTMEncoderDecoder
data2 = KGE_lat_LSTMEncoderDecoder
data3 = KGE_FTLSTMEncoderDecoder
data4 = KGE_lat_FTLSTMEncoderDecoder
# 将数据放入列表中
df1 = pd.DataFrame({'EDLSTM': data1, 'lat_EDLSTM': data2, 'FTEDLSTM': data3,'lat_FTEDLSTM': data4})# 绘制箱型图
# 绘制箱型图
# 绘制箱型图
sns.boxplot(data=df1,showfliers=False)
plt.ylabel('KGE')
# 显示图形
plt.show()

# ------------------------------------------------------------
# r2
# ------------------------------------------------------------
data1 = r2_LSTMEncoderDecoder
data2 = r2_lat_LSTMEncoderDecoder
data3 = r2_FTLSTMEncoderDecoder
data4 = r2_lat_FTLSTMEncoderDecoder
# 将数据放入列表中
df1 = pd.DataFrame({'EDLSTM': data1, 'lat_EDLSTM': data2, 'FTEDLSTM': data3,'lat_FTEDLSTM': data4})# 绘制箱型图
# 绘制箱型图
sns.boxplot(data=df1,showfliers=False)
plt.ylabel('R²')
# 显示图形
plt.show()

# ------------------------------------------------------------
# ubrmse
# ------------------------------------------------------------
data1 = ubrmse_LSTMEncoderDecoder
data2 = ubrmse_lat_LSTMEncoderDecoder
data3 = ubrmse_FTLSTMEncoderDecoder
data4 = ubrmse_lat_FTLSTMEncoderDecoder
# 将数据放入列表中
df1 = pd.DataFrame({'EDLSTM': data1, 'lat_EDLSTM': data2, 'FTEDLSTM': data3,'lat_FTEDLSTM': data4})# 绘制箱型图
# 绘制箱型图
sns.boxplot(data=df1,showfliers=False)
plt.ylabel('ubRMSE')
# 显示图形
plt.show()
