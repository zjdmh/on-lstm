import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


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

# out_path = '/data/jinxiaochun/test_LandBench/LandBench/1/LandBench-seed9995-epoch1000-r9/meta-LSTMEncoderDecoder/focast_time 0/'
out_path = 'D:/Outcome/Soil Moisture/'
y_test = np.load(out_path + 'observations.npy')
y_test = lon_transform(y_test)
mask[-int(mask.shape[0] / 5.4):, :] = 0
min_map = np.min(y_test, axis=0)
max_map = np.max(y_test, axis=0)
mask[min_map == max_map] = 0
# LSTMEncoderDecoder-------------------------------------------------------------------------
y_pred_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\KGE_LSTMEncoderDecoder.npy")
r2_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\r2_LSTMEncoderDecoder.npy")
ubrmse_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\urmse_LSTMEncoderDecoder.npy")
bias_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\bias_LSTMEncoderDecoder.npy")
r_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\LSTMEncoderDecoder\\focast_time 0\\r_LSTMEncoderDecoder.npy")
y_pred_LSTMEncoderDecoder = lon_transform(y_pred_LSTMEncoderDecoder)
KGE_LSTMEncoderDecoder = two_dim_lon_transform(KGE_LSTMEncoderDecoder)
r2_LSTMEncoderDecoder = two_dim_lon_transform(r2_LSTMEncoderDecoder)
ubrmse_LSTMEncoderDecoder = two_dim_lon_transform(ubrmse_LSTMEncoderDecoder)
bias_LSTMEncoderDecoder = two_dim_lon_transform(bias_LSTMEncoderDecoder)
r_LSTMEncoderDecoder = two_dim_lon_transform(r_LSTMEncoderDecoder)
# lat_LSTMEncoderDecoder-------------------------------------------------------------------------
y_pred_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\KGE_lat_LSTMEncoderDecoder.npy")
r2_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\r2_lat_LSTMEncoderDecoder.npy")
ubrmse_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\urmse_lat_LSTMEncoderDecoder.npy")
bias_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\bias_lat_LSTMEncoderDecoder.npy")
r_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTMEncoderDecoder\\focast_time 0\\r_lat_LSTMEncoderDecoder.npy")
y_pred_lat_LSTMEncoderDecoder = lon_transform(y_pred_lat_LSTMEncoderDecoder)
KGE_lat_LSTMEncoderDecoder = two_dim_lon_transform(KGE_lat_LSTMEncoderDecoder)
r2_lat_LSTMEncoderDecoder = two_dim_lon_transform(r2_lat_LSTMEncoderDecoder)
ubrmse_lat_LSTMEncoderDecoder = two_dim_lon_transform(ubrmse_lat_LSTMEncoderDecoder)
bias_lat_LSTMEncoderDecoder = two_dim_lon_transform(bias_lat_LSTMEncoderDecoder)
r_lat_LSTMEncoderDecoder = two_dim_lon_transform(r_lat_LSTMEncoderDecoder)
# FTLSTMEncoderDecoder-------------------------------------------------------------------------
y_pred_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\KGE_FTLSTMEncoderDecoder.npy")
r2_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\r2_FTLSTMEncoderDecoder.npy")
ubrmse_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\urmse_FTLSTMEncoderDecoder.npy")
bias_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\bias_FTLSTMEncoderDecoder.npy")
r_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\FTLSTMEncoderDecoder\\focast_time 0\\r_FTLSTMEncoderDecoder.npy")
y_pred_FTLSTMEncoderDecoder = lon_transform(y_pred_FTLSTMEncoderDecoder)
KGE_FTLSTMEncoderDecoder = two_dim_lon_transform(KGE_FTLSTMEncoderDecoder)
r2_FTLSTMEncoderDecoder = two_dim_lon_transform(r2_FTLSTMEncoderDecoder)
ubrmse_FTLSTMEncoderDecoder = two_dim_lon_transform(ubrmse_FTLSTMEncoderDecoder)
bias_FTLSTMEncoderDecoder = two_dim_lon_transform(bias_FTLSTMEncoderDecoder)
r_FTLSTMEncoderDecoder = two_dim_lon_transform(r_FTLSTMEncoderDecoder)
# lat_FTLSTMEncoderDecoder-------------------------------------------------------------------------
y_pred_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\KGE_lat_FTLSTMEncoderDecoder.npy")
r2_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\r2_lat_FTLSTMEncoderDecoder.npy")
ubrmse_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\urmse_lat_FTLSTMEncoderDecoder.npy")
bias_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\bias_lat_FTLSTMEncoderDecoder.npy")
r_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTMEncoderDecoder\\focast_time 0\\r_lat_FTLSTMEncoderDecoder.npy")
y_pred_lat_FTLSTMEncoderDecoder = lon_transform(y_pred_lat_FTLSTMEncoderDecoder)
KGE_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(KGE_lat_FTLSTMEncoderDecoder)
r2_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(r2_lat_FTLSTMEncoderDecoder)
ubrmse_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(ubrmse_lat_FTLSTMEncoderDecoder)
bias_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(bias_lat_FTLSTMEncoderDecoder)
r_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(r_lat_FTLSTMEncoderDecoder)

plt.rcParams['font.family'] = 'Times New Roman'

# 先用掩膜 mask==1 筛选出有效的地方，再去除nan和inf

KGE_LSTMEncoderDecoder = KGE_LSTMEncoderDecoder[mask==1]
r2_LSTMEncoderDecoder = r2_LSTMEncoderDecoder[mask==1]
ubrmse_LSTMEncoderDecoder = ubrmse_LSTMEncoderDecoder[mask==1]
bias_LSTMEncoderDecoder = bias_LSTMEncoderDecoder[mask==1]
r_LSTMEncoderDecoder = r_LSTMEncoderDecoder[mask==1]

KGE_lat_LSTMEncoderDecoder = KGE_lat_LSTMEncoderDecoder[mask==1]
r2_lat_LSTMEncoderDecoder = r2_lat_LSTMEncoderDecoder[mask==1]
ubrmse_lat_LSTMEncoderDecoder = ubrmse_lat_LSTMEncoderDecoder[mask==1]
bias_lat_LSTMEncoderDecoder = bias_lat_LSTMEncoderDecoder[mask==1]
r_lat_LSTMEncoderDecoder = r_lat_LSTMEncoderDecoder[mask==1]

KGE_FTLSTMEncoderDecoder = KGE_FTLSTMEncoderDecoder[mask==1]
r2_FTLSTMEncoderDecoder = r2_FTLSTMEncoderDecoder[mask==1]
ubrmse_FTLSTMEncoderDecoder = ubrmse_FTLSTMEncoderDecoder[mask==1]
bias_FTLSTMEncoderDecoder = bias_FTLSTMEncoderDecoder[mask==1]
r_FTLSTMEncoderDecoder = r_FTLSTMEncoderDecoder[mask==1]

KGE_lat_FTLSTMEncoderDecoder = KGE_lat_FTLSTMEncoderDecoder[mask==1]
r2_lat_FTLSTMEncoderDecoder = r2_lat_FTLSTMEncoderDecoder[mask==1]
ubrmse_lat_FTLSTMEncoderDecoder = ubrmse_lat_FTLSTMEncoderDecoder[mask==1]
bias_lat_FTLSTMEncoderDecoder = bias_lat_FTLSTMEncoderDecoder[mask==1]
r_lat_FTLSTMEncoderDecoder = r_lat_FTLSTMEncoderDecoder[mask==1]

KGE_LSTMEncoderDecoder = KGE_LSTMEncoderDecoder[~np.isnan(KGE_LSTMEncoderDecoder)]
KGE_LSTMEncoderDecoder = KGE_LSTMEncoderDecoder[~np.isinf(KGE_LSTMEncoderDecoder)]
r2_LSTMEncoderDecoder = r2_LSTMEncoderDecoder[~np.isnan(r2_LSTMEncoderDecoder)]
r2_LSTMEncoderDecoder = r2_LSTMEncoderDecoder[~np.isinf(r2_LSTMEncoderDecoder)]
ubrmse_LSTMEncoderDecoder = ubrmse_LSTMEncoderDecoder[~np.isnan(ubrmse_LSTMEncoderDecoder)]
ubrmse_LSTMEncoderDecoder = ubrmse_LSTMEncoderDecoder[~np.isinf(ubrmse_LSTMEncoderDecoder)]
bias_LSTMEncoderDecoder = bias_LSTMEncoderDecoder[~np.isnan(bias_LSTMEncoderDecoder)]
bias_LSTMEncoderDecoder = bias_LSTMEncoderDecoder[~np.isinf(bias_LSTMEncoderDecoder)]
r_LSTMEncoderDecoder = r_LSTMEncoderDecoder[~np.isnan(r_LSTMEncoderDecoder)]
r_LSTMEncoderDecoder = r_LSTMEncoderDecoder[~np.isinf(r_LSTMEncoderDecoder)]

KGE_lat_LSTMEncoderDecoder = KGE_lat_LSTMEncoderDecoder[~np.isnan(KGE_lat_LSTMEncoderDecoder)]
KGE_lat_LSTMEncoderDecoder = KGE_lat_LSTMEncoderDecoder[~np.isinf(KGE_lat_LSTMEncoderDecoder)]
r2_lat_LSTMEncoderDecoder = r2_lat_LSTMEncoderDecoder[~np.isnan(r2_lat_LSTMEncoderDecoder)]
r2_lat_LSTMEncoderDecoder = r2_lat_LSTMEncoderDecoder[~np.isinf(r2_lat_LSTMEncoderDecoder)]
ubrmse_lat_LSTMEncoderDecoder = ubrmse_lat_LSTMEncoderDecoder[~np.isnan(ubrmse_lat_LSTMEncoderDecoder)]
ubrmse_lat_LSTMEncoderDecoder = ubrmse_lat_LSTMEncoderDecoder[~np.isinf(ubrmse_lat_LSTMEncoderDecoder)]
bias_lat_LSTMEncoderDecoder = bias_lat_LSTMEncoderDecoder[~np.isnan(bias_lat_LSTMEncoderDecoder)]
bias_lat_LSTMEncoderDecoder = bias_lat_LSTMEncoderDecoder[~np.isinf(bias_lat_LSTMEncoderDecoder)]
r_lat_LSTMEncoderDecoder = r_lat_LSTMEncoderDecoder[~np.isnan(r_lat_LSTMEncoderDecoder)]
r_lat_LSTMEncoderDecoder = r_lat_LSTMEncoderDecoder[~np.isinf(r_lat_LSTMEncoderDecoder)]

KGE_FTLSTMEncoderDecoder = KGE_FTLSTMEncoderDecoder[~np.isnan(KGE_FTLSTMEncoderDecoder)]
KGE_FTLSTMEncoderDecoder = KGE_FTLSTMEncoderDecoder[~np.isinf(KGE_FTLSTMEncoderDecoder)]
r2_FTLSTMEncoderDecoder = r2_FTLSTMEncoderDecoder[~np.isnan(r2_FTLSTMEncoderDecoder)]
r2_FTLSTMEncoderDecoder = r2_FTLSTMEncoderDecoder[~np.isinf(r2_FTLSTMEncoderDecoder)]
ubrmse_FTLSTMEncoderDecoder = ubrmse_FTLSTMEncoderDecoder[~np.isnan(ubrmse_FTLSTMEncoderDecoder)]
ubrmse_FTLSTMEncoderDecoder = ubrmse_FTLSTMEncoderDecoder[~np.isinf(ubrmse_FTLSTMEncoderDecoder)]
bias_FTLSTMEncoderDecoder = bias_FTLSTMEncoderDecoder[~np.isnan(bias_FTLSTMEncoderDecoder)]
bias_FTLSTMEncoderDecoder = bias_FTLSTMEncoderDecoder[~np.isinf(bias_FTLSTMEncoderDecoder)]
r_FTLSTMEncoderDecoder = r_FTLSTMEncoderDecoder[~np.isnan(r_FTLSTMEncoderDecoder)]
r_FTLSTMEncoderDecoder = r_FTLSTMEncoderDecoder[~np.isinf(r_FTLSTMEncoderDecoder)]

KGE_lat_FTLSTMEncoderDecoder = KGE_lat_FTLSTMEncoderDecoder[~np.isnan(KGE_lat_FTLSTMEncoderDecoder)]
KGE_lat_FTLSTMEncoderDecoder = KGE_lat_FTLSTMEncoderDecoder[~np.isinf(KGE_lat_FTLSTMEncoderDecoder)]
r2_lat_FTLSTMEncoderDecoder = r2_lat_FTLSTMEncoderDecoder[~np.isnan(r2_lat_FTLSTMEncoderDecoder)]
r2_lat_FTLSTMEncoderDecoder = r2_lat_FTLSTMEncoderDecoder[~np.isinf(r2_lat_FTLSTMEncoderDecoder)]
ubrmse_lat_FTLSTMEncoderDecoder = ubrmse_lat_FTLSTMEncoderDecoder[~np.isnan(ubrmse_lat_FTLSTMEncoderDecoder)]
ubrmse_lat_FTLSTMEncoderDecoder = ubrmse_lat_FTLSTMEncoderDecoder[~np.isinf(ubrmse_lat_FTLSTMEncoderDecoder)]
bias_lat_FTLSTMEncoderDecoder = bias_lat_FTLSTMEncoderDecoder[~np.isnan(bias_FTLSTMEncoderDecoder)]
bias_lat_FTLSTMEncoderDecoder = bias_lat_FTLSTMEncoderDecoder[~np.isinf(bias_FTLSTMEncoderDecoder)]
r_lat_FTLSTMEncoderDecoder = r_lat_FTLSTMEncoderDecoder[~np.isnan(r_lat_FTLSTMEncoderDecoder)]
r_lat_FTLSTMEncoderDecoder = r_lat_FTLSTMEncoderDecoder[~np.isinf(r_lat_FTLSTMEncoderDecoder)]

# ------------------------------------------------------------
# R²
# ------------------------------------------------------------
data_sorted1 = np.sort(r2_LSTMEncoderDecoder.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(r2_lat_LSTMEncoderDecoder.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(r2_FTLSTMEncoderDecoder.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(r2_lat_FTLSTMEncoderDecoder.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='LSTMEncoderDecoder')
plt.plot(data_sorted2, cdf2, label='lat_LSTMEncoderDecoder')
plt.plot(data_sorted3, cdf3, label='FTLSTMEncoderDecoder')
plt.plot(data_sorted4, cdf4, label='lat_FTLSTMEncoderDecoder')
plt.xlabel('R²')
plt.ylabel('CDF')
# plt.title('CDFs of Data')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlim(0, 1.1)
plt.grid(True)
plt.show()
# ------------------------------------------------------------
# KGE
# ------------------------------------------------------------
data_sorted1 = np.sort(KGE_LSTMEncoderDecoder.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(KGE_lat_LSTMEncoderDecoder.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(KGE_FTLSTMEncoderDecoder.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(KGE_lat_FTLSTMEncoderDecoder.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='LSTMEncoderDecoder')
plt.plot(data_sorted2, cdf2, label='lat_LSTMEncoderDecoder')
plt.plot(data_sorted3, cdf3, label='FTLSTMEncoderDecoder')
plt.plot(data_sorted4, cdf4, label='lat_FTLSTMEncoderDecoder')

plt.xlabel('KGE')
plt.ylabel('CDF')
# plt.title('CDFs of Data')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlim(0, 1.1)
plt.grid(True)
plt.show()
# ------------------------------------------------------------
# ubRMSE
# ------------------------------------------------------------
data_sorted1 = np.sort(ubrmse_LSTMEncoderDecoder.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(ubrmse_lat_LSTMEncoderDecoder.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(ubrmse_FTLSTMEncoderDecoder.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(ubrmse_lat_FTLSTMEncoderDecoder.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)
#
# # Plot CDFs
# plt.plot(data_sorted1, cdf1, label='LSTMEncoderDecoder')
# plt.plot(data_sorted2, cdf2, label='lat_LSTMEncoderDecoder')
# plt.plot(data_sorted3, cdf3, label='FTLSTMEncoderDecoder')
# plt.plot(data_sorted4, cdf4, label='lat_FTLSTMEncoderDecoder')
# plt.xlabel('ubRMSE')
# plt.ylabel('CDF')
# # plt.title('CDFs of Data')
# plt.legend(loc='best')
# plt.ylim(0, 1)
# plt.xlim(0, 0.04)
# plt.grid(True)
# plt.show()
from matplotlib.patches import Rectangle  # 确保导入
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 一行两列

# 主图
ax1.plot(data_sorted1, cdf1, label='LSTMEncoderDecoder')
ax1.plot(data_sorted2, cdf2, label='lat_LSTMEncoderDecoder')
ax1.plot(data_sorted3, cdf3, label='FTLSTMEncoderDecoder')
ax1.plot(data_sorted4, cdf4, label='lat_FTLSTMEncoderDecoder')
ax1.set_xlabel('ubRMSE')
ax1.set_ylabel('CDF')
ax1.set_xlim(0, 0.04)
ax1.set_ylim(0, 1)
ax1.legend(loc='lower right')
ax1.grid(True)

# 添加虚框标出放大区间
rect = Rectangle((0.0208, 0.87), 0.0012, 0.09,
                 linewidth=1.5, edgecolor='red', linestyle='--', facecolor='none')
ax1.add_patch(rect)

# 放大图
ax2.plot(data_sorted1, cdf1)
ax2.plot(data_sorted2, cdf2)
ax2.plot(data_sorted3, cdf3)
ax2.plot(data_sorted4, cdf4)
ax2.set_xlim(0.0208, 0.022)
ax2.set_ylim(0.87, 0.96)
ax2.set_xlabel('ubRMSE')
ax2.set_ylabel('CDF')
ax2.grid(True)

plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# Bias
# ------------------------------------------------------------
data_sorted1 = np.sort(bias_LSTMEncoderDecoder.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(bias_lat_LSTMEncoderDecoder.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(bias_FTLSTMEncoderDecoder.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(bias_lat_FTLSTMEncoderDecoder.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='LSTMEncoderDecoder')
plt.plot(data_sorted2, cdf2, label='lat_LSTMEncoderDecoder')
plt.plot(data_sorted3, cdf3, label='FTLSTMEncoderDecoder')
plt.plot(data_sorted4, cdf4, label='lat_FTLSTMEncoderDecoder')
plt.xlabel('Bias')
plt.ylabel('CDF')
# plt.title('CDFs of Data')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlim(0, 0.06)
plt.grid(True)
plt.show()
# ------------------------------------------------------------
# R
# ------------------------------------------------------------
data_sorted1 = np.sort(r_LSTMEncoderDecoder.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(r_lat_LSTMEncoderDecoder.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(r_FTLSTMEncoderDecoder.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(r_lat_FTLSTMEncoderDecoder.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='LSTMEncoderDecoder')
plt.plot(data_sorted2, cdf2, label='lat_LSTMEncoderDecoder')
plt.plot(data_sorted3, cdf3, label='FTLSTMEncoderDecoder')
plt.plot(data_sorted4, cdf4, label='lat_FTLSTMEncoderDecoder')
plt.xlabel('R')
plt.ylabel('CDF')
# plt.title('CDFs of Data')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlim(0, 1.1)
plt.grid(True)
plt.show()
