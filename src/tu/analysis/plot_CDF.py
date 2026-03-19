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

# out_path = '/data/jinxiaochun/test_LandBench/LandBench/1/LandBench-seed9995-epoch1000-r9/meta-LSTM/focast_time 0/'
out_path = 'D:/Outcome/Soil Moisture/'
y_test = np.load(out_path + 'observations.npy')
y_test = lon_transform(y_test)
mask[-int(mask.shape[0] / 5.4):, :] = 0
min_map = np.min(y_test, axis=0)
max_map = np.max(y_test, axis=0)
mask[min_map == max_map] = 0
# LSTM-------------------------------------------------------------------------
y_pred_LSTM = np.load("D:\\Outcome\\Soil Moisture\\LSTM\\focast_time 0\\_predictions.npy")
KGE_LSTM = np.load("D:\\Outcome\\Soil Moisture\\LSTM\\focast_time 0\\KGE_LSTM.npy")
r2_LSTM = np.load("D:\\Outcome\\Soil Moisture\\LSTM\\focast_time 0\\r2_LSTM.npy")
ubrmse_LSTM = np.load("D:\\Outcome\\Soil Moisture\\LSTM\\focast_time 0\\urmse_LSTM.npy")
bias_LSTM = np.load("D:\\Outcome\\Soil Moisture\\LSTM\\focast_time 0\\bias_LSTM.npy")
r_LSTM = np.load("D:\\Outcome\\Soil Moisture\\LSTM\\focast_time 0\\r_LSTM.npy")
y_pred_LSTM = lon_transform(y_pred_LSTM)
KGE_LSTM = two_dim_lon_transform(KGE_LSTM)
r2_LSTM = two_dim_lon_transform(r2_LSTM)
ubrmse_LSTM = two_dim_lon_transform(ubrmse_LSTM)
bias_LSTM = two_dim_lon_transform(bias_LSTM)
r_LSTM = two_dim_lon_transform(r_LSTM)
# lat_LSTM-------------------------------------------------------------------------
y_pred_lat_LSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTM\\focast_time 0\\_predictions.npy")
KGE_lat_LSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTM\\focast_time 0\\KGE_lat_LSTM.npy")
r2_lat_LSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTM\\focast_time 0\\r2_lat_LSTM.npy")
ubrmse_lat_LSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTM\\focast_time 0\\urmse_lat_LSTM.npy")
bias_lat_LSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTM\\focast_time 0\\bias_lat_LSTM.npy")
r_lat_LSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTM\\focast_time 0\\r_lat_LSTM.npy")
y_pred_lat_LSTM = lon_transform(y_pred_lat_LSTM)
KGE_lat_LSTM = two_dim_lon_transform(KGE_lat_LSTM)
r2_lat_LSTM = two_dim_lon_transform(r2_lat_LSTM)
ubrmse_lat_LSTM = two_dim_lon_transform(ubrmse_lat_LSTM)
bias_lat_LSTM = two_dim_lon_transform(bias_lat_LSTM)
r_lat_LSTM = two_dim_lon_transform(r_lat_LSTM)
# FTLSTM-------------------------------------------------------------------------
y_pred_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\_predictions.npy")
KGE_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\KGE_FTAttentionLSTM.npy")
r2_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\r2_FTAttentionLSTM.npy")
ubrmse_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTAttentionLSTM\\focast_time 0\\urmse_FTAttentionLSTM.npy")
bias_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTAttentionLSTM\\focast_time 0\\bias_FTAttentionLSTM.npy")
r_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTAttentionLSTM\\focast_time 0\\r_FTAttentionLSTM.npy")
y_pred_FTLSTM = lon_transform(y_pred_FTLSTM)
KGE_FTLSTM = two_dim_lon_transform(KGE_FTLSTM)
r2_FTLSTM = two_dim_lon_transform(r2_FTLSTM)
ubrmse_FTLSTM = two_dim_lon_transform(ubrmse_FTLSTM)
bias_FTLSTM = two_dim_lon_transform(bias_FTLSTM)
r_FTLSTM = two_dim_lon_transform(r_FTLSTM)
# lat_FTLSTM-------------------------------------------------------------------------
y_pred_lat_FTAttentionLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTAttentionLSTM\\focast_time 0\\_predictions.npy")
KGE_lat_FTAttentionLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTAttentionLSTM\\focast_time 0\\KGE_lat_FTAttentionLSTM.npy")
r2_lat_FTAttentionLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTAttentionLSTM\\focast_time 0\\r2_lat_FTAttentionLSTM.npy")
ubrmse_lat_FTAttentionLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTAttentionLSTM\\focast_time 0\\urmse_lat_FTAttentionLSTM.npy")
bias_lat_FTAttentionLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTAttentionLSTM\\focast_time 0\\bias_lat_FTAttentionLSTM.npy")
r_lat_FTAttentionLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTAttentionLSTM\\focast_time 0\\r_lat_FTAttentionLSTM.npy")
y_pred_lat_FTAttentionLSTM = lon_transform(y_pred_lat_FTAttentionLSTM)
KGE_lat_FTAttentionLSTM = two_dim_lon_transform(KGE_lat_FTAttentionLSTM)
r2_lat_FTAttentionLSTM = two_dim_lon_transform(r2_lat_FTAttentionLSTM)
ubrmse_lat_FTAttentionLSTM = two_dim_lon_transform(ubrmse_lat_FTAttentionLSTM)
bias_lat_FTAttentionLSTM = two_dim_lon_transform(bias_lat_FTAttentionLSTM)
r_lat_FTAttentionLSTM = two_dim_lon_transform(r_lat_FTAttentionLSTM)

plt.rcParams['font.family'] = 'Times New Roman'

# 先用掩膜 mask==1 筛选出有效的地方，再去除nan和inf

KGE_AttentionLSTM = KGE_AttentionLSTM[mask==1]
r2_AttentionLSTM = r2_AttentionLSTM[mask==1]
ubrmse_AttentionLSTM = ubrmse_AttentionLSTM[mask==1]
bias_AttentionLSTM = bias_AttentionLSTM[mask==1]
r_AttentionLSTM = r_AttentionLSTM[mask==1]

KGE_lat_AttentionLSTM = KGE_lat_AttentionLSTM[mask==1]
r2_lat_AttentionLSTM = r2_lat_AttentionLSTM[mask==1]
ubrmse_lat_AttentionLSTM = ubrmse_lat_AttentionLSTM[mask==1]
bias_lat_AttentionLSTM = bias_lat_AttentionLSTM[mask==1]
r_lat_AttentionLSTM = r_lat_AttentionLSTM[mask==1]

KGE_FTAttentionLSTM = KGE_FTAttentionLSTM[mask==1]
r2_FTAttentionLSTM = r2_FTAttentionLSTM[mask==1]
ubrmse_FTAttentionLSTM = ubrmse_FTAttentionLSTM[mask==1]
bias_FTAttentionLSTM = bias_FTAttentionLSTM[mask==1]
r_FTAttentionLSTM = r_FTAttentionLSTM[mask==1]

KGE_lat_FTAttentionLSTM = KGE_lat_FTAttentionLSTM[mask==1]
r2_lat_FTAttentionLSTM = r2_lat_FTAttentionLSTM[mask==1]
ubrmse_lat_FTAttentionLSTM = ubrmse_lat_FTAttentionLSTM[mask==1]
bias_lat_FTAttentionLSTM = bias_lat_FTAttentionLSTM[mask==1]
r_lat_FTAttentionLSTM = r_lat_FTAttentionLSTM[mask==1]

KGE_AttentionLSTM = KGE_AttentionLSTM[~np.isnan(KGE_AttentionLSTM)]
KGE_AttentionLSTM = KGE_AttentionLSTM[~np.isinf(KGE_AttentionLSTM)]
r2_AttentionLSTM = r2_AttentionLSTM[~np.isnan(r2_AttentionLSTM)]
r2_AttentionLSTM = r2_AttentionLSTM[~np.isinf(r2_AttentionLSTM)]
ubrmse_AttentionLSTM = ubrmse_AttentionLSTM[~np.isnan(ubrmse_AttentionLSTM)]
ubrmse_AttentionLSTM = ubrmse_AttentionLSTM[~np.isinf(ubrmse_AttentionLSTM)]
bias_AttentionLSTM = bias_AttentionLSTM[~np.isnan(bias_AttentionLSTM)]
bias_AttentionLSTM = bias_AttentionLSTM[~np.isinf(bias_AttentionLSTM)]
r_AttentionLSTM = r_AttentionLSTM[~np.isnan(r_AttentionLSTM)]
r_AttentionLSTM = r_AttentionLSTM[~np.isinf(r_AttentionLSTM)]

KGE_lat_AttentionLSTM = KGE_lat_AttentionLSTM[~np.isnan(KGE_lat_AttentionLSTM)]
KGE_lat_AttentionLSTM = KGE_lat_AttentionLSTM[~np.isinf(KGE_lat_AttentionLSTM)]
r2_lat_AttentionLSTM = r2_lat_AttentionLSTM[~np.isnan(r2_lat_AttentionLSTM)]
r2_lat_AttentionLSTM = r2_lat_AttentionLSTM[~np.isinf(r2_lat_AttentionLSTM)]
ubrmse_lat_AttentionLSTM = ubrmse_lat_AttentionLSTM[~np.isnan(ubrmse_lat_AttentionLSTM)]
ubrmse_lat_AttentionLSTM = ubrmse_lat_AttentionLSTM[~np.isinf(ubrmse_lat_AttentionLSTM)]
bias_lat_AttentionLSTM = bias_lat_AttentionLSTM[~np.isnan(bias_lat_AttentionLSTM)]
bias_lat_AttentionLSTM = bias_lat_AttentionLSTM[~np.isinf(bias_lat_AttentionLSTM)]
r_lat_AttentionLSTM = r_lat_AttentionLSTM[~np.isnan(r_lat_AttentionLSTM)]
r_lat_AttentionLSTM = r_lat_AttentionLSTM[~np.isinf(r_lat_AttentionLSTM)]

KGE_FTAttentionLSTM = KGE_FTAttentionLSTM[~np.isnan(KGE_FTAttentionLSTM)]
KGE_FTAttentionLSTM = KGE_FTAttentionLSTM[~np.isinf(KGE_FTAttentionLSTM)]
r2_FTAttentionLSTM = r2_FTAttentionLSTM[~np.isnan(r2_FTAttentionLSTM)]
r2_FTAttentionLSTM = r2_FTAttentionLSTM[~np.isinf(r2_FTAttentionLSTM)]
ubrmse_FTAttentionLSTM = ubrmse_FTAttentionLSTM[~np.isnan(ubrmse_FTAttentionLSTM)]
ubrmse_FTAttentionLSTM = ubrmse_FTAttentionLSTM[~np.isinf(ubrmse_FTAttentionLSTM)]
bias_FTAttentionLSTM = bias_FTAttentionLSTM[~np.isnan(bias_FTAttentionLSTM)]
bias_FTAttentionLSTM = bias_FTAttentionLSTM[~np.isinf(bias_FTAttentionLSTM)]
r_FTAttentionLSTM = r_FTAttentionLSTM[~np.isnan(r_FTAttentionLSTM)]
r_FTAttentionLSTM = r_FTAttentionLSTM[~np.isinf(r_FTAttentionLSTM)]

KGE_lat_FTAttentionLSTM = KGE_lat_FTAttentionLSTM[~np.isnan(KGE_lat_FTAttentionLSTM)]
KGE_lat_FTAttentionLSTM = KGE_lat_FTAttentionLSTM[~np.isinf(KGE_lat_FTAttentionLSTM)]
r2_lat_FTAttentionLSTM = r2_lat_FTAttentionLSTM[~np.isnan(r2_lat_FTAttentionLSTM)]
r2_lat_FTAttentionLSTM = r2_lat_FTAttentionLSTM[~np.isinf(r2_lat_FTAttentionLSTM)]
ubrmse_lat_FTAttentionLSTM = ubrmse_lat_FTAttentionLSTM[~np.isnan(ubrmse_lat_FTAttentionLSTM)]
ubrmse_lat_FTAttentionLSTM = ubrmse_lat_FTAttentionLSTM[~np.isinf(ubrmse_lat_FTAttentionLSTM)]
bias_lat_FTAttentionLSTM = bias_lat_FTAttentionLSTM[~np.isnan(bias_FTAttentionLSTM)]
bias_lat_FTAttentionLSTM = bias_lat_FTAttentionLSTM[~np.isinf(bias_FTAttentionLSTM)]
r_lat_FTAttentionLSTM = r_lat_FTAttentionLSTM[~np.isnan(r_lat_FTAttentionLSTM)]
r_lat_FTAttentionLSTM = r_lat_FTAttentionLSTM[~np.isinf(r_lat_FTAttentionLSTM)]

# ------------------------------------------------------------
# R²
# ------------------------------------------------------------
data_sorted1 = np.sort(r2_AttentionLSTM.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(r2_lat_AttentionLSTM.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(r2_FTAttentionLSTM.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(r2_lat_FTAttentionLSTM.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='AttentionLSTM')
plt.plot(data_sorted2, cdf2, label='lat_AttentionLSTM')
plt.plot(data_sorted3, cdf3, label='FTAttentionLSTM')
plt.plot(data_sorted4, cdf4, label='lat_FTAttentionLSTM')
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
data_sorted1 = np.sort(KGE_AttentionLSTM.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(KGE_lat_AttentionLSTM.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(KGE_FTAttentionLSTM.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(KGE_lat_FTAttentionLSTM.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='AttentionLSTM')
plt.plot(data_sorted2, cdf2, label='lat_AttentionLSTM')
plt.plot(data_sorted3, cdf3, label='FTAttentionLSTM')
plt.plot(data_sorted4, cdf4, label='lat_FTAttentionLSTM')

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
data_sorted1 = np.sort(ubrmse_AttentionLSTM.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(ubrmse_lat_AttentionLSTM.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(ubrmse_FTAttentionLSTM.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(ubrmse_lat_FTAttentionLSTM.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='AttentionLSTM')
plt.plot(data_sorted2, cdf2, label='lat_AttentionLSTM')
plt.plot(data_sorted3, cdf3, label='FTAttentionLSTM')
plt.plot(data_sorted4, cdf4, label='lat_FTAttentionLSTM')
plt.xlabel('ubRMSE')
plt.ylabel('CDF')
# plt.title('CDFs of Data')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlim(0, 0.04)
plt.grid(True)
plt.show()
# from matplotlib.patches import Rectangle  # 确保导入
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 一行两列
#
# # 主图
# ax1.plot(data_sorted1, cdf1, label='AttentionLSTM')
# ax1.plot(data_sorted2, cdf2, label='lat_AttentionLSTM')
# ax1.plot(data_sorted3, cdf3, label='FTAttentionLSTM')
# ax1.plot(data_sorted4, cdf4, label='lat_FTAttentionLSTM')
# ax1.set_xlabel('ubRMSE')
# ax1.set_ylabel('CDF')
# ax1.set_xlim(0, 0.04)
# ax1.set_ylim(0, 1)
# ax1.legend(loc='lower right')
# ax1.grid(True)
#
# # 添加虚框标出放大区间
# rect = Rectangle((0.019, 0.75), 0.0022, 0.20,
#                  linewidth=1.5, edgecolor='red', linestyle='--', facecolor='none')
# ax1.add_patch(rect)
#
# # 放大图
# ax2.plot(data_sorted1, cdf1)
# ax2.plot(data_sorted2, cdf2)
# ax2.plot(data_sorted3, cdf3)
# ax2.plot(data_sorted4, cdf4)
# ax2.set_xlim(0.019, 0.0212)
# ax2.set_ylim(0.62, 0.82)
# ax2.set_xlabel('ubRMSE')
# ax2.set_ylabel('CDF')
# ax2.grid(True)
#
# plt.tight_layout()
# plt.show()
# ------------------------------------------------------------
# Bias
# ------------------------------------------------------------
data_sorted1 = np.sort(bias_AttentionLSTM.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(bias_lat_AttentionLSTM.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(bias_FTAttentionLSTM.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(bias_lat_FTAttentionLSTM.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='AttentionLSTM')
plt.plot(data_sorted2, cdf2, label='lat_AttentionLSTM')
plt.plot(data_sorted3, cdf3, label='FTAttentionLSTM')
plt.plot(data_sorted4, cdf4, label='lat_FTAttentionLSTM')
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
data_sorted1 = np.sort(r_AttentionLSTM.flatten())
cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)
data_sorted2 = np.sort(r_lat_AttentionLSTM.flatten())
cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)
data_sorted3 = np.sort(r_FTAttentionLSTM.flatten())
cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)
data_sorted4 = np.sort(r_lat_FTAttentionLSTM.flatten())
cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

# Plot CDFs
plt.plot(data_sorted1, cdf1, label='AttentionLSTM')
plt.plot(data_sorted2, cdf2, label='lat_AttentionLSTM')
plt.plot(data_sorted3, cdf3, label='FTAttentionLSTM')
plt.plot(data_sorted4, cdf4, label='lat_FTAttentionLSTM')
plt.xlabel('R')
plt.ylabel('CDF')
# plt.title('CDFs of Data')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlim(0, 1.1)
plt.grid(True)
plt.show()
