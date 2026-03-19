import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
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


mask = np.load("D:\\Outcome\\Soil Moisture1\\Mask with 1 spatial resolution.npy")
mask = two_dim_lon_transform(mask)

# out_path = '/data/jinxiaochun/test_LandBench/LandBench/1/LandBench-seed9995-epoch1000-r9/meta-LSTMEncoderDecoder/focast_time 0/'
out_path = 'D:/Outcome/Soil Moisture1/'
y_test = np.load(out_path + 'observations.npy')
y_test = lon_transform(y_test)
mask[-int(mask.shape[0] / 5.4):, :] = 0
min_map = np.min(y_test, axis=0)
max_map = np.max(y_test, axis=0)
mask[min_map == max_map] = 0
# LSTMEncoderDecoderEncoderDecoder-------------------------------------------------------------------------
y_pred_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\LSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\LSTMEncoderDecoder\\focast_time 0\\KGE_LSTMEncoderDecoder.npy")
r2_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\LSTMEncoderDecoder\\focast_time 0\\r2_LSTMEncoderDecoder.npy")
ubrmse_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\LSTMEncoderDecoder\\focast_time 0\\urmse_LSTMEncoderDecoder.npy")
bias_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\LSTMEncoderDecoder\\focast_time 0\\bias_LSTMEncoderDecoder.npy")
r_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\LSTMEncoderDecoder\\focast_time 0\\r_LSTMEncoderDecoder.npy")
y_pred_LSTMEncoderDecoder = lon_transform(y_pred_LSTMEncoderDecoder)
KGE_LSTMEncoderDecoder = two_dim_lon_transform(KGE_LSTMEncoderDecoder)
r2_LSTMEncoderDecoder = two_dim_lon_transform(r2_LSTMEncoderDecoder)
ubrmse_LSTMEncoderDecoder = two_dim_lon_transform(ubrmse_LSTMEncoderDecoder)
bias_LSTMEncoderDecoder = two_dim_lon_transform(bias_LSTMEncoderDecoder)
r_LSTMEncoderDecoder = two_dim_lon_transform(r_LSTMEncoderDecoder)
# lat_LSTMEncoderDecoderEncoderDecoder-------------------------------------------------------------------------
y_pred_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_LSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_LSTMEncoderDecoder\\focast_time 0\\KGE_lat_LSTMEncoderDecoder.npy")
r2_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_LSTMEncoderDecoder\\focast_time 0\\r2_lat_LSTMEncoderDecoder.npy")
ubrmse_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_LSTMEncoderDecoder\\focast_time 0\\urmse_lat_LSTMEncoderDecoder.npy")
bias_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_LSTMEncoderDecoder\\focast_time 0\\bias_lat_LSTMEncoderDecoder.npy")
r_lat_LSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_LSTMEncoderDecoder\\focast_time 0\\r_lat_LSTMEncoderDecoder.npy")
y_pred_lat_LSTMEncoderDecoder = lon_transform(y_pred_lat_LSTMEncoderDecoder)
KGE_lat_LSTMEncoderDecoder = two_dim_lon_transform(KGE_lat_LSTMEncoderDecoder)
r2_lat_LSTMEncoderDecoder = two_dim_lon_transform(r2_lat_LSTMEncoderDecoder)
ubrmse_lat_LSTMEncoderDecoder = two_dim_lon_transform(ubrmse_lat_LSTMEncoderDecoder)
bias_lat_LSTMEncoderDecoder = two_dim_lon_transform(bias_lat_LSTMEncoderDecoder)
r_lat_LSTMEncoderDecoder = two_dim_lon_transform(r_lat_LSTMEncoderDecoder)
# FTLSTMEncoderDecoderEncoderDecoder-------------------------------------------------------------------------
y_pred_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\FTLSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\FTLSTMEncoderDecoder\\focast_time 0\\KGE_FTLSTMEncoderDecoder.npy")
r2_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\FTLSTMEncoderDecoder\\focast_time 0\\r2_FTLSTMEncoderDecoder.npy")
ubrmse_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\FTLSTMEncoderDecoder\\focast_time 0\\urmse_FTLSTMEncoderDecoder.npy")
bias_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\FTLSTMEncoderDecoder\\focast_time 0\\bias_FTLSTMEncoderDecoder.npy")
r_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\FTLSTMEncoderDecoder\\focast_time 0\\r_FTLSTMEncoderDecoder.npy")
y_pred_FTLSTMEncoderDecoder = lon_transform(y_pred_FTLSTMEncoderDecoder)
KGE_FTLSTMEncoderDecoder = two_dim_lon_transform(KGE_FTLSTMEncoderDecoder)
r2_FTLSTMEncoderDecoder = two_dim_lon_transform(r2_FTLSTMEncoderDecoder)
ubrmse_FTLSTMEncoderDecoder = two_dim_lon_transform(ubrmse_FTLSTMEncoderDecoder)
bias_FTLSTMEncoderDecoder = two_dim_lon_transform(bias_FTLSTMEncoderDecoder)
r_FTLSTMEncoderDecoder = two_dim_lon_transform(r_FTLSTMEncoderDecoder)
# lat_FTLSTMEncoderDecoderEncoderDecoder-------------------------------------------------------------------------
y_pred_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_FTLSTMEncoderDecoder\\focast_time 0\\_predictions.npy")
KGE_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_FTLSTMEncoderDecoder\\focast_time 0\\KGE_lat_FTLSTMEncoderDecoder.npy")
r2_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_FTLSTMEncoderDecoder\\focast_time 0\\r2_lat_FTLSTMEncoderDecoder.npy")
ubrmse_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_FTLSTMEncoderDecoder\\focast_time 0\\urmse_lat_FTLSTMEncoderDecoder.npy")
bias_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_FTLSTMEncoderDecoder\\focast_time 0\\bias_lat_FTLSTMEncoderDecoder.npy")
r_lat_FTLSTMEncoderDecoder = np.load("D:\\Outcome\\Soil Moisture1\\lat_FTLSTMEncoderDecoder\\focast_time 0\\r_lat_FTLSTMEncoderDecoder.npy")
y_pred_lat_FTLSTMEncoderDecoder = lon_transform(y_pred_lat_FTLSTMEncoderDecoder)
KGE_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(KGE_lat_FTLSTMEncoderDecoder)
r2_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(r2_lat_FTLSTMEncoderDecoder)
ubrmse_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(ubrmse_lat_FTLSTMEncoderDecoder)
bias_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(bias_lat_FTLSTMEncoderDecoder)
r_lat_FTLSTMEncoderDecoder = two_dim_lon_transform(r_lat_FTLSTMEncoderDecoder)

plt.rcParams['font.family'] = 'Times New Roman'

# ------------------------------------------------------------
# lat_LSTMEncoderDecoder
# ------------------------------------------------------------

observed_sm = np.squeeze(y_test)
observed_sm = observed_sm[-365:, :, :]
observed_sm[:, mask == 0] = -0

predicted_LSTMEncoderDecoder = np.squeeze(y_pred_LSTMEncoderDecoder)
predicted_LSTMEncoderDecoder = predicted_LSTMEncoderDecoder[-365:, :, :]
predicted_LSTMEncoderDecoder[:, mask == 0] = -0

predicted_lat_LSTMEncoderDecoder = np.squeeze(y_pred_lat_LSTMEncoderDecoder)
predicted_lat_LSTMEncoderDecoder = predicted_lat_LSTMEncoderDecoder[-365:, :, :]
predicted_lat_LSTMEncoderDecoder[:, mask == 0] = -0

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes.flatten()

# 计算散点图
sizes = np.pi * 2 ** 1
observed_sm_mean = np.average(observed_sm, axis=0)
predicted_sm_mean = np.average(predicted_LSTMEncoderDecoder, axis=0)
############################################
sm_index = observed_sm_mean != 0
observed_sm_mean = observed_sm_mean[sm_index]
predicted_sm_mean = predicted_sm_mean[sm_index]
m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
regression_line = []
for a in observed_sm_mean:
    regression_line.append((m * a) + b)
radius = 0.1  # 半径

colormap = plt.get_cmap("jet")
Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
ax[0].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
              norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
ax[0].plot(observed_sm_mean, regression_line, 'red', lw=0.8, label='aaa')  #####回归线
ax[0].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
ax[0].set_title('LSTMEncoderDecoder Model')
ax[0].set_xlabel('observed SM (LSTMEncoderDecoder)')
ax[0].set_ylabel('predicted SM (LSTMEncoderDecoder)')
metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
ax[0].text(0.05, 0.7, metrics)
# plt.plot(X, Y_x3, label=u"sin函数")
print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)

# 计算散点图
sizes = np.pi * 2 ** 1
observed_sm_mean = np.average(observed_sm, axis=0)
predicted_sm_mean = np.average(predicted_lat_LSTMEncoderDecoder, axis=0)
############################################
sm_index = observed_sm_mean != 0
observed_sm_mean = observed_sm_mean[sm_index]
predicted_sm_mean = predicted_sm_mean[sm_index]
m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
regression_line = []
for a in observed_sm_mean:
    regression_line.append((m * a) + b)
radius = 0.1  # 半径

colormap = plt.get_cmap("jet")
Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
ax[1].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
              norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
ax[1].plot(observed_sm_mean, regression_line, 'red', lw=0.8)  #####回归线
ax[1].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
ax[1].set_title('lat_LSTMEncoderDecoder Model')
ax[1].set_xlabel('observed SM (lat_LSTMEncoderDecoder)')
ax[1].set_ylabel('predicted SM (lat_LSTMEncoderDecoder)')
metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
ax[1].text(0.05, 0.7, metrics)
print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)

plt.show()
# ------------------------------------------------------------

# ------------------------------------------------------------
# FTLSTMEncoderDecoder
# ------------------------------------------------------------

observed_sm = np.squeeze(y_test)
observed_sm = observed_sm[-365:, :, :]
observed_sm[:, mask == 0] = -0

predicted_LSTMEncoderDecoder = np.squeeze(y_pred_LSTMEncoderDecoder)
predicted_LSTMEncoderDecoder = predicted_LSTMEncoderDecoder[-365:, :, :]
predicted_LSTMEncoderDecoder[:, mask == 0] = -0

predicted_FTLSTMEncoderDecoder = np.squeeze(y_pred_FTLSTMEncoderDecoder)
predicted_FTLSTMEncoderDecoder = predicted_FTLSTMEncoderDecoder[-365:, :, :]
predicted_FTLSTMEncoderDecoder[:, mask == 0] = -0

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes.flatten()

# 计算散点图
sizes = np.pi * 2 ** 1
observed_sm_mean = np.average(observed_sm, axis=0)
predicted_sm_mean = np.average(predicted_LSTMEncoderDecoder, axis=0)
############################################
sm_index = observed_sm_mean != 0
observed_sm_mean = observed_sm_mean[sm_index]
predicted_sm_mean = predicted_sm_mean[sm_index]
m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
regression_line = []
for a in observed_sm_mean:
    regression_line.append((m * a) + b)
radius = 0.1  # 半径

colormap = plt.get_cmap("jet")
Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
ax[0].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
              norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
ax[0].plot(observed_sm_mean, regression_line, 'red', lw=0.8, label='aaa')  #####回归线
ax[0].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
ax[0].set_title('LSTMEncoderDecoder Model')
ax[0].set_xlabel('observed SM (LSTMEncoderDecoder)')
ax[0].set_ylabel('predicted SM (LSTMEncoderDecoder)')
metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
ax[0].text(0.05, 0.7, metrics)
# plt.plot(X, Y_x3, label=u"sin函数")
print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)

# 计算散点图
sizes = np.pi * 2 ** 1
observed_sm_mean = np.average(observed_sm, axis=0)
predicted_sm_mean = np.average(predicted_FTLSTMEncoderDecoder, axis=0)
############################################
sm_index = observed_sm_mean != 0
observed_sm_mean = observed_sm_mean[sm_index]
predicted_sm_mean = predicted_sm_mean[sm_index]
m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
regression_line = []
for a in observed_sm_mean:
    regression_line.append((m * a) + b)
radius = 0.1  # 半径

colormap = plt.get_cmap("jet")
Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
ax[1].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
              norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
ax[1].plot(observed_sm_mean, regression_line, 'red', lw=0.8)  #####回归线
ax[1].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
ax[1].set_title('FTLSTMEncoderDecoder Model')
ax[1].set_xlabel('observed SM (FTLSTMEncoderDecoder)')
ax[1].set_ylabel('predicted SM (FTLSTMEncoderDecoder)')
metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
ax[1].text(0.05, 0.7, metrics)
print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)

plt.show()
# ------------------------------------------------------------

# ------------------------------------------------------------
# lat_FTLSTMEncoderDecoder
# ------------------------------------------------------------

observed_sm = np.squeeze(y_test)
observed_sm = observed_sm[-365:, :, :]
observed_sm[:, mask == 0] = -0

predicted_LSTMEncoderDecoder = np.squeeze(y_pred_LSTMEncoderDecoder)
predicted_LSTMEncoderDecoder = predicted_LSTMEncoderDecoder[-365:, :, :]
predicted_LSTMEncoderDecoder[:, mask == 0] = -0

predicted_lat_FTLSTMEncoderDecoder = np.squeeze(y_pred_lat_FTLSTMEncoderDecoder)
predicted_lat_FTLSTMEncoderDecoder = predicted_lat_FTLSTMEncoderDecoder[-365:, :, :]
predicted_lat_FTLSTMEncoderDecoder[:, mask == 0] = -0

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes.flatten()

# 计算散点图
sizes = np.pi * 2 ** 1
observed_sm_mean = np.average(observed_sm, axis=0)
predicted_sm_mean = np.average(predicted_LSTMEncoderDecoder, axis=0)
############################################
sm_index = observed_sm_mean != 0
observed_sm_mean = observed_sm_mean[sm_index]
predicted_sm_mean = predicted_sm_mean[sm_index]
m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
regression_line = []
for a in observed_sm_mean:
    regression_line.append((m * a) + b)
radius = 0.1  # 半径

colormap = plt.get_cmap("jet")
Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
ax[0].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
              norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
ax[0].plot(observed_sm_mean, regression_line, 'red', lw=0.8, label='aaa')  #####回归线
ax[0].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
ax[0].set_title('LSTMEncoderDecoder Model')
ax[0].set_xlabel('observed SM (LSTMEncoderDecoder)')
ax[0].set_ylabel('predicted SM (LSTMEncoderDecoder)')
metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
ax[0].text(0.05, 0.7, metrics)
# plt.plot(X, Y_x3, label=u"sin函数")
print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)

# 计算散点图
sizes = np.pi * 2 ** 1
observed_sm_mean = np.average(observed_sm, axis=0)
predicted_sm_mean = np.average(predicted_lat_FTLSTMEncoderDecoder, axis=0)
############################################
sm_index = observed_sm_mean != 0
observed_sm_mean = observed_sm_mean[sm_index]
predicted_sm_mean = predicted_sm_mean[sm_index]
m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
regression_line = []
for a in observed_sm_mean:
    regression_line.append((m * a) + b)
radius = 0.1  # 半径

colormap = plt.get_cmap("jet")
Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
ax[1].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
              norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
ax[1].plot(observed_sm_mean, regression_line, 'red', lw=0.8)  #####回归线
ax[1].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
ax[1].set_title('lat_FTLSTMEncoderDecoder Model')
ax[1].set_xlabel('observed SM (lat_FTLSTMEncoderDecoder)')
ax[1].set_ylabel('predicted SM (lat_FTLSTMEncoderDecoder)')
metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
ax[1].text(0.05, 0.7, metrics)
print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)

plt.show()
# ------------------------------------------------------------

