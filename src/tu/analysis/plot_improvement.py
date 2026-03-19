import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as mtick

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
# GRU-------------------------------------------------------------------------
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
# lat_GRU-------------------------------------------------------------------------
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
# FTGRU-------------------------------------------------------------------------
y_pred_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\_predictions.npy")
KGE_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\KGE_FTLSTM.npy")
r2_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\r2_FTLSTM.npy")
ubrmse_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\urmse_FTLSTM.npy")
bias_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\bias_FTLSTM.npy")
r_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\r_FTLSTM.npy")
y_pred_FTLSTM = lon_transform(y_pred_FTLSTM)
KGE_FTLSTM = two_dim_lon_transform(KGE_FTLSTM)
r2_FTLSTM = two_dim_lon_transform(r2_FTLSTM)
ubrmse_FTLSTM = two_dim_lon_transform(ubrmse_FTLSTM)
bias_FTLSTM = two_dim_lon_transform(bias_FTLSTM)
r_FTLSTM = two_dim_lon_transform(r_FTLSTM)
# lat_FTGRU-------------------------------------------------------------------------
y_pred_lat_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTM\\focast_time 0\\_predictions.npy")
KGE_lat_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTM\\focast_time 0\\KGE_lat_FTLSTM.npy")
r2_lat_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTM\\focast_time 0\\r2_lat_FTLSTM.npy")
ubrmse_lat_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTM\\focast_time 0\\urmse_lat_FTLSTM.npy")
bias_lat_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTM\\focast_time 0\\bias_lat_FTLSTM.npy")
r_lat_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTM\\focast_time 0\\r_lat_FTLSTM.npy")
y_pred_lat_FTLSTM = lon_transform(y_pred_lat_FTLSTM)
KGE_lat_FTLSTM = two_dim_lon_transform(KGE_lat_FTLSTM)
r2_lat_FTLSTM = two_dim_lon_transform(r2_lat_FTLSTM)
ubrmse_lat_FTLSTM = two_dim_lon_transform(ubrmse_lat_FTLSTM)
bias_lat_FTLSTM = two_dim_lon_transform(bias_lat_FTLSTM)
r_lat_FTLSTM = two_dim_lon_transform(r_lat_FTLSTM)

#-------------------------------------------------------------------------------------

lat_ = np.load("D:\\Outcome\\Soil Moisture\\lat_1.npy")
lon_ = np.load("D:\\Outcome\\Soil Moisture\\lon_1.npy")
lon_ = np.linspace(-180,179,int(y_pred_LSTM.shape[2]))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']=13

# ---------------------------------
# 1: Improvement of lat_LSTM Model with KGE
# ---------------------------------
KGE_LSTM_improvement = KGE_lat_LSTM-KGE_LSTM
plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)
cpool = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027',
        '#a50026']
cmap = colors.ListedColormap(cpool)
cs = m.contourf(xi, yi, KGE_LSTM_improvement, np.arange(-1, 1.05, 0.05), cmap=cmap)

cbar = m.colorbar(cs, location='bottom', pad="10%",ticks=np.arange(-1,1.05,0.2))
# cbar.set_label('KGE_improvement')
cbar.ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.title('Improvement of lat_LSTM Model with KGE')
#plt.savefig(out_path + 'urmse_'+ cfg['modelname'] + '_spatial distributions.png')

plt.show()

# ---------------------------------
# 2: Improvement of FTGRU Model with KGE
# ---------------------------------
KGE_LSTM_improvement1 = KGE_FTLSTM-KGE_LSTM
plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)
cpool = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027',
        '#a50026']
cmap = colors.ListedColormap(cpool)
cs = m.contourf(xi, yi, KGE_LSTM_improvement1, np.arange(-1, 1.05, 0.05), cmap=cmap)

cbar = m.colorbar(cs, location='bottom', pad="10%",ticks=np.arange(-1,1.05,0.2))
# cbar.set_label('KGE_improvement')
cbar.ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.title('Improvement of FTLSTM Model with KGE')
#plt.savefig(out_path + 'urmse_'+ cfg['modelname'] + '_spatial distributions.png')

plt.show()
# ---------------------------------
# 3: Improvement of lat_FTGRUModel with KGE
# ---------------------------------
KGE_LSTM_improvement2 = KGE_lat_FTLSTM-KGE_LSTM
plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)
cpool = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027',
        '#a50026']
cmap = colors.ListedColormap(cpool)
cs = m.contourf(xi, yi, KGE_LSTM_improvement2, np.arange(-1, 1.05, 0.05), cmap=cmap)

cbar = m.colorbar(cs, location='bottom', pad="10%",ticks=np.arange(-1,1.05,0.2))
# cbar.set_label('KGE_improvement')
cbar.ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.title('Improvement of lat_FTLSTM Model with KGE')
#plt.savefig(out_path + 'urmse_'+ cfg['modelname'] + '_spatial distributions.png')

plt.show()