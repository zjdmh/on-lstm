import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap


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


prop_cycle = plt.rcParams['axes.prop_cycle']
default_colors = [*prop_cycle]

# 打印默认颜色
print([color['color'] for color in default_colors])
sea_mask = np.load("D:\\Outcome\\Soil Moisture\\Mask with 1 spatial resolution.npy")
sea_mask = two_dim_lon_transform(sea_mask)

# out_path = '/data/jinxiaochun/test_LandBench/LandBench/1/LandBench-seed9995-epoch1000-r9/meta-LSTM/focast_time 0/'
out_path = 'D:/Outcome/Soil Moisture/'
y_test = np.load(out_path + 'observations.npy')
y_test = lon_transform(y_test)
sea_mask[-int(sea_mask.shape[0] / 5.4):, :] = 0
min_map = np.min(y_test, axis=0)
max_map = np.max(y_test, axis=0)
sea_mask[min_map == max_map] = 0
climates_mask = np.load('./LandBench/'+"climates_mask_" + '1' + ".npy")
mask = sea_mask * climates_mask
# LSTM-------------------------------------------------------------------------
y_pred_LSTM = np.load("D:\\Outcome\\Soil Moisture\\LSTM\\focast_time 0\\_predictions.npy")
y_pred_LSTM = lon_transform(y_pred_LSTM)

# lat_LSTM-------------------------------------------------------------------------
y_pred_lat_LSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_LSTM\\focast_time 0\\_predictions.npy")
y_pred_lat_LSTM = lon_transform(y_pred_lat_LSTM)

# FTLSTM-------------------------------------------------------------------------
y_pred_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\FTLSTM\\focast_time 0\\_predictions.npy")
y_pred_FTLSTM = lon_transform(y_pred_FTLSTM)

# lat_FTLSTM-------------------------------------------------------------------------
y_pred_lat_FTLSTM = np.load("D:\\Outcome\\Soil Moisture\\lat_FTLSTM\\focast_time 0\\_predictions.npy")
y_pred_lat_FTLSTM = lon_transform(y_pred_lat_FTLSTM)


lat_ = np.load("D:\\Outcome\\Soil Moisture\\lat_1.npy")
lon_ = np.load("D:\\Outcome\\Soil Moisture\\lon_1.npy")
lon_ = np.linspace(-180,179,int(y_pred_LSTM.shape[2]))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']=13
# ------------------------------------------------------------
# lstm
# ------------------------------------------------------------


# climate1 lon -70  lat -2 110,91
# climate2 lon -111 lat 37 69,52
# climate2.1 lon 14 lat 23 194,66
# climate3 lon -85  lat 34 95,55

# climate3.1 lon 97  lat 24 277，65

# climate4 lon -108  lat 62 72,27
# climate4.1 lon 27  lat 67 207,22


# climate5.2 lon -160  lat 68 20,21
# climate5.2 lon -64  lat 58 116,31

# sites_lon_index = []
# sites_lat_index = []
sites_lon_index = [43,113,277,347,271]
sites_lat_index = [23,106,52,25,20]

# for i in range(5):
#     climates = i + 1
#     a = np.where(mask == climates)
#     lat = a[0]
#     lon = a[1]
#     high = int(lon.size) + 1
#     index = np.random.randint(0, high, 1)
#     sites_lon_index.append(lon[index])
#     sites_lat_index.append(lat[index])

plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90., 90, 18.)
meridians = np.arange(-180., 180., 36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)
for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
    # ndarray
    lon = lon_[int(lon_index)]
    lat = lat_[int(lat_index)]
    plt.plot(lon, lat, marker='*', color='red', markersize=9)
plt.legend(loc=0)
plt.show()

# lat_LSTM------------------------------------------------------------
# data_all = [y_test,y_pred_LSTM,y_pred_lat_LSTM]  # y_pred_process
# color_list = ['black', 'blue', 'red']  # red
# # name_plt5 = ['ERA5-Land values',cfg['modelname'],'process-based']
# for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
#     count = 0
#     fig, axs = plt.subplots(1, 1, figsize=(15, 2))
#     print('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
# 
#     for data_f5plt in (data_all):
#         axs.plot(data_f5plt[:, lat_index, lon_index], color=color_list[count])  # label=name_plt5[count]
#         axs.legend(loc=1)
#         count = count + 1
# 
#     axs.set_title('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
#     plt.show()
# FTLSTM------------------------------------------------------------
# data_all = [y_test,y_pred_LSTM,y_pred_FTLSTM]  # y_pred_process
# color_list = ['black', 'blue', 'red']  # red
# # name_plt5 = ['ERA5-Land values',cfg['modelname'],'process-based']
# for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
#     count = 0
#     fig, axs = plt.subplots(1, 1, figsize=(15, 2))
#     print('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
#
#     for data_f5plt in (data_all):
#         axs.plot(data_f5plt[:, lat_index, lon_index], color=color_list[count])  # label=name_plt5[count]
#         axs.legend(loc=1)
#         count = count + 1
#
#     axs.set_title('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
#     plt.show()
# lat_FTLSTM------------------------------------------------------------
data_all = [y_test,y_pred_LSTM,y_pred_lat_FTLSTM]  # y_pred_process
color_list = ['black', 'blue', 'red']  # red
# name_plt5 = ['ERA5-Land values',cfg['modelname'],'process-based']
for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
    count = 0
    fig, axs = plt.subplots(1, 1, figsize=(15, 2))
    print('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))

    for data_f5plt in (data_all):
        axs.plot(data_f5plt[:, lat_index, lon_index], color=color_list[count])  # label=name_plt5[count]
        axs.legend(loc=1)
        count = count + 1

    axs.set_title('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
    plt.show()
