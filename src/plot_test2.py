import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from config import get_args
from matplotlib.colors import Normalize
import numpy.ma as ma
from statistics import mean
import matplotlib.colors as colors

plt_f='Fig.6'

def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):]
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)] 
  return x_new

def two_dim_lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:int(x.shape[1]/2)] = x[:,int(x.shape[1]/2):]
  x_new[:,int(x.shape[1]/2):] = x[:,:int(x.shape[1]/2)]
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


# configures
cfg = get_args()
#PATH = '/data/jinxiaochun/test_LandBench/LandBench/1/'
PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
mask = np.load(PATH+file_name_mask)
mask = two_dim_lon_transform(mask)

#out_path = '/data/jinxiaochun/test_LandBench/LandBench/1/LandBench-seed9995-epoch1000-r9/meta-LSTM/focast_time 0/'
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
new_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'

static_climate = np.load(new_path+'static_norm.npy')
static_climate = np.squeeze(static_climate)
static_climate = two_dim_lon_transform(static_climate)


name_pred = cfg['modelname']
# if cfg['modelname'] in ["Process"] and cfg['label'] == ["volumetric_soil_water_20cm"]:
# 	y_pred = (y_pred[1:])/(1000)
# 	y_test = np.load(out_path+'observations.npy')
# elif cfg['modelname'] in ["Process"] and cfg['label'] == ["surface_sensible_heat_flux"]:
# 	y_test = np.load(out_path+'observations.npy')
# 	y_pred = -(y_pred[1:])/(86400*cfg['forcast_time'])
#
# else:
y_test = np.load(out_path+'observations.npy')

y_test = lon_transform(y_test)


mask[-int(mask.shape[0]/5.4):,:]=0
min_map = np.min(y_test,axis=0)
max_map = np.max(y_test,axis=0)
mask[min_map==max_map] = 0

num_models = [9992, 9993, 9995, 9998, 9999]
bias1_sum = bias1_sum1 = bias1_sum2 = bias1_sum3 = bias_sum = bias_sum1 = bias_sum2 = bias_sum3 = r2_sum = r2_sum1 = r2_sum2 = r2_sum3 = rmse_sum = rmse_sum1 = rmse_sum2 = rmse_sum3 = KGE_sum = KGE_sum1 = KGE_sum2 = KGE_sum3 = NSE_sum = NSE_sum1 = NSE_sum2 = NSE_sum3 = y_pred_sum = y_pred_sum1 = y_pred_sum2 = y_pred_sum3 = 0
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'LSTM/focast_time {cfg["forcast_time"]}/' + f'r2_LSTM.npy')
    r2_sum += np.load(file_path)
r2_ = r2_sum / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'r2_{cfg["modelname"]}.npy')
    r2_sum1 += np.load(file_path)
r2_1 = r2_sum1 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'r2_{cfg["modelname"]}.npy')
    r2_sum2 += np.load(file_path)
r2_2 = r2_sum2 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'r2_{cfg["modelname"]}.npy')
    r2_sum3 += np.load(file_path)
r2_3 = r2_sum3 / len(num_models)

for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'LSTM/focast_time {cfg["forcast_time"]}/' + f'bias_LSTM.npy')
    bias_sum += np.load(file_path)
bias_ = bias_sum / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'bias_{cfg["modelname"]}.npy')
    bias_sum1 += np.load(file_path)
bias_1 = bias_sum1 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'bias_{cfg["modelname"]}.npy')
    bias_sum2 += np.load(file_path)
bias_2 = bias_sum2 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'bias_{cfg["modelname"]}.npy')
    bias_sum3 += np.load(file_path)
bias_3 = bias_sum3 / len(num_models)

for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'LSTM/focast_time {cfg["forcast_time"]}/' + f'bias1_LSTM.npy')
    bias1_sum += np.load(file_path)
bias1_ = bias1_sum / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'bias1_{cfg["modelname"]}.npy')
    bias1_sum1 += np.load(file_path)
bias1_1 = bias1_sum1 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'bias1_{cfg["modelname"]}.npy')
    bias1_sum2 += np.load(file_path)
bias1_2 = bias1_sum2 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'bias1_{cfg["modelname"]}.npy')
    bias1_sum3 += np.load(file_path)
bias1_3 = bias1_sum3 / len(num_models)


for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'LSTM/focast_time {cfg["forcast_time"]}/' + f'rmse_LSTM.npy')
    rmse_sum += np.load(file_path)
rmse_ = rmse_sum / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'rmse_{cfg["modelname"]}.npy')
    rmse_sum1 += np.load(file_path)
rmse_1 = rmse_sum1 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'rmse_{cfg["modelname"]}.npy')
    rmse_sum2 += np.load(file_path)
rmse_2 = rmse_sum2 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'rmse_{cfg["modelname"]}.npy')
    rmse_sum3 += np.load(file_path)
rmse_3 = rmse_sum3 / len(num_models)


for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'LSTM/focast_time {cfg["forcast_time"]}/' + f'KGE_LSTM.npy')
    KGE_sum += np.load(file_path)
KGE_ = KGE_sum / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'KGE_{cfg["modelname"]}.npy')
    KGE_sum1 += np.load(file_path)
KGE_1 = KGE_sum1 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'KGE_{cfg["modelname"]}.npy')
    KGE_sum2 += np.load(file_path)
KGE_2 = KGE_sum2 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'KGE_{cfg["modelname"]}.npy')
    KGE_sum3 += np.load(file_path)
KGE_3 = KGE_sum3 / len(num_models)

for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'LSTM/focast_time {cfg["forcast_time"]}/' + f'NSE_LSTM.npy')
    NSE_sum += np.load(file_path)
NSE_ = NSE_sum / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'NSE_{cfg["modelname"]}.npy')
    NSE_sum1 += np.load(file_path)
NSE_1 = NSE_sum1 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'NSE_{cfg["modelname"]}.npy')
    NSE_sum2 += np.load(file_path)
NSE_2 = NSE_sum2 / len(num_models)
for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'NSE_{cfg["modelname"]}.npy')
    NSE_sum3 += np.load(file_path)
NSE_3 = NSE_sum3 / len(num_models)

for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'LSTM/focast_time {cfg["forcast_time"]}/' + f'_predictions.npy')
    y_pred_sum += np.load(file_path)
y_pred = y_pred_sum / len(num_models)

for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'_predictions.npy')
    y_pred_sum1 += np.load(file_path)
y_pred1 = y_pred_sum1 / len(num_models)

for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'_predictions.npy')
    y_pred_sum2 += np.load(file_path)
y_pred2 = y_pred_sum2 / len(num_models)

for model_num in num_models:
    file_path = (new_path + 'LandBench/' + f'{cfg["modelname"]}/focast_time {cfg["forcast_time"]}/' + f'_predictions.npy')
    y_pred_sum3 += np.load(file_path)
y_pred3 = y_pred_sum3 / len(num_models)

# y_pred = np.load(out_path+'_predictions.npy')
y_pred = lon_transform(y_pred)
y_pred1 = lon_transform(y_pred1)
y_pred2 = lon_transform(y_pred2)
y_pred3 = lon_transform(y_pred3)


# r2_  = np.load(out_path+'r2_'+cfg['modelname'] +'.npy')
r2_ = two_dim_lon_transform(r2_)
r2_1 = two_dim_lon_transform(r2_1)
r2_2 = two_dim_lon_transform(r2_2)
r2_3 = two_dim_lon_transform(r2_3)

# rmse_  = np.load(out_path+'rmse_'+cfg['modelname'] +'.npy')
rmse_ = two_dim_lon_transform(rmse_)
rmse_1 = two_dim_lon_transform(rmse_1)
rmse_2 = two_dim_lon_transform(rmse_2)
rmse_3 = two_dim_lon_transform(rmse_3)

bias_ = two_dim_lon_transform(bias_)
bias_1 = two_dim_lon_transform(bias_1)
bias_2 = two_dim_lon_transform(bias_2)
bias_3 = two_dim_lon_transform(bias_3)

bias1_ = two_dim_lon_transform(bias1_)
bias1_1 = two_dim_lon_transform(bias1_1)
bias1_2 = two_dim_lon_transform(bias1_2)
bias1_3 = two_dim_lon_transform(bias1_3)

# KGE_  = np.load(out_path+'KGE_'+cfg['modelname'] +'.npy')
KGE_ = two_dim_lon_transform(KGE_)
KGE_1 = two_dim_lon_transform(KGE_1)
KGE_2 = two_dim_lon_transform(KGE_2)
KGE_3 = two_dim_lon_transform(KGE_3)

# NSE_  = np.load(out_path+'NSE_'+cfg['modelname'] +'.npy')
NSE_ = two_dim_lon_transform(NSE_)
NSE_1 = two_dim_lon_transform(NSE_1)
NSE_2 = two_dim_lon_transform(NSE_2)
NSE_3 = two_dim_lon_transform(NSE_3)

lat_file_name = 'lat_{s}.npy'.format(s=cfg['spatial_resolution'])
lon_file_name = 'lon_{s}.npy'.format(s=cfg['spatial_resolution'])

# gernate lon and lat
lat_ = np.load(PATH+lat_file_name)
lon_ = np.load(PATH+lon_file_name)
lon_ = np.linspace(-180,179,int(y_pred.shape[2]))

sites_lon_index=[190,305,38,109,200]
sites_lat_index=[35,35,26,136,90]

# sites_lon_index=[200,190,195,198,180]
# sites_lat_index=[30,35,85,98,90]

mask_data = r2_[mask==1]
total_data = mask_data.shape[0]

r2_values = []
r2_values1 = []
r2_values2 = []
r2_values3 = []
conditions = [
    static_climate == 1.0,
    static_climate == 2.0,
    static_climate == 3.0,
    (static_climate == 4.0) | (static_climate == 5.0),
    (static_climate == 6.0) | (static_climate == 7.0),
    (static_climate == 8.0) | (static_climate == 9.0) | (static_climate == 10.0),
    (static_climate == 11.0) | (static_climate == 12.0) | (static_climate == 13.0),
    (static_climate == 14.0) | (static_climate == 15.0) | (static_climate == 16.0),
    (static_climate == 17.0) | (static_climate == 18.0) | (static_climate == 19.0) | (static_climate == 20.0),
    (static_climate == 21.0) | (static_climate == 22.0) | (static_climate == 23.0) | (static_climate == 24.0),
    (static_climate == 25.0) | (static_climate == 26.0) | (static_climate == 27.0) | (static_climate == 28.0),
    (static_climate == 29.0),
    (static_climate == 30.0)
]

for i in range(1, 13):
    subset = r2_[conditions[i - 1]]
    r2_values.append(subset)
for i in range(1, 13):
    subset1 = r2_1[conditions[i - 1]]
    r2_values1.append(subset1)
for i in range(1, 13):
    subset2 = r2_2[conditions[i - 1]]
    r2_values2.append(subset2)
for i in range(1, 13):
    subset3 = r2_3[conditions[i - 1]]
    r2_values3.append(subset3)

# R6_r2_ = np.where(conditions[5], r2_, np.nan)
# R6_r2_1 = np.where(conditions[5], r2_1, np.nan)
# R6_r2_2 = np.where(conditions[5], r2_2, np.nan)
# R6_KGE_ = np.where(conditions[5], KGE_, np.nan)
# R6_KGE_1 = np.where(conditions[5], KGE_1, np.nan)
# R6_KGE_2 = np.where(conditions[5], KGE_2, np.nan)
# R6_NSE_ = np.where(conditions[5], NSE_, np.nan)
# R6_NSE_1 = np.where(conditions[5], NSE_1, np.nan)
# R6_NSE_2 = np.where(conditions[5], NSE_2, np.nan)

# R9_r2_ = np.where(conditions[8], r2_, np.nan)
# R9_r2_1 = np.where(conditions[8], r2_1, np.nan)
# R9_r2_2 = np.where(conditions[8], r2_2, np.nan)
# R9_KGE_ = np.where(conditions[8], KGE_, np.nan)
# R9_KGE_1 = np.where(conditions[8], KGE_1, np.nan)
# R9_KGE_2 = np.where(conditions[8], KGE_2, np.nan)
# R9_NSE_ = np.where(conditions[8], NSE_, np.nan)
# R9_NSE_1 = np.where(conditions[8], NSE_1, np.nan)
# R9_NSE_2 = np.where(conditions[8], NSE_2, np.nan)

# # 创建字典存储变量，分别对应三个模型
# model_variables = {'LSTM': {}, 'Meta-R9': {}, 'Meta-R12': {}}
#
# # 根据区域索引和指标类型存储变量
# for i in range(1, 13):  # 区域从1到12
# 	region_index = f'R{i}'
#
# 	model_variables['LSTM'][f'{region_index}_r2'] = np.where(conditions[i - 1], r2_, np.nan)
# 	model_variables['LSTM'][f'{region_index}_KGE'] = np.where(conditions[i - 1], KGE_, np.nan)
# 	model_variables['LSTM'][f'{region_index}_NSE'] = np.where(conditions[i - 1], NSE_, np.nan)
#
# 	model_variables['Meta-R9'][f'{region_index}_r2'] = np.where(conditions[i - 1], r2_1, np.nan)
# 	model_variables['Meta-R9'][f'{region_index}_KGE'] = np.where(conditions[i - 1], KGE_1, np.nan)
# 	model_variables['Meta-R9'][f'{region_index}_NSE'] = np.where(conditions[i - 1], NSE_1, np.nan)
#
# 	model_variables['Meta-R12'][f'{region_index}_r2'] = np.where(conditions[i - 1], r2_2, np.nan)
# 	model_variables['Meta-R12'][f'{region_index}_KGE'] = np.where(conditions[i - 1], KGE_2, np.nan)
# 	model_variables['Meta-R12'][f'{region_index}_NSE'] = np.where(conditions[i - 1], NSE_2, np.nan)
#
# # 打印每个区域的每个指标的平均值
# for model_name, model_data in model_variables.items():
# 	print(f'{model_name}模型：')
#
# 	for i in range(1, 13):
# 		region_index = f'R{i}'
#
# 		R_r2 = model_data[f'{region_index}_r2']
# 		R_KGE = model_data[f'{region_index}_KGE']
# 		R_NSE = model_data[f'{region_index}_NSE']
#
# 		print(f'{region_index}的r2平均值：', np.nanmedian(R_r2[mask == 1]))
# 		print(f'{region_index}的KGE平均值：', np.nanmedian(R_KGE[mask == 1]))
# 		print(f'{region_index}的NSE平均值：', np.nanmedian(R_NSE[mask == 1]))


# print('the average R9_r2 of LSTM model is :',np.nanmedian(R9_r2_[mask==1]))
# print('the average R9_KGE of LSTM model is :',np.nanmedian(R9_KGE_[mask==1]))
# print('the average R9_NSE of LSTM model is :',np.nanmedian(R9_NSE_[mask==1]))
# print('the average R9_r2 of Meta-R9 model is :',np.nanmedian(R9_r2_1[mask==1]))
# print('the average R9_KGE of Meta-R9 model is :',np.nanmedian(R9_KGE_1[mask==1]))
# print('the average R9_NSE of Meta-R9 model is :',np.nanmedian(R9_NSE_1[mask==1]))
# print('the average R9_r2 of Meta-R12 model is :',np.nanmedian(R9_r2_2[mask==1]))
# print('the average R9_KGE of Meta-R12 model is :',np.nanmedian(R9_KGE_2[mask==1]))
# print('the average R9_NSE of Meta-R12 model is :',np.nanmedian(R9_NSE_2[mask==1]))

# print('the average r2 of LSTM model is :',np.nanmedian(r2_[mask==1]))
# print('the average rmse of LSTM model is :',np.nanmedian(rmse_[mask==1]))
# print('the average KGE of LSTM model is :',np.nanmedian(KGE_[mask==1]))
# print('the average NSE of LSTM model is :',np.nanmedian(NSE_[mask==1]))
# print('the average r2 of Meta-R9 model is :',np.nanmedian(r2_1[mask==1]))
# print('the average rmse of Meta-R9 model is :',np.nanmedian(rmse_1[mask==1]))
# print('the average KGE of Meta-R9 model is :',np.nanmedian(KGE_1[mask==1]))
# print('the average NSE of Meta-R9 model is :',np.nanmedian(NSE_1[mask==1]))
# print('the average r2 of Meta-R12 model is :',np.nanmedian(r2_2[mask==1]))
# print('the average rmse of Meta-R12 model is :',np.nanmedian(rmse_2[mask==1]))
# print('the average KGE of Meta-R12 model is :',np.nanmedian(KGE_2[mask==1]))
# print('the average NSE of Meta-R12 model is :',np.nanmedian(NSE_2[mask==1]))
# print('the average r2 of Meta-R9+12 model is :',np.nanmedian(r2_3[mask==1]))
# print('the average rmse of Meta-R9+12 model is :',np.nanmedian(rmse_3[mask==1]))
# print('the average KGE of Meta-R9+12 model is :',np.nanmedian(KGE_3[mask==1]))
# print('the average NSE of Meta-R9+12 model is :',np.nanmedian(NSE_3[mask==1]))

# ------------------------------------------------------------------
# Figure 1： spatial distributions for r2
# ------------------------------------------------------------------
if plt_f in ['Fig.1']:
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.subplot(1, 1, 1)
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90., 91., 18.)
	meridians = np.arange(-180., 180., 36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	# convlstm
	r2_[mask == 0] = -9999
	cs = m.contourf(xi, yi, r2_, np.arange(-2, 1.1, 0.1), cmap='seismic')
	cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(-2, 1.1, 0.3))
	cbar.set_label('R$^{2}$')
	plt.title(name_pred)
	plt.savefig('../meta-result/lstm_r2(-2,1.1).pdf')
	plt.show()

# ------------------------------------------------------------------
# Figure 2： spatial distributions for rmse
# ------------------------------------------------------------------
if plt_f in ['Fig.2']:
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.figure
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,91.,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)

	# convlstm
	if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"]  or cfg['label'] == ["volumetric_soil_water_layer_20"]:
		cs = m.contourf(xi,yi, rmse_, np.arange(0, 0.13, 0.01), cmap='RdBu')
		cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(0, 0.13, 0.02))
		cbar.set_label('RMSE(m$^{3}$/m$^{3}$)')
	elif cfg['label'] == ["surface_sensible_heat_flux"]:
		cs = m.contourf(xi,yi, rmse_, np.arange(0, 51, 5), cmap='RdBu')
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('RMSE(W/m$^{2}$)')
	plt.title('LSTM')
	print('Figure 8: spatial distributions for rmse completed!')
	plt.savefig('../meta-result/lstm_rmse(0, 0.21).pdf')
	plt.show()

# ------------------------------------------------------------------
# Figure 3： spatial distributions for KGE
# ------------------------------------------------------------------
if plt_f in ['Fig.3']:
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.subplot(1, 1, 1)
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90., 91., 18.)
	meridians = np.arange(-180., 180., 36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	# convlstm
	KGE_[mask == 0] = -9999
	cs = m.contourf(xi, yi, KGE_, np.arange(-2, 1.1, 0.1), cmap='seismic')
	cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(-2, 1.1, 0.3))
	cbar.set_label('KGE')
	plt.title(name_pred)
	plt.savefig('../meta-result/lstm_KGE(-2,1.1).pdf')
	plt.show()

# ------------------------------------------------------------------
# Figure 4： spatial distributions for NSE
# ------------------------------------------------------------------
if plt_f in ['Fig.4']:
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.subplot(1, 1, 1)
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90., 91., 18.)
	meridians = np.arange(-180., 180., 36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	# convlstm
	NSE_[mask == 0] = -9999
	cs = m.contourf(xi, yi, NSE_, np.arange(-2, 1.1, 0.1), cmap='seismic')
	cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(-2, 1.1, 0.3))
	cbar.set_label('NSE')
	plt.title(name_pred)
	plt.savefig('../meta-result/lstm_NSE(-2,1.1).pdf')
	plt.show()



# if plt_f in ['Fig.5']:
	# plt.rcParams['font.family'] = 'Times New Roman'
	# # 创建图和轴
	# fig, ax = plt.subplots()
	#
	# # 循环遍历每个子集并创建一个箱线图
	# for i, subset in enumerate(r2_values):
	# 	subset = subset[~np.isnan(subset)]
	# 	subset = subset[~np.isinf(subset)]
	# 	box = plt.boxplot(subset,
	# 					 positions=[i+1],  # 将箱子放置在x轴上的位置i+1
	# 					 notch=True,
	# 					 patch_artist=True,
	# 					 showfliers=False,
	# 					 boxprops=dict(facecolor='red', color='black'),
	# 					 widths=0.45)  # 将颜色设置为红色
	#
	# 	# 设置箱体的边框宽度
	# 	box_path = box['boxes'][0]
	# 	box_path.set_linewidth(1)
	#
	# 	# 设置中位数线的宽度
	# 	median_line = box['medians'][0]
	# 	median_line.set_linewidth(1)
	#
	# # 设置y轴的限制
	# plt.ylim(bottom=-1.30, top=1.30)
	#
	# # 添加标题和标签
	# plt.title('LSTM')
	# plt.ylabel('R$^{2}$')
	#
	# # 设置x轴刻度和标签
	# plt.xticks(np.arange(1, len(r2_values)+1), [f'R{i+1}' for i in range(len(r2_values))])
	#
	# # plt.savefig('../meta-result/lstm_r2_region12.pdf')
	# # 显示图形
	# plt.show()

# ------------------------------------------------------------------
# Figure 6： Box Plot
# ------------------------------------------------------------------
if plt_f in ['Fig.6']:

	plt.rcParams['font.family'] = 'Times New Roman'

	# Creating figure and axis
	fig, ax = plt.subplots(figsize=(13, 6), dpi=100)

	# Number of subsets
	num_subsets = len(r2_values)

	# Loop through each subset and create a boxplot
	colors = ['red', 'green', 'blue', 'orange']
	legend_labels = ['LSTM', 'Meta-R9', 'Meta-R12', 'Meta-R9+12']
	for i, (subset, subset1, subset2, subset3) in enumerate(zip(r2_values, r2_values1, r2_values2, r2_values3)):
		for j, data in enumerate([subset, subset1, subset2, subset3]):
			data = data[~np.isnan(data)]
			data = data[~np.isinf(data)]

			# Create boxplot at position (i + 1) * (j + 1)
			position = i * 4 + j + 1
			box = plt.boxplot(data,
							  positions=[position],
							  notch=True,
							  patch_artist=True,
							  showfliers=False,
							  boxprops=dict(facecolor=colors[j], color='black'),
							  widths=0.6)  # Adjust the width as needed

			# Set box and median line properties
			box_path = box['boxes'][0]
			box_path.set_linewidth(1)

			median_line = box['medians'][0]
			median_line.set_linewidth(1)

			# Add a dashed line behind each group of four boxes
			if j == 3:
				plt.axvline(x=position + 0.5, color='black', linestyle='--', linewidth=1)

	# Set y-axis limits
	plt.ylim(bottom=-1.20, top=1.20)

	# Add title and labels
	plt.title('LSTM')
	plt.ylabel('R$^{2}$')

	# Set x-axis ticks and labels
	midpoints = np.arange(1.5, num_subsets * 4 + 0.5, 4)
	plt.xticks(midpoints, [f'R{i // 4 + 1}' for i in range(0, num_subsets * 4, 4)])

	num_data_points = [763, 467, 1507, 2859, 1740, 286, 519, 896, 387, 705, 1599, 1989]

	# Create a twin Axes sharing the xaxis
	ax2 = ax.twinx()

	# Plot the number of data points on the new y-axis
	ax2.plot(midpoints, num_data_points, marker='o', linestyle='-', color='gray', label='Number of Data Points')

	# Set y-axis limits for the second y-axis
	ax2.set_ylim(bottom=0, top=max(num_data_points) + 1000)  # Adjust the top limit as needed

	# Add y-axis label for the second y-axis
	ax2.set_ylabel('Number of Data Points', color='gray')

	# Display the legend for both y-axes
	ax.legend([plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors],
			  legend_labels,
			  loc='lower left', bbox_to_anchor=(0.02, 0.02))  # Adjust the values as needed

	# Save or display the plot
	plt.savefig('../meta-result/lst+meta_r2_region12-1.pdf', bbox_inches='tight')
	plt.show()

# ---------------------------------
# Figure 7： time series plot
# ---------------------------------
if plt_f in ['Fig.7']:
	plt.rcParams['font.family'] = 'Times New Roman'

	# 绘制第一张图
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90., 91, 18.)
	meridians = np.arange(-180., 181., 36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
		lon = lon_[int(lon_index)]
		lat = lat_[int(lat_index)]
		plt.plot(lon, lat, marker='*', color='red', markersize=9)
	plt.legend(loc=0)

	plt.savefig('../meta-result/time/figure1.pdf')  # 保存第一张图
	plt.show()

	# 绘制第二张图
	data_all = [y_test, y_pred, y_pred2]
	color_list = ['black', 'blue', 'red']
	plt.rcParams['font.size'] = 18
	legend_labels = ['Real', 'LSTM', 'Meta-R12']
	for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
		count = 0
		fig, axs = plt.subplots(1, 1, figsize=(15, 3))
		# print('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
		# print('r2 of meta is', r2_2[lat_index, lon_index])
		# print('KGE_ of meta is', KGE_2[lat_index, lon_index])
		# print('rmse of meta is', rmse_2[lat_index, lon_index])
		# print('NSE_ of meta is', NSE_2[lat_index, lon_index])
		# print('The difference of r2 is', r2_2[lat_index, lon_index] - r2_[lat_index, lon_index])
		# print('The difference of KGE is', KGE_2[lat_index, lon_index] - KGE_[lat_index, lon_index])
		# print('The difference of rmse is', rmse_2[lat_index, lon_index] - rmse_[lat_index, lon_index])
		# print('The difference of NSE is', NSE_2[lat_index, lon_index] - NSE_[lat_index, lon_index])
		for data_f5plt in (data_all):
			axs.plot(data_f5plt[:, lat_index, lon_index], color=color_list[count])
			axs.legend(loc=1)
			count = count + 1

		axs.set_title('Latitude:{lat_v} and Longitude:{ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
		# plt.legend([plt.Line2D([0], [0], color=color, linewidth=2) for color in color_list], legend_labels, loc='upper right')

		plt.savefig(f'../meta-result/time/figure1{lon_[int(lon_index)]}_{lat_[int(lat_index)]}.pdf')  # 保存当前图
		plt.show()

	print('Figure 7: 时间序列图绘制完成！')


#CDF
if plt_f in ['Fig.8']:
	plt.rcParams['font.family'] = 'Times New Roman'

	# r2_ = r2_[~np.isnan(r2_)]
	# r2_ = r2_[~np.isinf(r2_)]
	# r2_ = r2_[r2_ != 1]
	# r2_1 = r2_1[~np.isnan(r2_1)]
	# r2_1 = r2_1[~np.isinf(r2_1)]
	# r2_1 = r2_1[r2_1 != 1]
	# r2_2 = r2_2[~np.isnan(r2_2)]
	# r2_2 = r2_2[~np.isinf(r2_2)]
	# r2_2 = r2_2[r2_2 != 1]
	# r2_3 = r2_3[~np.isnan(r2_3)]
	# r2_3 = r2_3[~np.isinf(r2_3)]
	# r2_3 = r2_3[r2_3 != 1]

	# KGE_ = KGE_[~np.isnan(KGE_)]
	# KGE_ = KGE_[~np.isinf(KGE_)]
	# KGE_1 = KGE_1[~np.isnan(KGE_1)]
	# KGE_1 = KGE_1[~np.isinf(KGE_1)]
	# KGE_2 = KGE_2[~np.isnan(KGE_2)]
	# KGE_2 = KGE_2[~np.isinf(KGE_2)]
	# KGE_3 = KGE_3[~np.isnan(KGE_3)]
	# KGE_3 = KGE_3[~np.isinf(KGE_3)]

	NSE_ = NSE_[~np.isnan(NSE_)]
	NSE_ = NSE_[~np.isinf(NSE_)]
	NSE_1 = NSE_1[~np.isnan(NSE_1)]
	NSE_1 = NSE_1[~np.isinf(NSE_1)]
	NSE_2 = NSE_2[~np.isnan(NSE_2)]
	NSE_2 = NSE_2[~np.isinf(NSE_2)]
	NSE_3 = NSE_3[~np.isnan(NSE_3)]
	NSE_3 = NSE_3[~np.isinf(NSE_3)]

	# bias_ = bias_[~np.isnan(bias_)]
	# bias_ = bias_[~np.isinf(bias_)]
	# bias_1 = bias_1[~np.isnan(bias_1)]
	# bias_1 = bias_1[~np.isinf(bias_1)]
	# bias_2 = bias_2[~np.isnan(bias_2)]
	# bias_2 = bias_2[~np.isinf(bias_2)]
	# bias_3 = bias_3[~np.isnan(bias_3)]
	# bias_3 = bias_3[~np.isinf(bias_3)]

	# bias1_ = bias1_[~np.isnan(bias1_)]
	# bias1_ = bias1_[~np.isinf(bias1_)]
	# bias1_1 = bias1_1[~np.isnan(bias1_1)]
	# bias1_1 = bias1_1[~np.isinf(bias1_1)]
	# bias1_2 = bias1_2[~np.isnan(bias1_2)]
	# bias1_2 = bias1_2[~np.isinf(bias1_2)]
	# bias1_3 = bias1_3[~np.isnan(bias1_3)]
	# bias1_3 = bias1_3[~np.isinf(bias1_3)]

	# rmse_ = rmse_[~np.isnan(rmse_)]
	# rmse_ = rmse_[~np.isinf(rmse_)]
	# rmse_1 = rmse_1[~np.isnan(rmse_1)]
	# rmse_1 = rmse_1[~np.isinf(rmse_1)]
	# rmse_2 = rmse_2[~np.isnan(rmse_2)]
	# rmse_2 = rmse_2[~np.isinf(rmse_2)]
	# rmse_3 = rmse_3[~np.isnan(rmse_3)]
	# rmse_3 = rmse_3[~np.isinf(rmse_3)]



	# lstm = np.sum(r2_ == 1)
	# meta_r9 = np.sum(r2_1 == 1)
	# meta_r12 = np.sum(r2_2 == 1)
	# meta_r9_12 = np.sum(r2_3 == 1)


	# Calculate CDF for each array
	data_sorted1 = np.sort(NSE_.flatten())
	cdf1 = np.arange(1, len(data_sorted1) + 1) / len(data_sorted1)

	data_sorted2 = np.sort(NSE_1.flatten())
	cdf2 = np.arange(1, len(data_sorted2) + 1) / len(data_sorted2)

	data_sorted3 = np.sort(NSE_2.flatten())
	cdf3 = np.arange(1, len(data_sorted3) + 1) / len(data_sorted3)

	data_sorted4 = np.sort(NSE_3.flatten())
	cdf4 = np.arange(1, len(data_sorted4) + 1) / len(data_sorted4)

	# Plot CDFs
	plt.plot(data_sorted1, cdf1, label='LSTM')
	plt.plot(data_sorted2, cdf2, label='Meta-R9')
	plt.plot(data_sorted3, cdf3, label='Meta-R12')
	plt.plot(data_sorted4, cdf4, label='Meta-R9+12')

	# Add legend and labels
	# plt.xlabel('R$^{2}$')
	plt.xlabel('NSE')
	plt.ylabel('CDF')
	# plt.title('CDFs of Data')
	plt.legend(loc='best')
	plt.ylim(0.98, 1.02)
	plt.xlim(0.975, 0.995)
	plt.grid(True)
	plt.savefig(f'../meta-result/CDF_NSE1.pdf')
	plt.show()

#GIS
if plt_f in ['Fig.9']:
	plt.rcParams['font.family'] = 'Times New Roman'

	# 创建经度和纬度网格
	lon, lat = np.meshgrid(lon_, lat_)

	plt.rcParams.update({'font.size': 25})


	m = Basemap()
	parallels = np.arange(-90., 91, 18.)
	meridians = np.arange(-180., 181., 36.)

	# Create a figure with one subplot
	fig, ax1 = plt.subplots(figsize=(15, 10))

	# Plotting the GIS map
	m.drawcoastlines()
	m.drawcountries()
	m.drawstates()
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])

	NSE_2 = np.where(conditions[5], NSE_2, np.inf)

	x, y = m(lon, lat)
	sc = m.scatter(x, y, c=NSE_2.flatten(), cmap='coolwarm', norm=Normalize(vmin=0, vmax=1), s=50)
	cbar = m.colorbar(sc, location='bottom',  pad="10%", label='NSE')
	# cbar.set_label('R$^{2}$')
	cbar.set_label('NSE')
	plt.title('Meta-R12')
	plt.rcParams.update({'font.size': 12})
	# Inset bar graph in the map (left-bottom corner)
	inset_ax = fig.add_axes([0.18, 0.35, 0.15, 0.15])  # x, y, width, height
	inset_ax.hist(NSE_2.flatten(), bins=5, color='blue', alpha=0.7, range=(0, 1))
	inset_ax.set_title('NSE Distribution')
	# inset_ax.set_xlabel('KGE')
	inset_ax.set_ylabel('Frequency')
	plt.savefig(f'../meta-result/Region6_meta_NSE.pdf')
	# Show the first plot
	plt.show()

if plt_f in ['Fig.10']:
	# Simulating some data for hydrograph
	days = np.linspace(1, 365, 365)
	observed_streamflow = np.sin(np.linspace(0, 2 * np.pi, 365)) * np.random.uniform(0.8, 1.2, 365) + 1
	ensemble_mean = np.sin(np.linspace(0, 2 * np.pi, 365)) + 1
	std_dev = np.random.uniform(0.05, 0.1, 365)

	# Create a new figure for the hydrograph
	fig, ax = plt.subplots(figsize=(15, 5))

	# Plotting the hydrograph
	ax.fill_between(days, ensemble_mean - 2 * std_dev, ensemble_mean + 2 * std_dev, color='lightblue',
					label='2σ interval')
	ax.plot(days, observed_streamflow, label='Observation', color='black')
	ax.plot(days, ensemble_mean, label='Ensemble mean', color='blue')
	ax.set_xlim(1, 365)
	ax.set_xticks(np.linspace(1, 365, 12))  # Approximate middle of each month
	ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
	ax.set_xlabel('Day')
	ax.set_ylabel('Normalized Streamflow')
	ax.set_title('Snowmelt Station; NSE = 0.87')
	ax.legend()

	# Show the second plot
	plt.show()

if plt_f in ['Fig.11']:
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams.update({'font.size': 28})


	median = conditions[12-1]
	indices = np.where(median)

	y_test[:, ~median] = -0
	y_test = np.squeeze(y_test)
	y_test = y_test[-365:, :, :]
	y_test[:, mask == 0] = -0

	y_pred[:, ~median] = -0
	y_pred = np.squeeze(y_pred)
	y_pred = y_pred[-365:, :, :]
	y_pred[:, mask == 0] = -0

	y_pred1[:, ~median] = -0
	y_pred1 = np.squeeze(y_pred1)
	y_pred1 = y_pred1[-365:, :, :]
	y_pred1[:, mask == 0] = -0

	y_pred2[:, ~median] = -0
	y_pred2 = np.squeeze(y_pred2)
	y_pred2 = y_pred2[-365:, :, :]
	y_pred2[:, mask == 0] = -0

	y_pred3[:, ~median] = -0
	y_pred3 = np.squeeze(y_pred3)
	y_pred3 = y_pred3[-365:, :, :]
	y_pred3[:, mask == 0] = -0

	fig, ax = plt.subplots(figsize=(16, 13))  # Use a single subplot

	colormap = plt.get_cmap("Blues")
	colormap1 = plt.get_cmap("Reds")
	colormap2 = plt.get_cmap("Greens")

	# # Calculate scatter plot for the first set of data
	# sizes = np.pi * 2 ** 1
	# observed_sm_mean1 = np.average(y_test, axis=0)
	# predicted_sm_mean1 = np.average(y_pred, axis=0)
	# sm_index = observed_sm_mean1 != 0
	# observed_sm_mean1 = observed_sm_mean1[sm_index]
	# predicted_sm_mean1 = predicted_sm_mean1[sm_index]
	# m, b = best_fit_slope_and_intercept(observed_sm_mean1, predicted_sm_mean1)
	# regression_line = [(m * a) + b for a in observed_sm_mean1]
	# radius = 0.1  # Radius
	#
	# Z1 = density_calc(observed_sm_mean1, predicted_sm_mean1, radius)
	#
	# ax.scatter(observed_sm_mean1, predicted_sm_mean1, c=Z1, s=sizes, cmap=colormap,
	# 		   norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
	# ax.plot(observed_sm_mean1, regression_line, 'red', lw=0.8)  # Regression line

	# # Calculate scatter plot for the second set of data
	# sizes = np.pi * 2 ** 1
	# observed_sm_mean2 = np.average(y_test, axis=0)
	# predicted_sm_mean2 = np.average(y_pred1, axis=0)
	# sm_index = observed_sm_mean2 != 0
	# observed_sm_mean2 = observed_sm_mean2[sm_index]
	# predicted_sm_mean2 = predicted_sm_mean2[sm_index]
	# m, b = best_fit_slope_and_intercept(observed_sm_mean2, predicted_sm_mean2)
	# regression_line = [(m * a) + b for a in observed_sm_mean2]
	# radius = 0.1  # Radius
	#
	# Z2 = density_calc(observed_sm_mean2, predicted_sm_mean2, radius)
	#
	# ax.scatter(observed_sm_mean2, predicted_sm_mean2, c=Z2, s=sizes, cmap=colormap1,
	# 		   norm=colors.LogNorm(vmin=Z2.min(), vmax=Z2.max()))
	# ax.plot(observed_sm_mean2, regression_line, 'red', lw=0.8)  # Regression line

	# Calculate scatter plot for the third set of data
	sizes = np.pi * 2 ** 1
	observed_sm_mean3 = np.average(y_test, axis=0)
	predicted_sm_mean3 = np.average(y_pred2, axis=0)
	sm_index = observed_sm_mean3 != 0
	observed_sm_mean3 = observed_sm_mean3[sm_index]
	predicted_sm_mean3 = predicted_sm_mean3[sm_index]
	m, b = best_fit_slope_and_intercept(observed_sm_mean3, predicted_sm_mean3)
	regression_line = [(m * a) + b for a in observed_sm_mean3]
	radius = 0.1  # Radius

	Z3 = density_calc(observed_sm_mean3, predicted_sm_mean3, radius)

	ax.scatter(observed_sm_mean3, predicted_sm_mean3, c=Z3, s=sizes, cmap=colormap2,
			   norm=colors.LogNorm(vmin=Z3.min(), vmax=Z3.max()))
	ax.plot(observed_sm_mean3, regression_line, 'red', lw=0.8)  # Regression line

	ax.plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  # y=x
	ax.set_title('Meta-R12(Region12)')
	ax.set_xlabel('Observed SM')
	ax.set_ylabel('0d Predicted SM')
	metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
	ax.text(0.05, 0.67, metrics)
	ax.text(0.05, 0.62, "R$^{2}$=0.3284")
	ax.text(0.05, 0.57, "KGE=0.6485")
	ax.text(0.05, 0.52, "NSE=0.0730")

	plt.tight_layout()
	plt.savefig(f'../meta-result/区域散点图/R12_meta-r12.pdf')
	plt.show()


