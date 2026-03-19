import time
import numpy as np
import pandas as pd
import os
import pandas as pd
import torch
import torch.nn
from tqdm import trange
from data_gen import load_test_data_for_rnn,load_train_data_for_rnn, load_test_data_for_cnn, load_train_data_for_cnn, \
    erath_data_transform, sea_mask_rnn, sea_mask_cnn
from loss import  NaNMSELoss
from model import LSTMModel,CNN,ConvLSTMModel
from cross_models.cross_former import Crossformer
import matplotlib.pyplot as plt

# Call the plotting function

def train(x,
          y,
          static,
          mask,
          scaler_x,
          scaler_y,
          cfg,
          num_repeat,
          PATH,
          out_path,
          device,
          num_task=None,
          valid_split=True, scaler=None):



    patience = cfg['patience']
    wait = 0
    best = 9999
    valid_split=cfg['valid_split']
    print('the device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static.dtype))
    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
#	Splice x according to the sphere shape
        lat_index,lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    if valid_split:
        nt,nf,nlat,nlon = x.shape  #x shape :nt,nf,nlat,nlon
	#Partition validation set and training set
        N = int(nt*cfg['split_ratio'])
        x_valid, y_valid, static_valid = x[N:], y[N:], static
        x, y = x[:N], y[:N]       

    lossmse = torch.nn.MSELoss()
#	filter Antatctica
    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static.shape)
    print('mask shape is', mask.shape)

    # mask see regions
    #Determine the land boundary
    if cfg['modelname'] in ['LSTM']:
        if valid_split:

            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['CNN','ConvLSTM']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['crossformer']:
        if valid_split:
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)

    # train and validate
    # NOTE: We preprare two callbacks for training:
    #       early stopping and save best model.
    for num_ in range(cfg['num_repeat']):
        # prepare models
	#Selection model
        if cfg['modelname'] in ['LSTM']:
            lstmmodel_cfg = {}
            lstmmodel_cfg['input_size'] = cfg["input_size"]
            lstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
            lstmmodel_cfg['out_size'] = 1
            model = LSTMModel(cfg,lstmmodel_cfg).to(device)
        elif cfg['modelname'] in ['crossformer']:
            model = Crossformer(cfg["input_size"],
                                cfg['seq_len'],1, 14,2,10,256,512,4,\
                                3,0.2,False,device).to(device)
        elif cfg['modelname'] in ['CNN']:
            model = CNN(cfg).to(device)
        elif cfg['modelname'] in ['ConvLSTM']:
            model = ConvLSTMModel(cfg).to(device)

      #  model.train()
	 # Prepare for training
    # NOTE: Only use `Adam`, we didn't apply adaptively
    #       learing rate schedule. We found `Adam` perform
    #       much better than `Adagrad`, `Adadelta`.
        optim = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])

        with trange(1, cfg['epochs']+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname']+' '+str(num_repeat))
                t_begin = time.time()
                # train
                MSELoss = 0
                for iter in range(0, cfg["niter"]):
 # ------------------------------------------------------------------------------------------------------------------------------
 #  train way for LSTM model
                    if cfg["modelname"] in \
                            ['LSTM']:
                        # generate batch data for Recurrent Neural Network

                        x_batch, y_batch, aux_batch, _, _ = \
                        load_train_data_for_rnn(cfg, x, y, static, scaler_y)
                        
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        #print('aux_batch[:,5,0]',aux_batch[:,5,0])
                        #print('x_batch[:,5,0]',x_batch[:,5,0])
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred = model(x_batch, aux_batch)
                        pred = torch.squeeze(pred,1)

                    elif cfg['modelname'] in ['crossformer']:
                         # generate batch data for Recurrent Neural Network
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn(cfg, x, y, static, scaler_y)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        #print('aux_batch[:,5,0]',aux_batch[:,5,0])
                        #print('x_batch[:,5,0]',x_batch[:,5,0])
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred = model(x_batch)
                        pred = torch.squeeze(pred,1)
 #  train way for CNN model
                    elif cfg['modelname'] in ['CNN']:
                        # generate batch data for Convolutional Neural Network
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index)
                        x_batch[np.isnan(x_batch)] = 0  # filter nan values to train cnn model
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        x_batch = x_batch.squeeze(dim=1)
                        x_batch = x_batch.reshape(x_batch.shape[0],x_batch.shape[1]*x_batch.shape[2],x_batch.shape[3],x_batch.shape[4])
                        x_batch = torch.cat([x_batch, aux_batch], 1)
                        pred = model(x_batch, aux_batch)
                    elif cfg['modelname'] in ['ConvLSTM']:
                        # generate batch data for Convolutional LSTM
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index) # same as Convolutional Neural Network
                        x_batch[np.isnan(x_batch)] = 0  # filter nan values to train cnn model
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1,1,1)
                        x_batch = x_batch.squeeze(dim=1)
                        pred = model(x_batch, aux_batch,cfg)
 # ------------------------------------------------------------------------------------------------------------------------------
                    loss = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(),lossmse)                              
                    optim.zero_grad()
                    loss.backward()
                    optim.step()                    
                    MSELoss += loss.item()
# ------------------------------------------------------------------------------------------------------------------------------
                t_end = time.time()
                # get loss log
                loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss / cfg["niter"], t_end - t_begin)
                print(loss_str)
                # validate
		#Use validation sets to test trained models
		#If the error is smaller than the minimum error, then save the model.
                if valid_split:
                    del x_batch, y_batch, aux_batch
                    MSE_valid_loss = 0
                    if epoch % 20 == 0:
                        wait += 1
                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
# ------------------------------------------------------------------------------------------------------------------------------
 #  validate way for LSTM model
                        if cfg["modelname"] in ['LSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid.shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)                              
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1)
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
#  validate way for CNN model
                        elif cfg['modelname'] in ['CNN']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network

                                x_valid_batch[np.isnan(x_valid_batch)] = 0
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                x_valid_batch = x_valid_batch.squeeze(1)
                                x_valid_batch = x_valid_batch.reshape(x_valid_batch.shape[0],x_valid_batch.shape[1]*x_valid_batch.shape[2],x_valid_batch.shape[3],x_valid_batch.shape[4])
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], axis=1)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid, y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
#  validate way for ConvLSTM model，same as CNN model
                        elif cfg['modelname'] in ['ConvLSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network

                                x_valid_batch[np.isnan(x_valid_batch)] = 0
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1,1,1)
                                # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                x_valid_batch = x_valid_batch
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch,cfg)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid, y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
# ------------------------------------------------------------------------------------------------------------------------------


                        t_end = time.time()
                        mse_valid_loss = MSE_valid_loss/(len(gt_list))
                        # get loss log




                        loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch,mse_valid_loss, 
                                    t_end-t_begin)
                        print(loss_str)
                        val_save_acc = mse_valid_loss
                        # Save drift records to a file
                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                    if val_save_acc < best:
                        #if MSE_valid_loss < best:
                            torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                            wait = 0  # release wait
                            best = val_save_acc #MSE_valid_loss
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                else:
            # save best model by train loss
                    if MSELoss < best  :
                        best = MSELoss
                        wait = 0
                        #if epoch <=300:
                        #    torch.save(model,out_path+'/300/'+cfg['modelname']+'_para.pkl')
                        #if epoch >300 and epoch<=500:
                       #     torch.save(model,out_path+'/500/'+cfg['modelname']+'_para.pkl')                  
                        torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
        



                # early stopping
                if wait >= patience:
             
                    return
                                        
        return




'''
class DifficultBatchManager:
    def __init__(self, threshold_factor=2, max_difficult_batches=100):
        """
        初始化困难批次管理器。
        :param threshold_factor: 损失阈值的倍数（相对于平均损失）。
        :param max_difficult_batches: 最多记录的困难批次数量。
        """
        self.threshold_factor = threshold_factor
        self.max_difficult_batches = max_difficult_batches
        self.difficult_batches = []

    def record_difficult_batches(self, loss, x_batch, y_batch, aux_batch):
        """
        记录损失过大的批次。
        :param loss: 当前批次的损失值。
        :param x_batch: 输入数据。
        :param y_batch: 目标值。
        :param aux_batch: 辅助数据（如果有）。
        """
        if len(self.difficult_batches) >= self.max_difficult_batches:
            return  # 如果超过最大记录数量，直接返回

        # 将当前批次的数据和损失值记录下来
        self.difficult_batches.append({
            "loss": loss,
            "x_batch": x_batch.cpu().numpy() if isinstance(x_batch, torch.Tensor) else x_batch,
            "y_batch": y_batch.cpu().numpy() if isinstance(y_batch, torch.Tensor) else y_batch,
            "aux_batch": aux_batch.cpu().numpy() if isinstance(aux_batch, torch.Tensor) else aux_batch
        })

    def get_difficult_batches(self, num_samples=None):
        """
        获取记录的困难批次。
        :param num_samples: 采样数量（如果为 None，则返回所有记录的批次）。
        :return: 采样后的困难批次列表。
        """
        if num_samples is None:
            return self.difficult_batches
        else:
            return np.random.choice(self.difficult_batches, size=min(num_samples, len(self.difficult_batches)), replace=False)

    def clear_difficult_batches(self):
        """
        清空记录的困难批次。
        """
        self.difficult_batches = []

    def compute_threshold(self, average_loss):
        """
        计算损失值的阈值。
        :param average_loss: 当前 epoch 的平均损失。
        :return: 阈值。
        """
        return average_loss * self.threshold_factor


# 在训练代码中集成 DifficultBatchManager


def train(x, y, static, mask, scaler_x, scaler_y, cfg, num_repeat, PATH, out_path, device, num_task=None, valid_split=True):
    patience = cfg['patience']
    wait = 0
    best = 9999
    valid_split = cfg['valid_split']  # Use cfg to determine valid_split
    print('The device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static.dtype))

    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
        # Splice x according to the sphere shape
        lat_index, lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))

    if valid_split:
        nt, nf, nlat, nlon = x.shape  # x shape: nt, nf, nlat, nlon
        # Partition validation set and training set
        N = int(nt * cfg['split_ratio'])
        x_valid, y_valid, static_valid = x[N:], y[N:], static
        x, y = x[:N], y[:N]

    lossmse = torch.nn.MSELoss()

    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static.shape)
    print('mask shape is', mask.shape)

    # Apply mask to remove regions (e.g., Antarctica)
    if cfg['modelname'] in ['LSTM']:
        if valid_split:
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['CNN', 'ConvLSTM']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['crossformer']:
        if valid_split:
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)

    # Train and validate
    for num_ in range(cfg['num_repeat']):
        # Initialize model
        if cfg['modelname'] in ['LSTM']:
            lstmmodel_cfg = {
                'input_size': cfg["input_size"],
                'hidden_size': cfg["hidden_size"],
                'out_size': 1,
                #"nhead": 2,  # 多头自注意力机制的头数
                #"num_layers": 3,  # Transformer 编码器的层数

            }
            model = LSTMModel(cfg, lstmmodel_cfg).to(device)
        elif cfg['modelname'] in ['crossformer']:
            model = Crossformer(cfg["input_size"], cfg['seq_len'], 1, 14, 2, 10, 256, 512, 4, 3, 0.2, False, device).to(device)
        elif cfg['modelname'] in ['CNN']:
            model = CNN(cfg).to(device)
        elif cfg['modelname'] in ['ConvLSTM']:
            model = ConvLSTMModel(cfg).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

        # Initialize DifficultBatchManager
        difficult_batch_manager = DifficultBatchManager(threshold_factor=2, max_difficult_batches=100)

        with trange(1, cfg['epochs'] + 1) as pbar:
            for epoch in pbar:
                pbar.set_description(cfg['modelname'] + ' ' + str(num_repeat))
                t_begin = time.time()
                MSELoss = 0
                epoch_loss = 0
                num_batches = 0

                # Normal training batches
                for iter in range(cfg["niter"]):
                    if cfg["modelname"] in ['LSTM']:
                        x_batch, y_batch, aux_batch, _, _ ,last_end_time_idx= load_train_data_for_rnn(cfg, x, y, static, scaler_y, last_end_time_idx=0)
                    elif cfg['modelname'] in ['crossformer', 'CNN', 'ConvLSTM']:
                        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_cnn(cfg, x, y, static, scaler_y, lat_index, lon_index, mask_index)

                    x_batch = torch.from_numpy(x_batch).to(device)
                    y_batch = torch.from_numpy(y_batch).to(device)
                    aux_batch = torch.from_numpy(aux_batch).to(device)

                    # Adjust input data based on model type







                    if cfg['modelname'] in ['LSTM', 'crossformer']:
                        aux_batch = aux_batch.unsqueeze(1).repeat(1, x_batch.shape[1],1)
                        x_batch = torch.cat([x_batch, aux_batch], dim=2)

                    elif cfg['modelname'] in ['CNN', 'ConvLSTM']:
                        aux_batch = aux_batch.unsqueeze(1)  # Ensure aux_batch has the same number of dimensions as x_batch
                        x_batch = torch.cat([x_batch, aux_batch.expand_as(x_batch)], dim=1)

                    # Forward pass and compute loss
                    pred = model(x_batch, aux_batch)
                    loss = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(), lossmse)

                    # Backward pass and optimization
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    # Accumulate loss
                    MSELoss += loss.item()
                    epoch_loss += loss.item()
                    num_batches += 1

                    # Check if the batch is difficult and record it
                    avg_loss = epoch_loss / num_batches
                    threshold = difficult_batch_manager.compute_threshold(avg_loss)
                    if loss.item() > threshold:
                        difficult_batch_manager.record_difficult_batches(loss.item(), x_batch.cpu().numpy(), y_batch.cpu().numpy(), aux_batch.cpu().numpy())

                # Retrain difficult batches
                difficult_batches = difficult_batch_manager.get_difficult_batches(num_samples=10)
                for batch in difficult_batches:
                    x_batch = torch.from_numpy(batch["x_batch"]).to(device)
                    y_batch = torch.from_numpy(batch["y_batch"]).to(device)
                    aux_batch = torch.from_numpy(batch["aux_batch"]).to(device)

                    # Adjust input data based on model type
                    # 假设 aux_batch 是一个已知的张量
                    aux_size = aux_batch.shape[-1]  # 获取辅助特征的维度
                    cfg["aux_size"] = aux_size  # 动态设置 aux_size

                    # 初始化模型
                    # 更新模型的输入层配置
                    # 数据处理部分保持不变
                    if cfg['modelname'] in ['LSTM', 'crossformer']:
                        # 检查 aux_batch 的形状是否符合要求
                        if aux_batch.dim() == 3 and aux_batch.shape[1] == x_batch.shape[1]:
                            pass  # aux_batch 已经是三维张量，且时间步维度与 x_batch 一致
                        else:
                            aux_batch = aux_batch.unsqueeze(1).repeat(1, x_batch.shape[1], 1)

                        # 打印形状以确认
                        print("x_batch shape:", x_batch.shape)  # 应为 [39, 7, 15]
                        print("aux_batch shape:", aux_batch.shape)  # 应为 [39, 7, 6]





                    ''''''if cfg['modelname'] in ['LSTM', 'crossformer']:
                        aux_batch = aux_batch.unsqueeze(1).repeat(1, x_batch.shape[1], 1)
                        x_batch = torch.cat([x_batch, aux_batch], dim=2)
                    elif cfg['modelname'] in ['CNN', 'ConvLSTM']:
                        aux_batch = aux_batch.unsqueeze(1)
                        x_batch = torch.cat([x_batch, aux_batch.expand_as(x_batch)], dim=1)''''''

                    # Forward pass and compute loss
                    pred = model(x_batch, aux_batch)
                    loss = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(), lossmse)

                    # Backward pass and optimization
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Log training progress
                t_end = time.time()
                print("Epoch {} Train MSE Loss {:.3f} Time {:.2f}".format(epoch, MSELoss / cfg["niter"], t_end - t_begin))
                val_save_acc = MSELoss / cfg["niter"]

                # save best model by val loss
                # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                #       save best NSE results get `multi_tasks` better than `single_task`
                if val_save_acc < best:
                    # if MSE_valid_loss < best:
                    torch.save(model, out_path + cfg['modelname'] + '_para.pkl')
                    wait = 0  # release wait
                    best = val_save_acc  # MSE_valid_loss
                    print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
            else:
                # save best model by train loss
                if MSELoss < best:
                    best = MSELoss
                    wait = 0
                    # if epoch <=300:
                    #    torch.save(model,out_path+'/300/'+cfg['modelname']+'_para.pkl')
                    # if epoch >300 and epoch<=500:
                    #     torch.save(model,out_path+'/500/'+cfg['modelname']+'_para.pkl')
                    torch.save(model, out_path + cfg['modelname'] + '_para.pkl')
                    print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                # Early stopping
                if wait >= patience:
                    print("Early stopping triggered.")
                    return

    return'''

'''
import torch
import numpy as np
import time
from tqdm import trange

class DifficultBatchManager:
    def __init__(self, threshold_factor=2, max_difficult_batches=100):
        """
        初始化困难批次管理器。
        :param threshold_factor: 损失阈值的倍数（相对于平均损失）。
        :param max_difficult_batches: 最多记录的困难批次数量。
        """
        self.threshold_factor = threshold_factor
        self.max_difficult_batches = max_difficult_batches
        self.difficult_batches = []

    def record_difficult_batches(self, loss, x_batch, y_batch, aux_batch):
        """
        记录损失过大的批次。
        :param loss: 当前批次的损失值。
        :param x_batch: 输入数据。
        :param y_batch: 目标值。
        :param aux_batch: 辅助数据（如果有）。
        """
        if len(self.difficult_batches) >= self.max_difficult_batches:
            return  # 如果超过最大记录数量，直接返回

        # 将当前批次的数据和损失值记录下来
        self.difficult_batches.append({
            "loss": loss,
            "x_batch": x_batch.cpu().numpy() if isinstance(x_batch, torch.Tensor) else x_batch,
            "y_batch": y_batch.cpu().numpy() if isinstance(y_batch, torch.Tensor) else y_batch,
            "aux_batch": aux_batch.cpu().numpy() if isinstance(aux_batch, torch.Tensor) else aux_batch
        })

    def get_difficult_batches(self, num_samples=None):
        """
        获取记录的困难批次。
        :param num_samples: 采样数量（如果为 None，则返回所有记录的批次）。
        :return: 采样后的困难批次列表。
        """
        if num_samples is None:
            return self.difficult_batches
        else:
            return np.random.choice(self.difficult_batches, size=min(num_samples, len(self.difficult_batches)), replace=False)

    def clear_difficult_batches(self):
        """
        清空记录的困难批次。
        """
        self.difficult_batches = []

    def compute_threshold(self, average_loss):
        """
        计算损失值的阈值。
        :param average_loss: 当前 epoch 的平均损失。
        :return: 阈值。
        """
        return average_loss * self.threshold_factor


# 在训练代码中集成 DifficultBatchManager

def train(x, y, static, mask, scaler_x, scaler_y, cfg, num_repeat, PATH, out_path, device, num_task=None, valid_split=True):
    patience = cfg['patience']
    wait = 0
    best = 9999
    valid_split = cfg['valid_split']  # Use cfg to determine valid_split
    print('The device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static.dtype))

    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
        # Splice x according to the sphere shape
        lat_index, lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))

    if valid_split:
        nt, nf, nlat, nlon = x.shape  # x shape: nt, nf, nlat, nlon
        # Partition validation set and training set
        N = int(nt * cfg['split_ratio'])
        x_valid, y_valid, static_valid = x[N:], y[N:], static
        x, y = x[:N], y[:N]

    lossmse = torch.nn.MSELoss()

    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static.shape)
    print('mask shape is', mask.shape)

    # Apply mask to remove regions (e.g., Antarctica)
    if cfg['modelname'] in ['LSTM']:
        if valid_split:
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['CNN', 'ConvLSTM']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['crossformer']:
        if valid_split:
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)

    # Train and validate
    for num_ in range(cfg['num_repeat']):
        # Initialize model
        if cfg['modelname'] in ['LSTM']:
            lstmmodel_cfg = {
                'input_size': cfg["input_size"],
                'hidden_size': cfg["hidden_size"],
                'out_size': 1,
                #"nhead": 2,  # 多头自注意力机制的头数
                #"num_layers": 3,  # Transformer 编码器的层数

            }
            model = LSTMModel(cfg, lstmmodel_cfg).to(device)
        elif cfg['modelname'] in ['crossformer']:
            model = Crossformer(cfg["input_size"], cfg['seq_len'], 1, 14, 2, 10, 256, 512, 4, 3, 0.2, False, device).to(device)
        elif cfg['modelname'] in ['CNN']:
            model = CNN(cfg).to(device)
        elif cfg['modelname'] in ['ConvLSTM']:
            model = ConvLSTMModel(cfg).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

        # Initialize DifficultBatchManager
        difficult_batch_manager = DifficultBatchManager(threshold_factor=2, max_difficult_batches=100)

        with trange(1, cfg['epochs'] + 1) as pbar:
            for epoch in pbar:
                pbar.set_description(cfg['modelname'] + ' ' + str(num_repeat))
                t_begin = time.time()
                MSELoss = 0
                epoch_loss = 0
                num_batches = 0

                # Normal training batches
                for iter in range(cfg["niter"]):
                    if cfg["modelname"] in ['LSTM']:
                        x_batch, y_batch, aux_batch, _, _, last_end_time_idx = load_train_data_for_rnn(cfg, x, y, static, scaler_y, last_end_time_idx=0)
                    elif cfg['modelname'] in ['crossformer', 'CNN', 'ConvLSTM']:
                        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_cnn(cfg, x, y, static, scaler_y, lat_index, lon_index, mask_index)

                    x_batch = torch.from_numpy(x_batch).to(device)
                    y_batch = torch.from_numpy(y_batch).to(device)
                    aux_batch = torch.from_numpy(aux_batch).to(device)

                    # Adjust input data based on model type
                    if cfg['modelname'] in ['LSTM', 'crossformer']:
                        aux_batch = aux_batch.unsqueeze(1).repeat(1, x_batch.shape[1], 1)
                        x_batch = torch.cat([x_batch, aux_batch], dim=2)
                    elif cfg['modelname'] in ['CNN', 'ConvLSTM']:
                        aux_batch = aux_batch.unsqueeze(1)
                        x_batch = torch.cat([x_batch, aux_batch.expand_as(x_batch)], dim=1)

                    # Forward pass and compute loss
                    pred = model(x_batch, aux_batch)
                    loss = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(), lossmse)

                    # Backward pass and optimization
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    # Accumulate loss
                    MSELoss += loss.item()
                    epoch_loss += loss.item()
                    num_batches += 1

                    # Check if the batch is difficult and record it
                    avg_loss = epoch_loss / num_batches
                    threshold = difficult_batch_manager.compute_threshold(avg_loss)
                    if loss.item() > threshold:
                        difficult_batch_manager.record_difficult_batches(loss.item(), x_batch.cpu().numpy(), y_batch.cpu().numpy(), aux_batch.cpu().numpy())

                # Retrain difficult batches
                difficult_batches = difficult_batch_manager.get_difficult_batches(num_samples=10)
                for batch in difficult_batches:
                    x_batch = torch.from_numpy(batch["x_batch"]).to(device)
                    y_batch = torch.from_numpy(batch["y_batch"]).to(device)
                    aux_batch = torch.from_numpy(batch["aux_batch"]).to(device)

                     # Adjust input data based on model type
                    # 假设 aux_batch 是一个已知的张量
                    aux_size = aux_batch.shape[-1]  # 获取辅助特征的维度
                    cfg["aux_size"] = aux_size  # 动态设置 aux_size

                    # 初始化模型
                    # 更新模型的输入层配置
                    # 数据处理部分保持不变
                    if cfg['modelname'] in ['LSTM', 'crossformer']:
                        # 检查 aux_batch 的形状是否符合要求
                        if aux_batch.dim() == 3 and aux_batch.shape[1] == x_batch.shape[1]:
                            pass  # aux_batch 已经是三维张量，且时间步维度与 x_batch 一致
                        else:
                            aux_batch = aux_batch.unsqueeze(1).repeat(1, x_batch.shape[1], 1)

                        # 打印形状以确认
                        print("x_batch shape:", x_batch.shape)  # 应为 [39, 7, 15]
                        print("aux_batch shape:", aux_batch.shape)  # 应为 [39, 7, 6]
                    elif cfg['modelname'] in ['CNN', 'ConvLSTM']:
                        aux_batch = aux_batch.unsqueeze(1)
                        x_batch = torch.cat([x_batch, aux_batch.expand_as(x_batch)], dim=1)

                    # Forward pass and compute loss
                    pred = model(x_batch, aux_batch)
                    loss = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(), lossmse)

                    # Backward pass and optimization
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Log training progress
                t_end = time.time()
                print("Epoch {} Train MSE Loss {:.3f} Time {:.2f}".format(epoch, MSELoss / cfg["niter"], t_end - t_begin))
                val_save_acc = MSELoss / cfg["niter"]

                # Clear difficult batches every 10 epochs
                if epoch % 10 == 0:
                    difficult_batch_manager.clear_difficult_batches()
                    print(f"Epoch {epoch}: Cleared difficult batches.")

                # Save best model by val loss
                if val_save_acc < best:
                    torch.save(model, out_path + cfg['modelname'] + '_para.pkl')
                    wait = 0  # release wait
                    best = val_save_acc  # MSE_valid_loss
                    print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
            else:
                # Save best model by train loss
                if MSELoss < best:
                    best = MSELoss
                    wait = 0
                    torch.save(model, out_path + cfg['modelname'] + '_para.pkl')
                    print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                # Early stopping
                if wait >= patience:
                    print("Early stopping triggered.")
                    return

    return'''