# mechanism.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelMechanismAnalyzer:
    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册钩子来捕获中间层激活"""
        if self.cfg['modelname'] in ['LSTM']:
            # 为LSTM注册钩子
            def lstm_hook(module, input, output):
                self.activations['lstm_output'] = output[0].detach().cpu().numpy()
                self.activations['lstm_hidden'] = output[1][0].detach().cpu().numpy()
                self.activations['lstm_cell'] = output[1][1].detach().cpu().numpy()
            
            self.hooks.append(self.model.lstm.register_forward_hook(lstm_hook))
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_feature_importance(self, x_test, static, mask, num_samples=10):
        """分析特征重要性通过排列重要性方法"""
        print("[机制分析] 开始特征重要性分析...")
        
        # 选择陆地网格点
        land_indices = np.where(mask == 1)
        land_lat_indices = land_indices[0]
        land_lon_indices = land_indices[1]
        
        # 随机选择一些陆地网格点
        selected_indices = np.random.choice(len(land_lat_indices), 
                                          min(num_samples, len(land_lat_indices)), 
                                          replace=False)
        
        feature_importance = {}
        n_features = x_test.shape[-1]  # 动态特征数量
        
        # 对每个特征进行分析
        for feature_idx in range(n_features):
            total_importance = 0
            valid_count = 0
            
            for idx in selected_indices:
                lat_idx = land_lat_indices[idx]
                lon_idx = land_lon_indices[idx]
                
                try:
                    # 获取该网格点的基准预测
                    baseline_pred = self._predict_single_location(
                        x_test, static, lat_idx, lon_idx
                    )
                    
                    # 创建扰动数据
                    x_perturbed = x_test.copy()
                    original_data = x_perturbed[:, lat_idx, lon_idx, feature_idx].copy()
                    np.random.shuffle(x_perturbed[:, lat_idx, lon_idx, feature_idx])
                    
                    # 获取扰动后的预测
                    perturbed_pred = self._predict_single_location(
                        x_perturbed, static, lat_idx, lon_idx
                    )
                    
                    # 计算重要性
                    importance = self._calculate_importance(baseline_pred, perturbed_pred)
                    total_importance += importance
                    valid_count += 1
                    
                except Exception as e:
                    continue
            
            if valid_count > 0:
                avg_importance = total_importance / valid_count
                feature_importance[feature_idx] = avg_importance
                print(f"动态特征 {feature_idx}: 平均重要性得分 = {avg_importance:.4f} (基于{valid_count}个网格点)")
        
        return feature_importance
    
    def analyze_activation_patterns(self, x_batch, static, mask):
        """分析激活模式"""
        print("[机制分析] 开始激活模式分析...")
        
        self.activations.clear()
        self.register_hooks()
        
        # 选择一些陆地网格点
        land_indices = np.where(mask == 1)
        land_lat_indices = land_indices[0]
        land_lon_indices = land_indices[1]
        
        selected_indices = np.random.choice(len(land_lat_indices), 
                                          min(5, len(land_lat_indices)), 
                                          replace=False)
        
        all_activations = {}
        
        for idx in selected_indices:
            lat_idx = land_lat_indices[idx]
            lon_idx = land_lon_indices[idx]
            
            try:
                # 获取该位置的数据
                x_location = x_batch[:, lat_idx, lon_idx, :]  # [时间步, 特征]
                static_location = static[lat_idx, lon_idx, :]  # [静态特征]
                
                # 转换为模型输入格式
                x_tensor, static_tensor = self._prepare_lstm_input(x_location, static_location)
                
                # 前向传播
                with torch.no_grad():
                    _ = self.model(x_tensor, static_tensor)
                
                # 收集激活
                for layer_name, activation in self.activations.items():
                    if layer_name not in all_activations:
                        all_activations[layer_name] = []
                    all_activations[layer_name].append(activation)
                    
            except Exception as e:
                print(f"网格点({lat_idx},{lon_idx})激活分析失败: {e}")
                continue
        
        # 分析激活统计
        activation_stats = {}
        for layer_name, activations_list in all_activations.items():
            if activations_list:
                # 合并所有激活
                all_acts = np.concatenate([act.reshape(-1) for act in activations_list])
                
                activation_stats[layer_name] = {
                    'mean': np.mean(all_acts),
                    'std': np.std(all_acts),
                    'max': np.max(all_acts),
                    'min': np.min(all_acts),
                    'sparsity': np.mean(all_acts == 0)
                }
                print(f"{layer_name}: 均值={activation_stats[layer_name]['mean']:.4f}, "
                      f"标准差={activation_stats[layer_name]['std']:.4f}, "
                      f"稀疏性={activation_stats[layer_name]['sparsity']:.4f}")
        
        self.remove_hooks()
        return activation_stats
    
    def temporal_dependency_analysis(self, x_test, static, mask, seq_len=30):
        """分析时间依赖性"""
        print("[机制分析] 开始时间依赖性分析...")
        
        # 选择一些陆地网格点
        land_indices = np.where(mask == 1)
        land_lat_indices = land_indices[0]
        land_lon_indices = land_indices[1]
        
        selected_indices = np.random.choice(len(land_lat_indices), 
                                          min(10, len(land_lat_indices)), 
                                          replace=False)
        
        temporal_stabilities = []
        
        for idx in selected_indices:
            lat_idx = land_lat_indices[idx]
            lon_idx = land_lon_indices[idx]
            
            try:
                # 获取该位置的数据
                x_location = x_test[:, lat_idx, lon_idx, :]  # [时间步, 特征]
                static_location = static[lat_idx, lon_idx, :]  # [静态特征]
                
                predictions = []
                
                # 逐步增加输入序列长度
                for t in range(self.cfg['seq_len'], min(seq_len, len(x_location))):
                    x_partial = x_location[:t]  # 使用前t个时间步
                    
                    # 转换为模型输入格式
                    x_tensor, static_tensor = self._prepare_lstm_input(x_partial, static_location)
                    
                    with torch.no_grad():
                        pred = self.model(x_tensor, static_tensor)
                        predictions.append(pred.cpu().numpy()[0, 0])
                
                # 计算预测稳定性
                if len(predictions) > 1:
                    stability = np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8)
                    temporal_stabilities.append(stability)
                    
            except Exception as e:
                continue
        
        if temporal_stabilities:
            avg_stability = np.mean(temporal_stabilities)
            print(f"平均时间稳定性: {avg_stability:.4f} (基于{len(temporal_stabilities)}个网格点)")
            return avg_stability
        else:
            print("无法计算时间稳定性")
            return 0
    
    def _predict_single_location(self, x_test, static, lat_idx, lon_idx):
        """预测单个位置"""
        # 获取该位置的数据
        x_location = x_test[:, lat_idx, lon_idx, :]  # [时间步, 特征]
        static_location = static[lat_idx, lon_idx, :]  # [静态特征]
        
        # 转换为模型输入格式
        x_tensor, static_tensor = self._prepare_lstm_input(x_location, static_location)
        
        with torch.no_grad():
            prediction = self.model(x_tensor, static_tensor)
            return prediction.cpu().numpy()[0, 0]
    
    def _prepare_lstm_input(self, x_location, static_location):
        """准备LSTM模型输入"""
        # x_location: [时间步, 动态特征]
        # static_location: [静态特征]
        
        # 转换为张量
        x_tensor = torch.from_numpy(x_location).float().to(self.device).unsqueeze(0)  # [1, 时间步, 动态特征]
        static_tensor = torch.from_numpy(static_location).float().to(self.device)  # [静态特征]
        
        # 重复静态特征以匹配时间步
        static_tensor = static_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 静态特征]
        static_tensor = static_tensor.repeat(1, x_tensor.shape[1], 1)  # [1, 时间步, 静态特征]
        
        # 拼接动态和静态特征
        x_combined = torch.cat([x_tensor, static_tensor], dim=-1)  # [1, 时间步, 动态特征+静态特征]
        
        return x_combined, static_tensor
    
    def _calculate_importance(self, baseline_pred, perturbed_pred):
        """计算特征重要性"""
        # 使用预测变化的绝对值作为重要性指标
        return abs(baseline_pred - perturbed_pred)

def run_mechanism(model, x_test, static, scaler_y, cfg, device):
    """运行模型机制分析的主函数"""
    print("=" * 60)
    print("开始模型机制分析")
    print("=" * 60)
    
    # 加载mask
    path = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(path + file_name_mask)
    
    print(f"数据形状: x_test {x_test.shape}, static {static.shape}, mask {mask.shape}")
    
    analyzer = ModelMechanismAnalyzer(model, cfg, device)
    results = {}
    
    try:
        # 1. 特征重要性分析
        print("\n1. 特征重要性分析")
        feature_importance = analyzer.analyze_feature_importance(
            x_test, static, mask, num_samples=20
        )
        results['feature_importance'] = feature_importance
        
        # 2. 激活模式分析
        print("\n2. 激活模式分析")
        activation_stats = analyzer.analyze_activation_patterns(
            x_test, static, mask
        )
        results['activation_stats'] = activation_stats
        
        # 3. 时间依赖性分析
        print("\n3. 时间依赖性分析")
        temporal_stability = analyzer.temporal_dependency_analysis(
            x_test, static, mask, seq_len=min(50, x_test.shape[0])
        )
        results['temporal_stability'] = temporal_stability
        
        # 生成分析报告
        _generate_analysis_report(results, cfg, x_test)
        
    except Exception as e:
        print(f"机制分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("模型机制分析完成")
    print("=" * 60)
    
    return results

def _generate_analysis_report(results, cfg, x_test):
    """生成分析报告"""
    print("\n" + "=" * 40)
    print("模型机制分析报告")
    print("=" * 40)
    
    print(f"模型类型: {cfg['modelname']}")
    print(f"序列长度: {cfg['seq_len']}")
    print(f"预测时间: {cfg['forcast_time']}")
    print(f"动态特征数量: {x_test.shape[-1]}")
    
    if 'feature_importance' in results and results['feature_importance']:
        print("\n--- 特征重要性排名 ---")
        importance_sorted = sorted(results['feature_importance'].items(), 
                                 key=lambda x: abs(x[1]), reverse=True)
        
        # 特征名称映射（根据您的数据调整）
        feature_names = {
            0: "2m_temperature",
            1: "10m_u_wind", 
            2: "10m_v_wind",
            3: "precipitation",
            4: "surface_pressure",
            5: "specific_humidity",
            6: "solar_radiation",
            7: "thermal_radiation", 
            8: "soil_temperature"
        }
        
        for idx, (feature_idx, importance) in enumerate(importance_sorted):
            feature_name = feature_names.get(feature_idx, f"特征{feature_idx}")
            print(f"第{idx+1}名: {feature_name}, 重要性: {importance:.4f}")
    else:
        print("\n--- 特征重要性分析不可用 ---")
    
    if 'activation_stats' in results and results['activation_stats']:
        print("\n--- 激活统计 ---")
        for layer, stats in results['activation_stats'].items():
            print(f"{layer}: 稀疏性 = {stats['sparsity']:.3f}")
    else:
        print("\n--- 激活统计不可用 ---")
    
    if 'temporal_stability' in results:
        stability = results['temporal_stability']
        print(f"\n时间稳定性: {stability:.4f}")
        if stability < 0.1:
            print("模型时间稳定性: 优秀")
        elif stability < 0.3:
            print("模型时间稳定性: 良好")
        else:
            print("模型时间稳定性: 需要改进")
    else:
        print("\n--- 时间依赖性分析不可用 ---")