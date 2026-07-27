"""
机器学习滚动预测框架 - 多因子合成与回测系统
支持线性模型、机器学习、神经网络等多种方法
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import joblib
import time

# ==================== 导入cross_sectional.py中的回测类 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from cross_sectional import BacktestEngine, ResultAnalyzer, Config
    print("✅ 成功从 cross_sectional.py 导入 BacktestEngine, ResultAnalyzer, Config")
except ImportError as e:
    print(f"⚠️ 导入失败: {e}")
    print("请确保 cross_sectional.py 文件存在")
    # 定义本地Config类作为备选
    class Config:
        TOP_N = 200
        MAX_TURNOVER = 0.30
        FEE_RATE = 0.0014
        LOT_SIZE = 100
        LOT_SIZE_688 = 200
        LIMIT_UP_10 = 0.095
        LIMIT_DOWN_10 = -0.095
        LIMIT_UP_20 = 0.195
        LIMIT_DOWN_20 = -0.195
        STOCK_POOL = 'all'
        FACTOR_NAME = 'composite_factor'
        BENCHMARK_STOCK = '000002.SZ'

# ==================== 配置参数 ====================
class ModelConfig:
    """模型配置"""
    # 训练配置
    TRAIN_WINDOW = 252 * 2  # 训练窗口（交易日），默认2年
    RETRAIN_FREQ = 63  # 重训练频率（交易日），默认每季度（约63个交易日）
    LABEL_PERIOD = 1  # 预测周期（交易日）
    
    # 模型选择
    MODEL_TYPE = 'linear'  # 'linear', 'ridge', 'lasso', 'elasticnet', 'rf', 'gbdt', 'svr', 'mlp', 'ensemble'
    
    # 特征选择
    USE_FEATURES = 'all'  # 'all' 或指定特征列表
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 数据增强
    ENABLE_AUGMENTATION = False  # 是否启用数据增强
    
    # 标签处理
    LABEL_CLIP = 0.1  # 标签裁剪阈值（去除极端值）

# ==================== 数据处理器 ====================
class DataProcessor:
    """数据处理器 - 处理因子数据，生成训练和预测数据"""
    
    def __init__(self, data):
        """
        初始化数据处理器
        
        Parameters:
        -----------
        data: DataFrame, 包含所有数据的DataFrame
        """
        self.data = data.copy()
        
        # 识别因子列（排除基础列）
        self.exclude_cols = [
            'trade_date', 'ts_code', 'symbol', 'name', 'area', 'industry', 'market',
            'list_date', 'days_since_listed', 'year', 'month', 'quarter',
            'open', 'high', 'low', 'close', 'vol', 'amount', 'twap',
            'close_adj', 'twap_adj', 'open_adj', 'high_adj', 'low_adj', 'accum_factor',
            'pre_close', 'change', 'pct_chg', 'daily_ret'  # 这些是衍生列，不作为因子
        ]
        
        self.factor_cols = [col for col in data.columns if col not in self.exclude_cols]
        print(f"识别到 {len(self.factor_cols)} 个因子列")
        
        # 选择使用的因子
        if ModelConfig.USE_FEATURES == 'all':
            self.use_cols = self.factor_cols
        else:
            self.use_cols = [col for col in ModelConfig.USE_FEATURES if col in self.factor_cols]
        
        print(f"实际使用 {len(self.use_cols)} 个因子")
        
    def prepare_training_data(self, train_dates):
        """
        准备训练数据
        
        Parameters:
        -----------
        train_dates: list, 训练日期列表
        
        Returns:
        --------
        X: array, 特征矩阵
        y: array, 标签
        stock_list: array, 股票代码列表
        date_list: array, 日期列表
        """
        train_df = self.data[self.data['trade_date'].isin(train_dates)].copy()
        
        X_list = []
        y_list = []
        stock_list = []
        date_list = []
        
        for ts_code, group in train_df.groupby('ts_code'):
            group = group.sort_values('trade_date')
            
            if len(group) < ModelConfig.TRAIN_WINDOW * 0.5:
                continue
            
            # 获取因子值
            X = group[self.use_cols].values
            
            # 计算标签：使用未来第LABEL_PERIOD天的收益率
            # 注意：标签计算使用后复权价格，避免分红影响
            future_returns = group['close_adj'].shift(-ModelConfig.LABEL_PERIOD) / group['close_adj'] - 1
            
            # 标签裁剪（去除极端值）
            if ModelConfig.LABEL_CLIP is not None:
                future_returns = future_returns.clip(-ModelConfig.LABEL_CLIP, ModelConfig.LABEL_CLIP)
            
            y = future_returns.values
            
            # 去除NaN
            valid_idx = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            
            if valid_idx.sum() > 0:
                X_list.append(X[valid_idx])
                y_list.append(y[valid_idx])
                stock_list.extend([ts_code] * valid_idx.sum())
                date_list.extend(group['trade_date'].values[valid_idx])
        
        if not X_list:
            return None, None, None, None
        
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        
        return X_all, y_all, np.array(stock_list), np.array(date_list)
    
    def prepare_prediction_data(self, pred_dates):
        """
        准备预测数据
        
        Parameters:
        -----------
        pred_dates: list, 预测日期列表
        
        Returns:
        --------
        X: array, 特征矩阵
        stock_list: array, 股票代码列表
        date_list: array, 日期列表
        """
        pred_df = self.data[self.data['trade_date'].isin(pred_dates)].copy()
        
        X_list = []
        stock_list = []
        date_list = []
        
        for ts_code, group in pred_df.groupby('ts_code'):
            group = group.sort_values('trade_date')
            
            X = group[self.use_cols].values
            valid_idx = ~np.isnan(X).any(axis=1)
            
            if valid_idx.sum() > 0:
                X_list.append(X[valid_idx])
                stock_list.extend([ts_code] * valid_idx.sum())
                date_list.extend(group['trade_date'].values[valid_idx])
        
        if not X_list:
            return None, None, None
        
        X_all = np.vstack(X_list)
        
        return X_all, np.array(stock_list), np.array(date_list)
    
    def get_factor_columns(self):
        """获取因子列名"""
        return self.use_cols

# ==================== 模型训练器 ====================
class ModelTrainer:
    """模型训练器 - 支持多种模型"""
    
    def __init__(self, model_type='linear'):
        """
        初始化模型训练器
        
        Parameters:
        -----------
        model_type: str, 模型类型
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.training_time = 0
        
    def create_model(self):
        """创建模型"""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            return Lasso(alpha=0.01, max_iter=10000)
        elif self.model_type == 'elasticnet':
            return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
        elif self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=ModelConfig.RANDOM_SEED,
                n_jobs=1  # 单进程
            )
        elif self.model_type == 'gbdt':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=ModelConfig.RANDOM_SEED
            )
        elif self.model_type == 'svr':
            return SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif self.model_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=ModelConfig.RANDOM_SEED,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练模型
        
        Parameters:
        -----------
        X_train: array, 训练特征
        y_train: array, 训练标签
        X_val: array, 验证特征（可选）
        y_val: array, 验证标签（可选）
        
        Returns:
        --------
        metrics: dict, 训练指标
        """
        start_time = time.time()
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 创建并训练模型
        self.model = self.create_model()
        self.model.fit(X_train_scaled, y_train)
        
        self.training_time = time.time() - start_time
        
        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)
        
        # 评估
        train_pred = self.model.predict(X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        
        # IC（预测值与实际值的秩相关系数）
        train_ic = spearmanr(train_pred, y_train)[0]
        
        print(f"  训练集 - RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R2: {train_r2:.4f}, IC: {train_ic:.4f}")
        
        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'train_ic': train_ic,
            'train_size': len(X_train),
            'train_time': self.training_time
        }
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_ic = spearmanr(val_pred, y_val)[0]
            
            print(f"  验证集 - RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R2: {val_r2:.4f}, IC: {val_ic:.4f}")
            
            metrics.update({
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'val_ic': val_ic,
                'val_size': len(X_val)
            })
        
        return metrics
    
    def predict(self, X):
        """预测"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names=None):
        """获取特征重要性"""
        if self.feature_importance is not None and feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            return importance_df
        return self.feature_importance

# ==================== 滚动预测框架 ====================
class RollingPredictor:
    """滚动预测器 - 单进程版本"""
    
    def __init__(self, data, model_type='linear'):
        """
        初始化滚动预测器
        
        Parameters:
        -----------
        data: DataFrame, 数据
        model_type: str, 模型类型
        """
        self.data = data
        self.model_type = model_type
        
        # 初始化数据处理器
        self.processor = DataProcessor(data)
        
        # 存储结果
        self.predictions = pd.DataFrame()
        self.models = {}
        self.model_results = []
        self.feature_importance_history = []
        
    def rolling_train_predict(self):
        """
        滚动训练和预测
        
        Returns:
        --------
        predictions: DataFrame, 预测结果
        """
        print(f"\n{'='*60}")
        print(f"开始滚动预测 - 模型: {self.model_type}")
        print(f"{'='*60}")
        print(f"训练窗口: {ModelConfig.TRAIN_WINDOW} 天")
        print(f"重训练频率: {ModelConfig.RETRAIN_FREQ} 天")
        print(f"预测周期: {ModelConfig.LABEL_PERIOD} 天")
        print(f"使用因子: {len(self.processor.get_factor_columns())} 个")
        
        # 获取所有日期
        all_dates = sorted(self.data['trade_date'].unique())
        print(f"总交易日: {len(all_dates)}")
        
        # 检查是否有足够的数据
        min_days = ModelConfig.TRAIN_WINDOW + 10
        if len(all_dates) < min_days:
            print(f"⚠️ 数据不足: 需要至少 {min_days} 个交易日，当前只有 {len(all_dates)} 个")
            return pd.DataFrame()
        
        # 初始化
        pred_results = []
        total_predictions = 0
        
        # 滚动训练
        train_start_idx = ModelConfig.TRAIN_WINDOW
        total_iterations = len(all_dates) - ModelConfig.LABEL_PERIOD - train_start_idx
        
        for i in range(train_start_idx, len(all_dates) - ModelConfig.LABEL_PERIOD, ModelConfig.RETRAIN_FREQ):
            # 训练日期
            train_dates = all_dates[i - ModelConfig.TRAIN_WINDOW:i]
            
            # 预测日期
            pred_date = all_dates[i]
            
            print(f"\n[{i - train_start_idx + 1}/{total_iterations}] 训练窗口: {train_dates[0].strftime('%Y-%m-%d')} -> {train_dates[-1].strftime('%Y-%m-%d')}")
            print(f"  预测日期: {pred_date.strftime('%Y-%m-%d')}")
            
            # 准备训练数据
            X_train, y_train, _, _ = self.processor.prepare_training_data(train_dates)
            
            if X_train is None or len(X_train) < 50:
                print(f"  ⚠️ 训练数据不足 (样本数: {0 if X_train is None else len(X_train)})，跳过")
                continue
            
            # 划分训练集和验证集
            if len(X_train) > 200:
                split_idx = int(len(X_train) * 0.8)
                X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
                y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
            else:
                X_tr, X_val = X_train, None
                y_tr, y_val = y_train, None
            
            # 训练模型
            trainer = ModelTrainer(self.model_type)
            metrics = trainer.train(X_tr, y_tr, X_val, y_val)
            
            # 保存模型
            self.models[pred_date] = trainer
            
            # 记录特征重要性
            if trainer.feature_importance is not None:
                importance_df = trainer.get_feature_importance(self.processor.get_factor_columns())
                importance_df['pred_date'] = pred_date
                self.feature_importance_history.append(importance_df)
            
            # 准备预测数据
            X_pred, stock_list, date_list = self.processor.prepare_prediction_data([pred_date])
            
            if X_pred is None or len(X_pred) == 0:
                print(f"  ⚠️ 无预测数据，跳过")
                continue
            
            # 预测
            y_pred = trainer.predict(X_pred)
            
            # 保存预测结果
            for j, (stock, date, pred) in enumerate(zip(stock_list, date_list, y_pred)):
                pred_results.append({
                    'trade_date': date,
                    'ts_code': stock,
                    'pred_score': pred,
                    'model_type': self.model_type
                })
            
            total_predictions += len(X_pred)
            
            # 记录模型结果
            self.model_results.append({
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'pred_date': pred_date,
                'train_size': len(X_train),
                'predictions': len(X_pred),
                'metrics': metrics
            })
            
            print(f"  ✅ 预测完成: {len(X_pred)} 个样本, 累计: {total_predictions}")
        
        # 保存预测结果
        self.predictions = pd.DataFrame(pred_results)
        
        print(f"\n{'='*60}")
        print(f"滚动预测完成!")
        print(f"总预测样本: {total_predictions}")
        print(f"模型数量: {len(self.models)}")
        print(f"{'='*60}")
        
        if self.predictions.empty:
            print("⚠️ 没有生成任何预测结果")
        else:
            print(f"预测结果形状: {self.predictions.shape}")
            print(f"预测日期范围: {self.predictions['trade_date'].min()} 到 {self.predictions['trade_date'].max()}")
        
        return self.predictions

# ==================== 因子合成器 ====================
class FactorCombiner:
    """因子合成器 - 将模型预测结果合成为复合因子"""
    
    def __init__(self, data, predictions):
        """
        初始化因子合成器
        
        Parameters:
        -----------
        data: DataFrame, 原始数据
        predictions: DataFrame, 预测结果
        """
        self.data = data
        self.predictions = predictions
        
    def combine_factors(self):
        """
        合成因子
        
        Returns:
        --------
        combined: DataFrame, 包含复合因子的数据
        """
        print(f"\n{'='*60}")
        print("合成复合因子")
        print(f"{'='*60}")
        
        # 合并预测结果到原始数据
        combined = self.data.merge(
            self.predictions[['trade_date', 'ts_code', 'pred_score']],
            on=['trade_date', 'ts_code'],
            how='left'
        )
        
        # 填充缺失值
        # 1. 按股票分组，使用前向填充
        combined['pred_score'] = combined.groupby('ts_code')['pred_score'].ffill()
        
        # 2. 如果还有缺失，使用滚动均值填充
        combined['pred_score'] = combined.groupby('ts_code')['pred_score'].transform(
            lambda x: x.fillna(x.rolling(20, min_periods=1).mean())
        )
        
        # 3. 如果还有缺失，使用整体均值填充
        global_mean = combined['pred_score'].mean()
        combined['pred_score'] = combined['pred_score'].fillna(global_mean)
        
        # 标准化合成因子（横截面标准化）
        def cross_section_standardize(group):
            """横截面标准化"""
            mean = group.mean()
            std = group.std()
            if std > 0 and not np.isnan(std):
                return (group - mean) / std
            return group
        
        combined['composite_factor'] = combined.groupby('trade_date')['pred_score'].transform(
            cross_section_standardize
        )
        
        # 处理标准化后可能出现的NaN
        combined['composite_factor'] = combined['composite_factor'].fillna(0)
        
        print(f"合成完成!")
        print(f"数据形状: {combined.shape}")
        print(f"复合因子统计:")
        print(f"  均值: {combined['composite_factor'].mean():.6f}")
        print(f"  标准差: {combined['composite_factor'].std():.6f}")
        print(f"  最小值: {combined['composite_factor'].min():.6f}")
        print(f"  最大值: {combined['composite_factor'].max():.6f}")
        print(f"  缺失值: {combined['composite_factor'].isnull().sum()}")
        
        return combined

# ==================== 复合因子回测引擎 ====================
class CompositeFactorBacktest:
    """复合因子回测引擎 - 使用cross_sectional.py中的BacktestEngine"""
    
    def __init__(self, data, factor_name='composite_factor'):
        """
        初始化回测引擎
        
        Parameters:
        -----------
        data: DataFrame, 包含复合因子的数据
        factor_name: str, 复合因子列名
        """
        self.data = data
        self.factor_name = factor_name
        self.results = {}
        self.engine = None
        self.analyzer = None
        self.metrics = None
        
    def prepare_data_for_engine(self, config):
        """
        准备BacktestEngine需要的数据格式
        
        Parameters:
        -----------
        config: Config对象, 回测配置
        
        Returns:
        --------
        data_dict: dict, BacktestEngine所需的数据字典
        """
        backtest_data = self.data.copy()
        
        # 获取股票信息
        stock_info = backtest_data[['ts_code', 'symbol', 'name', 'industry']].drop_duplicates('ts_code')
        
        # 获取基准数据
        benchmark_stock = getattr(config, 'BENCHMARK_STOCK', stock_info['ts_code'].iloc[0])
        benchmark_df = backtest_data[backtest_data['ts_code'] == benchmark_stock].copy()
        if benchmark_df.empty:
            benchmark_stock = stock_info['ts_code'].iloc[0]
            benchmark_df = backtest_data[backtest_data['ts_code'] == benchmark_stock].copy()
        
        # 获取价格数据
        price_data = backtest_data.pivot(index='trade_date', columns='ts_code', values='close_adj')
        twap_data = backtest_data.pivot(index='trade_date', columns='ts_code', values='twap_adj')
        accum_factor = backtest_data.pivot(index='trade_date', columns='ts_code', values='accum_factor')
        
        # 构建因子数据字典（包含复合因子）
        factor_data = {
            self.factor_name: backtest_data.pivot(index='trade_date', columns='ts_code', values=self.factor_name)
        }
        
        # 获取ST标识
        st_data = backtest_data[['trade_date', 'ts_code', 'name']].copy()
        st_data['is_st'] = st_data['name'].str.contains('ST|\\*ST', case=False, na=False)
        st_pivot = st_data.pivot(index='trade_date', columns='ts_code', values='is_st').fillna(False)
        
        return {
            'df': backtest_data,
            'stock_info': stock_info,
            'benchmark': benchmark_df,
            'benchmark_stock': benchmark_stock,
            'price': price_data,
            'twap': twap_data,
            'accum_factor': accum_factor,
            'factors': factor_data,
            'st': st_pivot,
            'factor_cols': [self.factor_name]
        }
    
    def run_backtest(self, config):
        """
        运行回测
        
        Parameters:
        -----------
        config: Config对象, 回测配置
        
        Returns:
        --------
        results: dict, 回测结果
        """
        print(f"\n{'='*60}")
        print(f"使用复合因子进行回测: {self.factor_name}")
        print(f"{'='*60}")
        
        # 准备数据
        data_for_engine = self.prepare_data_for_engine(config)
        
        # 使用cross_sectional.py中的BacktestEngine
        self.engine = BacktestEngine(data_for_engine, config)
        
        # 运行回测
        self.results = self.engine.run(factor_name=self.factor_name)
        
        # 分析结果
        self.analyzer = ResultAnalyzer(
            self.engine,
            data_for_engine['benchmark'],
            data_for_engine['benchmark_stock']
        )
        self.metrics = self.analyzer.analyze()
        
        return self.results
    
    def get_performance_metrics(self):
        """获取绩效指标"""
        return self.metrics
    
    def plot_results(self, save_path=None):
        """绘制结果图"""
        if self.analyzer is not None:
            return self.analyzer.plot_results(save_path)
        return None
    
    def save_results(self, output_dir):
        """保存结果"""
        if self.analyzer is not None:
            return self.analyzer.save_results(output_dir)
        return None
    
    def get_trade_records(self):
        """获取交易记录"""
        if self.engine is not None:
            return pd.DataFrame(self.engine.trade_records)
        return None
    
    def get_daily_holdings(self):
        """获取每日持仓"""
        if self.engine is not None:
            return pd.DataFrame(self.engine.daily_holdings)
        return None
    
    def get_daily_nav(self):
        """获取每日净值"""
        if self.engine is not None:
            nav_df = pd.DataFrame({
                'date': list(self.engine.portfolio_value.keys()),
                'total_value': list(self.engine.portfolio_value.values()),
                'cash': list(self.engine.cash.values())
            })
            return nav_df
        return None

# ==================== 结果对比分析器 ====================
class ResultComparator:
    """结果对比分析器"""
    
    def __init__(self, results_dict):
        self.results = results_dict
        
    def compare(self):
        """对比不同模型的结果"""
        print(f"\n{'='*60}")
        print("模型对比分析")
        print(f"{'='*60}")
        
        comparison = []
        
        for model_name, backtest in self.results.items():
            if backtest is None:
                continue
            
            metrics = backtest.get_performance_metrics()
            if metrics is None:
                continue
            
            # 提取关键指标
            portfolio_metrics = metrics.get('portfolio_metrics', {})
            
            comparison.append({
                'model': model_name,
                'Total Return': portfolio_metrics.get('Total Return', 0),
                'Annual Return': portfolio_metrics.get('Annual Return', 0),
                'Sharpe Ratio': portfolio_metrics.get('Sharpe Ratio', 0),
                'Max Drawdown': portfolio_metrics.get('Max Drawdown', 0),
                'Win Rate': portfolio_metrics.get('Win Rate', 0),
                'Trading Days': portfolio_metrics.get('Trading Days', 0)
            })
        
        if not comparison:
            print("没有可对比的结果")
            return None
        
        df = pd.DataFrame(comparison)
        df.set_index('model', inplace=True)
        
        print("\n模型性能对比:")
        print(df.round(4))
        
        return df

# ==================== 主程序 ====================
def main():
    """主程序"""
    print("="*80)
    print("机器学习滚动预测与多因子合成回测系统")
    print("="*80)
    
    
    
    # 获取当前工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前工作目录: {current_dir}")
    
    # 构建数据路径
    DATA_PATH = os.path.join(current_dir, '..', 'final_data')
    if not os.path.exists(DATA_PATH):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(script_dir, '..', 'final_data')
    
    DATA_PATH = os.path.normpath(DATA_PATH)
    OUTPUT_PATH = os.path.join(current_dir, 'rolling_results')
    OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)
    
    print(f"数据路径: {DATA_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    
    # 检查数据路径
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 数据路径不存在 - {DATA_PATH}")
        return
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    data_file = os.path.join(DATA_PATH, "final_200stocks_with_benchmark_2020_2025.csv")
    if not os.path.exists(data_file):
        data_file = os.path.join(DATA_PATH, "final_200stocks_with_benchmark_2020_2025.parquet")
    
    if os.path.exists(data_file):
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        else:
            df = pd.read_parquet(data_file)
        print(f"✅ 数据加载成功，形状: {df.shape}")
    else:
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    print(f"日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
    print(f"股票数量: {df['ts_code'].nunique()}")
    
    # 定义要测试的模型（单进程模式）
    models_to_test = ['linear', 'ridge', 'rf', 'gbdt']
    
    print(f"\n将测试的模型: {models_to_test}")
    
    all_backtests = {}
    
    # 对每个模型进行训练和回测
    for model_name in models_to_test:
        print(f"\n{'#'*60}")
        print(f"测试模型: {model_name}")
        print(f"{'#'*60}")
        
        try:
            # 1. 滚动预测
            print(f"\n[1/3] 开始 {model_name} 滚动预测...")
            predictor = RollingPredictor(df, model_type=model_name)
            predictions = predictor.rolling_train_predict()
            
            if predictions.empty:
                print(f"⚠️ {model_name} 预测结果为空，跳过")
                continue
            
            print(f"✅ 预测完成，预测结果数量: {len(predictions)}")
            
            # 2. 合成因子
            print(f"\n[2/3] 合成复合因子...")
            combiner = FactorCombiner(df, predictions)
            combined_data = combiner.combine_factors()
            
            # 3. 运行回测
            print(f"\n[3/3] 运行回测...")
            output_dir = os.path.join(OUTPUT_PATH, model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # 配置回测
            config = Config()
            config.TOP_N = 200
            config.MAX_TURNOVER = 0.30
            config.FEE_RATE = 0.0014
            config.FACTOR_NAME = 'composite_factor'
            config.BENCHMARK_STOCK = '000002.SZ'
            
            # 创建回测实例
            backtest = CompositeFactorBacktest(combined_data, 'composite_factor')
            
            # 运行回测
            backtest.run_backtest(config)
            
            # 保存结果
            backtest.save_results(output_dir)
            
            # 保存交易记录
            trade_records = backtest.get_trade_records()
            if trade_records is not None and not trade_records.empty:
                trade_records.to_csv(os.path.join(output_dir, 'trade_records.csv'), index=False)
                print(f"  交易记录已保存: {len(trade_records)} 条")
            
            # 保存每日持仓
            daily_holdings = backtest.get_daily_holdings()
            if daily_holdings is not None and not daily_holdings.empty:
                daily_holdings.to_csv(os.path.join(output_dir, 'daily_holdings.csv'), index=False)
            
            # 保存每日净值
            daily_nav = backtest.get_daily_nav()
            if daily_nav is not None and not daily_nav.empty:
                daily_nav.to_csv(os.path.join(output_dir, 'daily_nav.csv'), index=False)
            
            # 绘制图表
            plot_path = os.path.join(output_dir, 'backtest_results.png')
            backtest.plot_results(save_path=plot_path)
            
            # 保存结果
            all_backtests[model_name] = backtest
            
            print(f"✅ {model_name} 完成!")
            
        except Exception as e:
            print(f"❌ 模型 {model_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 对比分析
    if all_backtests:
        print(f"\n{'='*60}")
        print("模型对比分析")
        print(f"{'='*60}")
        
        comparator = ResultComparator(all_backtests)
        comparison_df = comparator.compare()
        
        if comparison_df is not None and not comparison_df.empty:
            comparison_df.to_csv(os.path.join(OUTPUT_PATH, 'model_comparison.csv'))
            print(f"\n✅ 对比结果已保存: {os.path.join(OUTPUT_PATH, 'model_comparison.csv')}")
            
            # 打印最佳模型
            if 'Sharpe Ratio' in comparison_df.columns:
                best_model = comparison_df['Sharpe Ratio'].idxmax()
                print(f"\n🏆 最佳模型: {best_model}")
                print(f"   夏普比率: {comparison_df.loc[best_model, 'Sharpe Ratio']:.4f}")
                print(f"   年化收益: {comparison_df.loc[best_model, 'Annual Return']:.4f}")
    
    print(f"\n{'='*80}")
    print("所有模型测试完成！")
    print(f"结果保存在: {OUTPUT_PATH}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()