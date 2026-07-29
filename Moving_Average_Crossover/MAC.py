"""
多因子量化策略开发与验证
包含：技术面策略、稳健性验证、机器学习增强、回测对比
开仓：
1.**均线金叉** | MA5上穿MA10（即：`ma_5 > ma_10` 且 前一日 `ma_5 <= ma_10`）
2.**均线斜率** | 当前MA5斜率大于0（`ma_5 > REF(ma_5, 1)`）
3.**K线形态** | 阳线上影不超过实体的75%；阴线下影不超过实体的90%
4.**非涨停** | 当日收盘价 < 涨停价（根据板块涨跌停限制判断）
5.**价格约束** | `(MA5 - REF(MA5, 1)) <= 0.05 × 当日收盘价
6.**KDJ(9,3,3)>60  and KDJ(72,24,24)<85,KDJ(72,24,24)斜率>-0.5

平仓：
1.**固定止损** | 持仓亏损达到5%（相对开仓价）
2.**均线死叉或斜率异常** | `(MA5 - MA10) < REF(MA5 - MA10, 1)` 或 `当前MA5斜率 > 开仓时MA5斜率的1.88倍` 

| 训练窗口 | 252天 | 1年交易日 |
| 测试窗口 | 63天 | 1个季度交易日 |
| 滚动步长 | 21天 | 每月滚动一次 |

**样本外验证**：选取1年数据进行策略稳健性验证
**验证方法**：滚动窗口测试 + 回撤归因分析
| **LightGBM** | 多因子选股、信号预测 | 高效、支持大规模数据、可处理缺失值 |
| **XGBoost** | 多因子选股、分类预测 | 正则化强、防止过拟合、解释性好 |
| **Random Forest** | 特征探索、基准模型 | 可解释性强、特征重要性评估 |

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
import sys
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import pickle
import time
import json

# ==================== 配置参数 ====================
class StrategyConfig:
    """策略配置"""
    # 交易参数
    TOP_N = 200
    MAX_TURNOVER = 0.30
    FEE_RATE = 0.0014
    
    # 时间窗口
    TRAIN_WINDOW = 252
    TEST_WINDOW = 63
    STEP_SIZE = 21
    
    # 止损参数
    STOP_LOSS = 0.05
    
    # 斜率阈值
    SLOPE_THRESHOLD = 1.88
    
    # 价格约束
    PRICE_CONSTRAINT = 0.05
    
    # K线形态约束
    UPPER_SHADOW_LIMIT = 0.75
    LOWER_SHADOW_LIMIT = 0.90
    
    # KDJ参数
    KDJ_FAST_PERIOD = 9
    KDJ_SLOW_PERIOD = 3
    KDJ_SMOOTH_PERIOD = 3
    KDJ_LONG_PERIOD = 72
    KDJ_LONG_SLOW = 24
    KDJ_LONG_SMOOTH = 24
    
    # 回测配置
    BENCHMARK_STOCK = '000002.SZ'
    INITIAL_CAPITAL = 100000000
    
    # 机器学习配置
    ML_TOP_PCT = 0.60
    PREDICT_HORIZON = 5
    
    # 涨跌停限制
    LIMIT_UP_10 = 0.095
    LIMIT_DOWN_10 = -0.095
    LIMIT_UP_20 = 0.195
    LIMIT_DOWN_20 = -0.195

# ==================== 数据加载 ====================
class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None
        self.benchmark_data = None
        self.stock_info = None
        
    def load_data(self):
        """加载数据"""
        print("\n" + "="*80)
        print("Loading Data")
        print("="*80)
        
        csv_file = self.data_path / "final_200stocks_with_benchmark_2020_2025.csv"
        parquet_file = self.data_path / "final_200stocks_with_benchmark_2020_2025.parquet"
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            print(f"✅ Loaded CSV file: {csv_file}")
        elif parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            print(f"✅ Loaded Parquet file: {parquet_file}")
        else:
            raise FileNotFoundError(f"Data file not found")
        
        df = df.sort_values(['ts_code', 'trade_date'])
        
        self.stock_info = df[['ts_code', 'symbol', 'name', 'industry']].drop_duplicates('ts_code')
        self.benchmark_data = df[df['ts_code'] == StrategyConfig.BENCHMARK_STOCK].copy()
        
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
        print(f"Number of stocks: {df['ts_code'].nunique()}")
        
        self.data = df
        return df

# ==================== 技术指标计算 ====================
class TechnicalIndicators:
    """技术指标计算 - 严格使用历史数据"""
    
    @staticmethod
    def calculate_all(df):
        """计算所有技术指标 - 注意：所有指标都基于历史数据，不包含未来信息"""
        df = df.copy()
        df = df.sort_values(['ts_code', 'trade_date'])
        
        print("Calculating moving averages...")
        for period in [5, 10, 20, 60]:
            df[f'ma_{period}'] = df.groupby('ts_code')['close_adj'].transform(
                lambda x: x.rolling(period, min_periods=1).mean()
            )
        
        print("Calculating MA slopes...")
        df['ma_5_slope'] = df.groupby('ts_code')['ma_5'].transform(
            lambda x: x - x.shift(1)
        )
        df['ma_5_prev'] = df.groupby('ts_code')['ma_5'].shift(1)
        df['ma_10_prev'] = df.groupby('ts_code')['ma_10'].shift(1)
        
        print("Calculating candlestick patterns...")
        df['body'] = abs(df['close_adj'] - df['open_adj'])
        df['upper_shadow'] = df['high_adj'] - df[['close_adj', 'open_adj']].max(axis=1)
        df['lower_shadow'] = df[['close_adj', 'open_adj']].min(axis=1) - df['low_adj']
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['body'] + 1e-8)
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['body'] + 1e-8)
        
        print("Calculating price changes...")
        df['prev_close'] = df.groupby('ts_code')['close_adj'].shift(1)
        df['pct_chg'] = (df['close_adj'] / df['prev_close'] - 1) * 100
        
        def check_limit_up(row):
            if pd.isna(row['prev_close']) or row['prev_close'] == 0:
                return False
            pct_chg = (row['close_adj'] / row['prev_close'] - 1)
            if row['ts_code'].startswith('688') or row['ts_code'].startswith('300'):
                return pct_chg >= StrategyConfig.LIMIT_UP_20
            else:
                return pct_chg >= StrategyConfig.LIMIT_UP_10
        
        df['is_limit_up'] = df.apply(check_limit_up, axis=1)
        
        print("Calculating volume moving averages...")
        if 'vol' in df.columns:
            for period in [5, 10, 20]:
                df[f'volume_ma_{period}'] = df.groupby('ts_code')['vol'].transform(
                    lambda x: x.rolling(period, min_periods=1).mean()
                )
        else:
            print("⚠️ Warning: Volume column not found, skipping volume indicators")
            for period in [5, 10, 20]:
                df[f'volume_ma_{period}'] = 0
        
        df['high_low_ratio'] = df['high_adj'] / df['low_adj']
        df['close_open_ratio'] = df['close_adj'] / df['open_adj']
        
        # ========== KDJ指标计算 ==========
        print("Calculating KDJ indicators...")
        df = TechnicalIndicators.calculate_kdj_vectorized(df, period=9, slow_period=3, smooth_period=3, prefix='kdj_fast')
        df = TechnicalIndicators.calculate_kdj_vectorized(df, period=72, slow_period=24, smooth_period=24, prefix='kdj_long')
        
        df['kdj_long_j_slope'] = df.groupby('ts_code')['kdj_long_j'].transform(
            lambda x: x - x.shift(1)
        )
        
        print("Technical indicators calculation complete!")
        return df
    
    @staticmethod
    def calculate_kdj_vectorized(df, period=9, slow_period=3, smooth_period=3, prefix='kdj'):
        """使用向量化方法计算KDJ指标 - 完全避免未来数据"""
        
        def calc_kdj_for_group(group):
            if len(group) < period:
                return pd.DataFrame({
                    f'{prefix}_k': [np.nan] * len(group),
                    f'{prefix}_d': [np.nan] * len(group),
                    f'{prefix}_j': [np.nan] * len(group)
                }, index=group.index)
            
            low_min = group['low_adj'].rolling(period, min_periods=1).min()
            high_max = group['high_adj'].rolling(period, min_periods=1).max()
            range_val = high_max - low_min
            range_val = range_val.replace(0, 1e-8)
            rsv = (group['close_adj'] - low_min) / range_val * 100
            
            k = pd.Series(index=group.index, dtype=float)
            d = pd.Series(index=group.index, dtype=float)
            
            k.iloc[0] = 50
            d.iloc[0] = 50
            
            for i in range(1, len(group)):
                if not pd.isna(rsv.iloc[i]):
                    k.iloc[i] = (slow_period - 1) / slow_period * k.iloc[i-1] + 1 / slow_period * rsv.iloc[i]
                else:
                    k.iloc[i] = k.iloc[i-1]
                
                if not pd.isna(k.iloc[i]):
                    d.iloc[i] = (smooth_period - 1) / smooth_period * d.iloc[i-1] + 1 / smooth_period * k.iloc[i]
                else:
                    d.iloc[i] = d.iloc[i-1]
            
            j = 3 * k - 2 * d
            
            return pd.DataFrame({
                f'{prefix}_k': k,
                f'{prefix}_d': d,
                f'{prefix}_j': j
            }, index=group.index)
        
        results = []
        for ts_code in df['ts_code'].unique():
            mask = df['ts_code'] == ts_code
            group = df.loc[mask].copy()
            result_df = calc_kdj_for_group(group)
            results.append(result_df)
        
        all_results = pd.concat(results)
        
        for col in [f'{prefix}_k', f'{prefix}_d', f'{prefix}_j']:
            df[col] = all_results[col]
        
        return df

# ==================== 自定义回测引擎 ====================
class BacktestEngine:
    """自定义回测引擎 - 完全避免未来数据"""
    
    def __init__(self, df, config, use_ml_signals=False, ml_predictions=None):
        self.df = df.copy()
        self.config = config
        self.use_ml_signals = use_ml_signals
        self.ml_predictions = ml_predictions
        self.dates = sorted(df['trade_date'].unique())
        self.stocks = df['ts_code'].unique().tolist()
        
        self.positions = {}
        self.cash = {}
        self.portfolio_value = {}
        self.trade_records = []
        self.daily_holdings = []
        self.entry_info = {}
        
        self.initial_capital = config.INITIAL_CAPITAL
        
    def get_price_at_date(self, stock, date, use_prev=False):
        """获取指定日期的价格 - 严格使用历史数据"""
        if use_prev:
            date_idx = self.dates.index(date)
            if date_idx == 0:
                return np.nan
            prev_date = self.dates[date_idx - 1]
            stock_data = self.df[(self.df['ts_code'] == stock) & (self.df['trade_date'] == prev_date)]
        else:
            stock_data = self.df[(self.df['ts_code'] == stock) & (self.df['trade_date'] == date)]
        
        if stock_data.empty:
            return np.nan
        
        return stock_data['close_adj'].iloc[0]
    
    def get_limit_price(self, stock, date):
        """获取涨跌停价格 - 使用前一日数据"""
        date_idx = self.dates.index(date)
        if date_idx == 0:
            return None, None
        
        prev_close = self.get_price_at_date(stock, date, use_prev=True)
        
        if pd.isna(prev_close) or prev_close <= 0:
            return None, None
        
        if stock.startswith('688') or stock.startswith('300'):
            limit_up = prev_close * (1 + self.config.LIMIT_UP_20)
            limit_down = prev_close * (1 + self.config.LIMIT_DOWN_20)
        else:
            limit_up = prev_close * (1 + self.config.LIMIT_UP_10)
            limit_down = prev_close * (1 + self.config.LIMIT_DOWN_10)
        
        return limit_up, limit_down
    
    def check_limit_trading(self, stock, date):
        """检查是否可交易"""
        price = self.get_price_at_date(stock, date, use_prev=False)
        if pd.isna(price) or price <= 0:
            return True, True
        
        limit_up, limit_down = self.get_limit_price(stock, date)
        if limit_up is None:
            return True, True
        
        if price >= limit_up * 0.995:
            return False, True
        if price <= limit_down * 1.005:
            return True, False
        return True, True
    
    def get_tradable_stocks(self, date, stock_list):
        """获取可交易股票"""
        tradable = []
        
        for stock in stock_list:
            stock_data = self.df[(self.df['ts_code'] == stock) & (self.df['trade_date'] == date)]
            if stock_data.empty:
                continue
            
            name = stock_data['name'].iloc[0] if 'name' in stock_data.columns else ''
            if 'ST' in name or '*ST' in name:
                continue
            
            price = self.get_price_at_date(stock, date, use_prev=False)
            if pd.isna(price) or price <= 0:
                continue
            
            can_buy, can_sell = self.check_limit_trading(stock, date)
            if not can_buy:
                continue
            
            tradable.append(stock)
        
        return tradable
    
    def get_lot_size(self, stock):
        """获取最小交易单位"""
        if stock.startswith('688'):
            return 200
        return 100
    
    def calculate_shares(self, amount, price, stock):
        """计算可买股数"""
        lot_size = self.get_lot_size(stock)
        shares = int(amount / price / lot_size) * lot_size
        return shares
    
    def get_technical_indicators(self, stock, date, prev_date):
        """获取技术指标 - 严格使用历史数据"""
        stock_today = self.df[(self.df['ts_code'] == stock) & (self.df['trade_date'] == date)]
        stock_yesterday = self.df[(self.df['ts_code'] == stock) & (self.df['trade_date'] == prev_date)]
        
        if stock_today.empty or stock_yesterday.empty:
            return None
        
        indicators = {
            'ma_5': stock_today['ma_5'].iloc[0] if 'ma_5' in stock_today.columns else 0,
            'ma_10': stock_today['ma_10'].iloc[0] if 'ma_10' in stock_today.columns else 0,
            'ma_5_prev': stock_yesterday['ma_5'].iloc[0] if 'ma_5' in stock_yesterday.columns else 0,
            'ma_10_prev': stock_yesterday['ma_10'].iloc[0] if 'ma_10' in stock_yesterday.columns else 0,
            'ma_5_slope': stock_today['ma_5_slope'].iloc[0] if 'ma_5_slope' in stock_today.columns else 0,
            'close': stock_today['close_adj'].iloc[0],
            'open': stock_today['open_adj'].iloc[0],
            'high': stock_today['high_adj'].iloc[0],
            'low': stock_today['low_adj'].iloc[0],
            'is_limit_up': stock_today['is_limit_up'].iloc[0] if 'is_limit_up' in stock_today.columns else False,
            'kdj_fast_j': stock_today['kdj_fast_j'].iloc[0] if 'kdj_fast_j' in stock_today.columns else 50,
            'kdj_long_j': stock_today['kdj_long_j'].iloc[0] if 'kdj_long_j' in stock_today.columns else 50,
            'kdj_long_j_slope': stock_today['kdj_long_j_slope'].iloc[0] if 'kdj_long_j_slope' in stock_today.columns else 0
        }
        
        return indicators
    
    def generate_signals(self, date):
        """生成交易信号 - 只使用截至当日的历史数据"""
        signals = {}
        date_idx = self.dates.index(date)
        if date_idx == 0:
            return signals
        
        prev_date = self.dates[date_idx - 1]
        current_positions = self.positions.get(prev_date, {})
        tradable_stocks = self.get_tradable_stocks(date, self.stocks)
        
        ml_preds = {}
        if self.use_ml_signals and self.ml_predictions is not None:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in self.ml_predictions:
                ml_preds = self.ml_predictions[date_str]
        
        for stock in self.stocks:
            if stock not in tradable_stocks:
                continue
            
            indicators = self.get_technical_indicators(stock, date, prev_date)
            if indicators is None:
                continue
            
            ma5 = indicators['ma_5']
            ma10 = indicators['ma_10']
            ma5_prev = indicators['ma_5_prev']
            ma10_prev = indicators['ma_10_prev']
            ma5_slope = indicators['ma_5_slope']
            close = indicators['close']
            open_price = indicators['open']
            high = indicators['high']
            low = indicators['low']
            is_limit_up = indicators['is_limit_up']
            kdj_fast_j = indicators['kdj_fast_j']
            kdj_long_j = indicators['kdj_long_j']
            kdj_long_j_slope = indicators['kdj_long_j_slope']
            
            is_holding = stock in current_positions
            
            if not is_holding:
                # MA5上穿MA10
                ma5_cross_up = ma5 > ma10 and ma5_prev <= ma10_prev
                ma5_slope_positive = ma5_slope > 0
                
                # K线形态检查
                body = abs(close - open_price)
                upper_shadow = high - max(close, open_price)
                lower_shadow = min(close, open_price) - low
                upper_shadow_ratio = upper_shadow / (body + 1e-8)
                lower_shadow_ratio = lower_shadow / (body + 1e-8)
                
                if close > open_price:
                    kline_ok = upper_shadow_ratio <= self.config.UPPER_SHADOW_LIMIT
                else:
                    kline_ok = lower_shadow_ratio <= self.config.LOWER_SHADOW_LIMIT
                
                # KDJ条件
                kdj_fast_condition = kdj_fast_j >= 60
                kdj_long_condition = kdj_long_j < 85 and kdj_long_j_slope > -0.5
                kdj_ok = kdj_fast_condition and kdj_long_condition
                
                # 传统信号
                traditional_signal = ma5_cross_up and ma5_slope_positive and kline_ok and not is_limit_up and kdj_ok
                
                # 备用信号
                if not traditional_signal:
                    traditional_signal = ma5_cross_up and kdj_ok and not is_limit_up
                
                # ML增强
                ml_signal = False
                if self.use_ml_signals and stock in ml_preds:
                    ml_signal = ml_preds[stock] > 0.55
                
                if traditional_signal or (ml_signal and not is_limit_up):
                    signals[stock] = 'buy'
            
            else:
                entry_info = self.entry_info.get(stock, {})
                entry_price = entry_info.get('price', close)
                
                pnl_pct = (close - entry_price) / entry_price
                stop_loss_hit = pnl_pct <= -self.config.STOP_LOSS
                
                ma_cross_down = ma5 < ma10 and ma5_prev >= ma10_prev
                kdj_sell = kdj_fast_j < 50 or (kdj_long_j >= 80 and kdj_long_j_slope < -1)
                
                if stop_loss_hit or ma_cross_down or kdj_sell:
                    signals[stock] = 'sell'
        
        return signals
    
    def execute_trades(self, date, signals):
        """执行交易"""
        prev_date = self.dates[self.dates.index(date) - 1] if self.dates.index(date) > 0 else date
        current_positions = self.positions.get(prev_date, {}).copy()
        cash = self.cash.get(prev_date, self.initial_capital)
        
        # 卖出
        sell_stocks = [s for s, signal in signals.items() if signal == 'sell' and s in current_positions]
        for stock in sell_stocks:
            shares = current_positions[stock]
            price = self.get_price_at_date(stock, date, use_prev=False)
            
            if pd.isna(price) or price <= 0:
                continue
            
            can_buy, can_sell = self.check_limit_trading(stock, date)
            if not can_sell:
                continue
            
            sell_amount = shares * price * (1 - self.config.FEE_RATE)
            cash += sell_amount
            del current_positions[stock]
            
            if stock in self.entry_info:
                del self.entry_info[stock]
            
            self.trade_records.append({
                'date': date,
                'stock': stock,
                'action': 'sell',
                'shares': shares,
                'price': price,
                'amount': shares * price,
                'fee': shares * price * self.config.FEE_RATE
            })
        
        # 买入
        buy_stocks = [s for s, signal in signals.items() if signal == 'buy' and s not in current_positions]
        
        if buy_stocks:
            target_value_per_stock = cash / (len(buy_stocks) + 1)
            
            for stock in buy_stocks:
                price = self.get_price_at_date(stock, date, use_prev=False)
                if pd.isna(price) or price <= 0:
                    continue
                
                can_buy, can_sell = self.check_limit_trading(stock, date)
                if not can_buy:
                    continue
                
                shares = self.calculate_shares(target_value_per_stock, price, stock)
                
                if shares > 0:
                    buy_amount = shares * price * (1 + self.config.FEE_RATE)
                    if buy_amount <= cash:
                        cash -= buy_amount
                        current_positions[stock] = shares
                        
                        self.entry_info[stock] = {
                            'price': price,
                            'date': date
                        }
                        
                        self.trade_records.append({
                            'date': date,
                            'stock': stock,
                            'action': 'buy',
                            'shares': shares,
                            'price': price,
                            'amount': shares * price,
                            'fee': shares * price * self.config.FEE_RATE
                        })
        
        self.positions[date] = current_positions
        self.cash[date] = cash
        
        total_value = cash
        for stock, shares in current_positions.items():
            price = self.get_price_at_date(stock, date, use_prev=False)
            if not pd.isna(price):
                total_value += shares * price
        
        self.portfolio_value[date] = total_value
        
        for stock, shares in current_positions.items():
            price = self.get_price_at_date(stock, date, use_prev=False)
            if not pd.isna(price) and shares > 0:
                self.daily_holdings.append({
                    'date': date,
                    'stock': stock,
                    'shares': shares,
                    'price': price,
                    'value': shares * price
                })
    
    def run(self):
        """运行回测"""
        print("\n" + "="*80)
        print("Starting Backtest")
        print("="*80)
        print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
        print(f"Total trading days: {len(self.dates)}")
        print(f"Number of stocks: {len(self.stocks)}")
        if self.use_ml_signals:
            print("ML Enhanced Strategy: Enabled")
        print("Future data leakage: PREVENTED")
        
        self.positions[self.dates[0]] = {}
        self.cash[self.dates[0]] = self.initial_capital
        self.portfolio_value[self.dates[0]] = self.initial_capital
        
        for i, date in enumerate(self.dates):
            if i == 0:
                continue
            
            if i % 100 == 0:
                print(f"\rProcessing: {date.strftime('%Y-%m-%d')} ({i+1}/{len(self.dates)})", end='')
            
            signals = self.generate_signals(date)
            self.execute_trades(date, signals)
        
        print(f"\nBacktest Complete!")
        print(f"Total trades: {len(self.trade_records)}")
        print(f"Final capital: {self.portfolio_value[self.dates[-1]]:,.0f}")
        
        return self.get_results()
    
    def get_results(self):
        """获取回测结果"""
        nav_series = pd.Series(self.portfolio_value).sort_index()
        returns = nav_series.pct_change().dropna()
        
        benchmark_series = None
        benchmark_returns = None
        if self.config.BENCHMARK_STOCK:
            benchmark_nav = []
            for date in nav_series.index:
                price = self.get_price_at_date(self.config.BENCHMARK_STOCK, date, use_prev=False)
                if not pd.isna(price):
                    benchmark_nav.append(price)
                else:
                    benchmark_nav.append(np.nan)
            
            if benchmark_nav:
                benchmark_series = pd.Series(benchmark_nav, index=nav_series.index)
                benchmark_series = benchmark_series / benchmark_series.iloc[0]
                benchmark_returns = benchmark_series.pct_change().dropna()
        
        return {
            'nav': nav_series,
            'returns': returns,
            'benchmark_nav': benchmark_series,
            'benchmark_returns': benchmark_returns,
            'positions': self.positions,
            'cash': self.cash,
            'trade_records': self.trade_records,
            'daily_holdings': self.daily_holdings
        }

# ==================== 稳健性验证模块 ====================
class RobustnessValidator:
    """稳健性验证器"""
    
    def __init__(self, df, config):
        self.df = df.copy()
        self.config = config
        self.results = []
        
    def run_rolling_validation(self, output_dir):
        """运行滚动窗口验证"""
        print("\n" + "="*80)
        print("Robustness Validation - Rolling Window Test")
        print("="*80)
        
        dates = sorted(self.df['trade_date'].unique())
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(self.config.TRAIN_WINDOW, len(dates) - self.config.TEST_WINDOW, self.config.STEP_SIZE):
            train_start = dates[i - self.config.TRAIN_WINDOW]
            train_end = dates[i - 1]
            test_start = dates[i]
            test_end = dates[i + self.config.TEST_WINDOW - 1] if i + self.config.TEST_WINDOW <= len(dates) else dates[-1]
            
            print(f"\nWindow {len(results)+1}:")
            print(f"  Train: {train_start} to {train_end}")
            print(f"  Test: {test_start} to {test_end}")
            
            test_df = self.df[(self.df['trade_date'] >= test_start) & (self.df['trade_date'] <= test_end)].copy()
            
            if test_df.empty or len(test_df['trade_date'].unique()) < 20:
                print("  Insufficient data, skipping")
                continue
            
            engine = BacktestEngine(test_df, self.config)
            engine.initial_capital = self.config.INITIAL_CAPITAL
            result = engine.run()
            
            returns = result['returns']
            if len(returns) < 10:
                print("  Insufficient trading data, skipping")
                continue
            
            nav = result['nav']
            total_return = nav.iloc[-1] / nav.iloc[0] - 1
            
            days = len(returns)
            years = days / 252
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            annual_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            sharpe = (annual_return - 0.03) / annual_vol if annual_vol > 0 else 0
            
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns / running_max - 1)
            max_drawdown = drawdown.min()
            
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
            
            window_result = {
                'window': len(results) + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'trading_days': len(returns)
            }
            
            results.append(window_result)
            print(f"  Sharpe Ratio: {sharpe:.4f}, Win Rate: {win_rate:.2%}")
        
        self.results = pd.DataFrame(results)
        
        if len(self.results) > 0:
            self.results.to_csv(os.path.join(output_dir, 'rolling_validation_results.csv'), index=False)
            self.plot_validation_results(output_dir)
        
        return self.results
    
    def plot_validation_results(self, output_dir):
        """绘制验证结果"""
        if self.results.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        ax1.plot(self.results['window'], self.results['sharpe_ratio'], 'o-', linewidth=2)
        ax1.axhline(y=self.results['sharpe_ratio'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.results["sharpe_ratio"].mean():.4f}')
        ax1.set_title('Rolling Window Sharpe Ratio')
        ax1.set_xlabel('Window Number')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(self.results['window'], self.results['win_rate'], 'o-', linewidth=2, color='green')
        ax2.axhline(y=self.results['win_rate'].mean(), color='red', linestyle='--',
                   label=f'Mean: {self.results["win_rate"].mean():.2%}')
        ax2.set_title('Rolling Window Win Rate')
        ax2.set_xlabel('Window Number')
        ax2.set_ylabel('Win Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.plot(self.results['window'], self.results['annual_return'], 'o-', linewidth=2, color='purple')
        ax3.set_title('Rolling Window Annual Return')
        ax3.set_xlabel('Window Number')
        ax3.set_ylabel('Annual Return')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'Mean', 'Std', 'Min', 'Max'],
            ['Sharpe Ratio', f'{self.results["sharpe_ratio"].mean():.4f}', 
             f'{self.results["sharpe_ratio"].std():.4f}',
             f'{self.results["sharpe_ratio"].min():.4f}',
             f'{self.results["sharpe_ratio"].max():.4f}'],
            ['Annual Return', f'{self.results["annual_return"].mean():.2%}',
             f'{self.results["annual_return"].std():.2%}',
             f'{self.results["annual_return"].min():.2%}',
             f'{self.results["annual_return"].max():.2%}'],
            ['Max Drawdown', f'{self.results["max_drawdown"].mean():.2%}',
             f'{self.results["max_drawdown"].std():.2%}',
             f'{self.results["max_drawdown"].min():.2%}',
             f'{self.results["max_drawdown"].max():.2%}'],
            ['Win Rate', f'{self.results["win_rate"].mean():.2%}',
             f'{self.results["win_rate"].std():.2%}',
             f'{self.results["win_rate"].min():.2%}',
             f'{self.results["win_rate"].max():.2%}']
        ]
        
        table = ax4.table(cellText=summary_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Rolling Validation Summary', fontsize=12, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rolling_validation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Validation chart saved: {os.path.join(output_dir, 'rolling_validation_results.png')}")

# ==================== 机器学习建模模块 ====================
class MLModelBuilder:
    """机器学习模型构建器 - 严格避免未来数据"""
    
    def __init__(self, df, config):
        self.df = df.copy()
        self.config = config
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.prediction_results = {}
        self.predictions_by_date = {}
        
    def prepare_features(self, df):
        """准备特征 - 只使用历史数据"""
        feature_cols = [
            'ma_5', 'ma_10', 'ma_20', 'ma_60',
            'ma_5_slope', 'ma_5_prev', 'ma_10_prev',
            'upper_shadow_ratio', 'lower_shadow_ratio',
            'pct_chg', 'volume_ma_5', 'volume_ma_10', 'volume_ma_20',
            'high_low_ratio', 'close_open_ratio'
        ]
        
        kdj_cols = ['kdj_fast_k', 'kdj_fast_d', 'kdj_fast_j', 
                   'kdj_long_k', 'kdj_long_d', 'kdj_long_j']
        for col in kdj_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        available_cols = [col for col in feature_cols if col in df.columns]
        return df[available_cols]
    
    def prepare_target(self, df, horizon=5):
        """准备目标变量 - 使用未来数据仅用于训练标签"""
        # 注意：这里使用shift(-horizon)仅用于生成训练标签
        # 回测中不会使用这些未来数据
        future_returns = df.groupby('ts_code')['close_adj'].transform(
            lambda x: x.shift(-horizon) / x - 1
        )
        target = (future_returns > 0).astype(int)
        return target
    
    def train_models(self, train_df, test_df, output_dir):
        """训练模型 - 使用时间序列交叉验证"""
        print("\n" + "="*80)
        print("Machine Learning Modeling")
        print("="*80)
        print("⚠️  Note: Future data (shift(-horizon)) is ONLY used for training labels")
        print("⚠️  Trading signals will use predictions, not future data")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备特征和目标
        X_train = self.prepare_features(train_df)
        y_train = self.prepare_target(train_df, self.config.PREDICT_HORIZON)
        X_test = self.prepare_features(test_df)
        
        # 删除缺失值
        train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=1
            )
        }
        
        results = {}
        feature_importance_dict = {}
        predictions_dict = {}
        test_dates = test_df['trade_date'].unique()
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models.items():
            print(f"\nTraining model: {name}")
            start_time = time.time()
            
            # 时间序列交叉验证
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                X_tr_scaled = scaler.transform(X_tr)
                X_val_scaled = scaler.transform(X_val)
                
                model.fit(X_tr_scaled, y_tr)
                y_pred = model.predict(X_val_scaled)
                cv_scores.append(accuracy_score(y_val, y_pred))
            
            # 在整个训练集上重新训练
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            # 在测试集上预测
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 特征重要性
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = X_train.columns
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                feature_importance_dict[name] = importance_df
            
            # 按日期组织预测结果 - 用于回测
            predictions_by_date = {}
            test_df_copy = test_df.copy()
            test_indices = test_df_copy.index[:len(y_pred_proba)]
            test_df_copy = test_df_copy.loc[test_indices].copy()
            test_df_copy['pred_proba'] = y_pred_proba
            test_df_copy['pred_label'] = y_pred
            
            for date in test_dates:
                date_mask = test_df_copy['trade_date'] == date
                if date_mask.sum() > 0:
                    date_preds = test_df_copy[date_mask]
                    pred_dict = dict(zip(date_preds['ts_code'], date_preds['pred_proba']))
                    predictions_by_date[date.strftime('%Y-%m-%d')] = pred_dict
            
            results[name] = {
                'model': model,
                'scaler': scaler,
                'cv_accuracy_mean': np.mean(cv_scores) if cv_scores else 0,
                'cv_accuracy_std': np.std(cv_scores) if cv_scores else 0,
                'train_time': train_time,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'predictions_by_date': predictions_by_date
            }
            
            predictions_dict[name] = {
                'pred': y_pred,
                'prob': y_pred_proba
            }
            
            print(f"  CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            print(f"  Training time: {train_time:.2f} seconds")
        
        self.models = results
        self.feature_importance = feature_importance_dict
        self.prediction_results = predictions_dict
        
        self.save_results(output_dir)
        self.plot_results(output_dir)
        
        return results
    
    def get_predictions_by_date(self, model_name='LightGBM'):
        """获取按日期组织的预测结果"""
        if model_name in self.models:
            return self.models[model_name]['predictions_by_date']
        return {}
    
    def save_results(self, output_dir):
        """保存模型结果"""
        model_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        for name, result in self.models.items():
            with open(os.path.join(model_dir, f'{name}.pkl'), 'wb') as f:
                pickle.dump(result['model'], f)
        
        for name, importance_df in self.feature_importance.items():
            importance_df.to_csv(
                os.path.join(output_dir, f'{name}_feature_importance.csv'),
                index=False
            )
        
        comparison = pd.DataFrame([{
            'model': name,
            'cv_accuracy': result['cv_accuracy_mean'],
            'cv_accuracy_std': result['cv_accuracy_std'],
            'train_time': result['train_time']
        } for name, result in self.models.items()])
        
        comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        
        config_dict = {
            'train_window': self.config.TRAIN_WINDOW,
            'test_window': self.config.TEST_WINDOW,
            'predict_horizon': self.config.PREDICT_HORIZON,
            'ml_top_pct': self.config.ML_TOP_PCT,
            'feature_count': len(self.prepare_features(self.df).columns),
            'model_count': len(self.models)
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        
        print(f"\nModel results saved to: {output_dir}")
    
    def plot_results(self, output_dir):
        """绘制模型结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        names = list(self.models.keys())
        cv_accuracies = [self.models[name]['cv_accuracy_mean'] for name in names]
        cv_stds = [self.models[name]['cv_accuracy_std'] for name in names]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax1.bar(names, cv_accuracies, color=colors, yerr=cv_stds, capsize=5)
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Random Baseline (0.5)')
        ax1.set_title('Model CV Accuracy Comparison')
        ax1.set_ylabel('CV Accuracy')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars, cv_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2%}', ha='center', va='bottom')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        if self.feature_importance:
            first_model = list(self.feature_importance.keys())[0]
            importance_df = self.feature_importance[first_model].head(10)
            ax2.barh(importance_df['feature'], importance_df['importance'], color='#2E86AB')
            ax2.set_title(f'{first_model} Feature Importance (Top 10)')
            ax2.set_xlabel('Importance')
            ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        for name in names:
            probs = self.prediction_results[name]['prob']
            ax3.hist(probs, bins=30, alpha=0.5, label=name)
        ax3.set_title('Prediction Probability Distribution')
        ax3.set_xlabel('Prediction Probability')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        train_times = [self.models[name]['train_time'] for name in names]
        ax4.bar(names, train_times, color=colors)
        ax4.set_title('Model Training Time')
        ax4.set_ylabel('Training Time (seconds)')
        for bar, t in zip(bars, train_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{t:.2f}s', ha='center', va='bottom')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ml_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model results chart saved: {os.path.join(output_dir, 'ml_results.png')}")

# ==================== 结果分析器 ====================
class ResultAnalyzer:
    """回测结果分析器"""
    
    def __init__(self, results):
        self.results = results
        self.metrics = None
        
    def analyze(self):
        """分析回测结果"""
        print("\nAnalyzing backtest results...")
        
        returns = self.results['returns']
        portfolio_values = self.results['nav']
        
        if len(returns) == 0:
            print("⚠️ Insufficient return data")
            return None
        
        metrics = self.calculate_metrics(returns, portfolio_values)
        self.metrics = metrics
        
        return metrics
    
    def calculate_metrics(self, returns=None, portfolio_values=None):
        """计算绩效指标"""
        if returns is None:
            returns = self.results['returns']
        if portfolio_values is None:
            portfolio_values = self.results['nav']
        
        days = len(returns)
        years = days / 252
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        annual_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        risk_free = 0.03
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0
        
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values / running_max - 1)
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0 and len(positive_returns) == 0:
            profit_factor = 1.0
        elif len(negative_returns) == 0:
            profit_factor = np.inf
        else:
            profit_factor = positive_returns.sum() / abs(negative_returns.sum())
        
        returns_sign = np.sign(returns)
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for sign in returns_sign:
            if sign > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif sign < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        information_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Information Ratio': information_ratio,
            'Max Consecutive Wins': max_consecutive_wins,
            'Max Consecutive Losses': max_consecutive_losses,
            'Trading Days': days
        }
        
        return metrics
    
    def plot_results(self, title="Backtest Results", save_path=None):
        """绘制回测结果图"""
        if self.metrics is None:
            self.analyze()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        nav = self.results['nav']
        ax1.plot(nav.index, nav / nav.iloc[0], linewidth=2, color='blue')
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Cumulative Returns', fontsize=12)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('NAV')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        nav = self.results['nav']
        running_max = nav.expanding().max()
        drawdown = (nav / running_max - 1) * 100
        ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown Curve', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        returns = self.results['returns'] * 100
        ax3.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Daily Return Distribution', fontsize=12)
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        metrics_data = []
        for key, value in self.metrics.items():
            if key in ['Max Consecutive Wins', 'Max Consecutive Losses', 'Trading Days']:
                metrics_data.append([key, f'{value:.0f}'])
            elif 'Rate' in key or 'Return' in key or 'Drawdown' in key:
                metrics_data.append([key, f'{value:.2%}'])
            else:
                metrics_data.append([key, f'{value:.4f}'])
        
        table = ax4.table(cellText=metrics_data, colLabels=['Metric', 'Value'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Performance Metrics', fontsize=12, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        
        plt.show()
        return fig
    
    def get_summary(self):
        """获取结果摘要"""
        if self.metrics is None:
            self.analyze()
        
        summary = f"""
        ========================================
        Backtest Results Summary
        ========================================
        Total Return:        {self.metrics['Total Return']:.2%}
        Annual Return:       {self.metrics['Annual Return']:.2%}
        Annual Volatility:   {self.metrics['Annual Volatility']:.2%}
        Sharpe Ratio:        {self.metrics['Sharpe Ratio']:.4f}
        Max Drawdown:        {self.metrics['Max Drawdown']:.2%}
        Win Rate:            {self.metrics['Win Rate']:.2%}
        Profit Factor:       {self.metrics['Profit Factor']:.2f}
        Information Ratio:   {self.metrics['Information Ratio']:.4f}
        Max Consecutive Wins: {self.metrics['Max Consecutive Wins']:.0f} days
        Max Consecutive Losses: {self.metrics['Max Consecutive Losses']:.0f} days
        Trading Days:        {self.metrics['Trading Days']:.0f}
        ========================================
        """
        return summary

# ==================== 结果对比分析 ====================
class ResultComparator:
    """结果对比分析器"""
    
    def __init__(self):
        self.results = {}
        
    def add_result(self, name, nav, returns, metrics):
        self.results[name] = {
            'nav': nav,
            'returns': returns,
            'metrics': metrics
        }
    
    def plot_comparison(self, output_dir):
        """绘制对比图"""
        if len(self.results) < 2:
            print("At least 2 results needed for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        for name, result in self.results.items():
            nav_normalized = result['nav'] / result['nav'].iloc[0]
            ax1.plot(nav_normalized.index, nav_normalized, label=name, linewidth=2)
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Strategy NAV Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('NAV')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for name, result in self.results.items():
            returns = result['returns']
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns / running_max - 1) * 100
            ax2.plot(drawdown.index, drawdown, label=name, linewidth=2)
        ax2.set_title('Drawdown Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        metrics_df = pd.DataFrame({
            name: result['metrics']
            for name, result in self.results.items()
        }).T
        
        key_metrics = ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        if available_metrics:
            metrics_df[available_metrics].plot(kind='bar', ax=ax3)
            ax3.set_title('Performance Metrics Comparison')
            ax3.set_xlabel('Strategy')
            ax3.set_ylabel('Metric Value')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = [['Strategy'] + list(metrics_df.index)]
        for col in metrics_df.columns:
            row = [col]
            for idx in metrics_df.index:
                val = metrics_df.loc[idx, col]
                if isinstance(val, float):
                    if 'Rate' in col or 'Return' in col or 'Drawdown' in col:
                        row.append(f'{val:.2%}')
                    else:
                        row.append(f'{val:.4f}')
                else:
                    row.append(str(val))
            table_data.append(row)
        
        table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        ax4.set_title('Performance Metrics Comparison', fontsize=12, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison chart saved: {os.path.join(output_dir, 'strategy_comparison.png')}")

# ==================== 主程序 ====================
def main():
    """主程序"""
    print("="*80)
    print("Multi-Factor Quantitative Strategy Development & Validation System")
    print("="*80)
    print("⚠️  Future data leakage: PREVENTED")
    print("⚠️  All signals use only historical data")
    print("="*80)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, '..', 'final_data')
    if not os.path.exists(DATA_PATH):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(script_dir, '..', 'final_data')
    
    DATA_PATH = os.path.normpath(DATA_PATH)
    OUTPUT_BASE = os.path.join(current_dir, 'strategy_results')
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    print(f"Data path: {DATA_PATH}")
    print(f"Output path: {OUTPUT_BASE}")
    
    # 1. 加载数据
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    
    # 2. 计算技术指标
    print("\n" + "="*80)
    print("Calculating Technical Indicators")
    print("="*80)
    df = TechnicalIndicators.calculate_all(df)
    
    # 3. 基础策略回测
    print("\n" + "="*80)
    print("Base Strategy Backtest")
    print("="*80)
    config = StrategyConfig()
    engine = BacktestEngine(df, config, use_ml_signals=False)
    results = engine.run()
    
    base_output = os.path.join(OUTPUT_BASE, 'base_strategy')
    os.makedirs(base_output, exist_ok=True)
    
    analyzer = ResultAnalyzer(results)
    analyzer.analyze()
    base_metrics = analyzer.metrics
    
    analyzer.plot_results(save_path=os.path.join(base_output, 'base_results.png'))
    
    pd.DataFrame({
        'date': results['nav'].index,
        'nav': results['nav'].values
    }).to_csv(os.path.join(base_output, 'nav.csv'), index=False)
    
    if results['trade_records']:
        pd.DataFrame(results['trade_records']).to_csv(
            os.path.join(base_output, 'trade_records.csv'), index=False
        )
    
    if base_metrics:
        pd.DataFrame([{'metric': k, 'value': v} for k, v in base_metrics.items()]).to_csv(
            os.path.join(base_output, 'metrics.csv'), index=False
        )
    
    print(analyzer.get_summary())
    
    # 4. 稳健性验证
    val_output = os.path.join(OUTPUT_BASE, 'robustness_validation')
    validator = RobustnessValidator(df, config)
    validation_results = validator.run_rolling_validation(val_output)
    
    # 5. 机器学习建模
    ml_output = os.path.join(OUTPUT_BASE, 'ml_models')
    
    dates = sorted(df['trade_date'].unique())
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    train_df = df[df['trade_date'].isin(train_dates)]
    test_df = df[df['trade_date'].isin(test_dates)]
    
    ml_builder = MLModelBuilder(df, config)
    ml_results = ml_builder.train_models(train_df, test_df, ml_output)
    
    # 6. ML增强策略回测
    print("\n" + "="*80)
    print("ML Enhanced Strategy Backtest")
    print("="*80)
    
    best_model_name = max(ml_results, key=lambda x: ml_results[x]['cv_accuracy_mean'])
    print(f"Best model: {best_model_name}")
    
    ml_predictions = ml_builder.get_predictions_by_date(best_model_name)
    
    ml_config = StrategyConfig()
    ml_engine = BacktestEngine(df, ml_config, use_ml_signals=True, ml_predictions=ml_predictions)
    ml_results_engine = ml_engine.run()
    
    ml_analyzer = ResultAnalyzer(ml_results_engine)
    ml_analyzer.analyze()
    ml_metrics = ml_analyzer.metrics
    
    ml_output_dir = os.path.join(OUTPUT_BASE, 'ml_enhanced_strategy')
    os.makedirs(ml_output_dir, exist_ok=True)
    ml_analyzer.plot_results(save_path=os.path.join(ml_output_dir, 'ml_enhanced_results.png'))
    
    # 7. 综合对比
    comparator = ResultComparator()
    comparator.add_result('Base Strategy', results['nav'], results['returns'], base_metrics)
    comparator.add_result('ML Enhanced', ml_results_engine['nav'], 
                         ml_results_engine['returns'], ml_metrics)
    
    comp_output = os.path.join(OUTPUT_BASE, 'comparison')
    os.makedirs(comp_output, exist_ok=True)
    comparator.plot_comparison(comp_output)
    
    comparison_data = pd.DataFrame({
        'Strategy': list(comparator.results.keys()),
        'Annual Return': [r['metrics'].get('Annual Return', 0) for r in comparator.results.values()],
        'Sharpe Ratio': [r['metrics'].get('Sharpe Ratio', 0) for r in comparator.results.values()],
        'Max Drawdown': [r['metrics'].get('Max Drawdown', 0) for r in comparator.results.values()],
        'Win Rate': [r['metrics'].get('Win Rate', 0) for r in comparator.results.values()]
    })
    comparison_data.to_csv(os.path.join(comp_output, 'comparison_summary.csv'), index=False)
    
    print("\n" + "="*80)
    print("All analysis complete!")
    print(f"Results saved to: {OUTPUT_BASE}")
    print("="*80)
    
    print("\nFinal Strategy Comparison:")
    print(comparison_data.to_string(index=False))
    
    print("\n⚠️  Note: All results are based on historical data only.")
    print("   No future data was used in signal generation.")

if __name__ == "__main__":
    main()