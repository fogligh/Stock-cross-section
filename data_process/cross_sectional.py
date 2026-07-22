"""
量化回测系统 - 基于因子选股的每日换手策略
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置参数 ====================
class Config:
    """回测配置"""
    # 交易参数
    TOP_N = 200  # 持有股票数量
    MAX_TURNOVER = 0.30  # 最大换手率30%
    FEE_RATE = 0.0014  # 双边费率14bp
    LOT_SIZE = 100  # 基本交易单位
    LOT_SIZE_688 = 200  # 688开头股票交易单位
    
    # 涨跌停限制
    LIMIT_UP_10 = 0.095  # 10%涨跌幅（实际约9.5%）
    LIMIT_DOWN_10 = -0.095
    LIMIT_UP_20 = 0.195  # 20%涨跌幅（实际约19.5%）
    LIMIT_DOWN_20 = -0.195
    
    # 股票池配置
    STOCK_POOL = 'all'  # 'all' 或 'A500' 或 'ZZ1000'
    
    # 因子配置
    FACTOR_NAME = 'ret_20d'  # 使用的因子名称（20日收益率）
    
    # 基准股票
    BENCHMARK_STOCK = '000002.SZ'

# ==================== 数据加载模块 ====================
class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path, benchmark_stock):
        self.data_path = Path(data_path)
        self.benchmark_stock = benchmark_stock
        
    def load_data(self):
        """加载所有数据"""
        print("加载数据...")
        
        # 加载主数据
        csv_file = self.data_path / "final_200stocks_with_benchmark_2020_2025.csv"
        
        if csv_file.exists():
            print(f"读取CSV文件: {csv_file}")
            df = pd.read_csv(csv_file)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            print(f"数据形状: {df.shape}")
        else:
            # 尝试Parquet格式
            parquet_file = self.data_path / "final_200stocks_with_benchmark_2020_2025.parquet"
            if parquet_file.exists():
                print(f"读取Parquet文件: {parquet_file}")
                df = pd.read_parquet(parquet_file)
                print(f"数据形状: {df.shape}")
            else:
                raise FileNotFoundError(f"数据文件不存在: {csv_file} 或 {parquet_file}")
        
        # 排序
        df = df.sort_values(['ts_code', 'trade_date'])
        
        # 获取股票信息
        stock_info = df[['ts_code', 'symbol', 'name', 'industry']].drop_duplicates('ts_code')
        print(f"股票数量: {len(stock_info)}")
        
        # 获取基准数据
        benchmark_df = df[df['ts_code'] == self.benchmark_stock].copy()
        if benchmark_df.empty:
            print(f"⚠️ 警告: 基准股票 {self.benchmark_stock} 不存在于数据中")
            print(f"将使用第一个股票作为基准")
            first_stock = stock_info['ts_code'].iloc[0]
            benchmark_df = df[df['ts_code'] == first_stock].copy()
            self.benchmark_stock = first_stock
            print(f"使用 {self.benchmark_stock} 作为基准")
        
        # 获取价格数据
        price_data = df.pivot(index='trade_date', columns='ts_code', values='close_adj')
        twap_data = df.pivot(index='trade_date', columns='ts_code', values='twap_adj')
        accum_factor = df.pivot(index='trade_date', columns='ts_code', values='accum_factor')
        
        # 获取因子数据 - 从factor列中提取
        factor_cols = [col for col in df.columns if col not in ['trade_date', 'ts_code', 'symbol', 'name', 
                                                                 'industry', 'open', 'high', 'low', 'close', 
                                                                 'vol', 'amount', 'twap', 'accum_factor', 
                                                                 'close_adj', 'twap_adj', 'open_adj', 
                                                                 'high_adj', 'low_adj']]
        
        print(f"因子数量: {len(factor_cols)}")
        
        factor_data = {}
        for col in factor_cols:
            factor_data[col] = df.pivot(index='trade_date', columns='ts_code', values=col)
        
        # 获取ST标识
        st_data = df[['trade_date', 'ts_code', 'name']].copy()
        st_data['is_st'] = st_data['name'].str.contains('ST|\\*ST', case=False, na=False)
        st_pivot = st_data.pivot(index='trade_date', columns='ts_code', values='is_st').fillna(False)
        
        print("数据加载完成！")
        print(f"日期范围: {price_data.index.min()} 到 {price_data.index.max()}")
        print(f"股票数量: {len(price_data.columns)}")
        
        return {
            'df': df,
            'stock_info': stock_info,
            'benchmark': benchmark_df,
            'benchmark_stock': self.benchmark_stock,
            'price': price_data,
            'twap': twap_data,
            'accum_factor': accum_factor,
            'factors': factor_data,
            'st': st_pivot,
            'factor_cols': factor_cols
        }

# ==================== 回测引擎 ====================
class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.dates = sorted(data['price'].index)
        self.stocks = data['stock_info']['ts_code'].tolist()
        
        # 初始化结果
        self.positions = {}  # 持仓 {date: {stock: shares}}
        self.cash = {}  # 现金 {date: amount}
        self.portfolio_value = {}  # 总资产 {date: value}
        self.trade_records = []  # 交易记录
        self.daily_holdings = []  # 每日持仓记录
        
        # 账户初始资金
        self.initial_capital = 100000000  # 1亿
        
    def get_limit_price(self, stock, date):
        """获取涨跌停价格"""
        # 获取昨日收盘价
        date_idx = self.dates.index(date)
        if date_idx == 0:
            return None, None
        prev_date = self.dates[date_idx - 1]
        
        # 获取昨日收盘价（后复权）
        prev_close = self.data['price'].loc[prev_date, stock] if stock in self.data['price'].columns else None
        if pd.isna(prev_close):
            return None, None
        
        # 判断涨跌停限制
        if stock.startswith('688') or stock.startswith('300'):
            limit_up = prev_close * (1 + self.config.LIMIT_UP_20)
            limit_down = prev_close * (1 + self.config.LIMIT_DOWN_20)
        else:
            limit_up = prev_close * (1 + self.config.LIMIT_UP_10)
            limit_down = prev_close * (1 + self.config.LIMIT_DOWN_10)
        
        return limit_up, limit_down
    
    def is_limit_trading(self, stock, date):
        """判断是否涨跌停"""
        if date not in self.data['twap'].index:
            return True, True
        
        if stock not in self.data['twap'].columns:
            return True, True
        
        price = self.data['twap'].loc[date, stock]
        if pd.isna(price):
            return True, True
        
        limit_up, limit_down = self.get_limit_price(stock, date)
        if limit_up is None:
            return True, True
        
        # 判断是否触及涨跌停（价格在涨跌停价±0.5%范围内视为触及）
        if price >= limit_up * 0.995:
            return True, False
        if price <= limit_down * 1.005:
            return False, True
        
        return False, False
    
    def get_tradable_stocks(self, date, factor_values):
        """获取可交易股票"""
        tradable = []
        
        for stock in factor_values.index:
            # 检查是否在ST中
            if stock in self.data['st'].columns:
                if self.data['st'].loc[date, stock]:
                    continue
            
            # 检查是否有价格数据
            if stock not in self.data['twap'].columns:
                continue
            price = self.data['twap'].loc[date, stock]
            if pd.isna(price) or price <= 0:
                continue
            
            # 检查是否涨跌停
            is_up, is_down = self.is_limit_trading(stock, date)
            if is_up or is_down:
                continue
            
            tradable.append(stock)
        
        return tradable
    
    def get_lot_size(self, stock):
        """获取交易单位"""
        if stock.startswith('688'):
            return self.config.LOT_SIZE_688
        return self.config.LOT_SIZE
    
    def calculate_shares(self, amount, price, stock):
        """计算可买入股数（按交易单位取整）"""
        lot_size = self.get_lot_size(stock)
        shares = int(amount / price / lot_size) * lot_size
        return shares
    
    def calculate_factor_scores(self, date, factor_name):
        """计算因子得分"""
        if factor_name not in self.data['factors']:
            print(f"⚠️ 警告: 因子 {factor_name} 不存在")
            available_factors = list(self.data['factors'].keys())
            print(f"可用因子: {available_factors[:10]}...")
            # 使用第一个可用因子
            if available_factors:
                factor_name = available_factors[0]
                print(f"使用 {factor_name} 替代")
            else:
                raise ValueError("没有可用的因子")
        
        # 获取前一天的因子值（使用上一个交易日）
        date_idx = self.dates.index(date)
        if date_idx == 0:
            return pd.Series()
        
        prev_date = self.dates[date_idx - 1]
        factor_values = self.data['factors'][factor_name].loc[prev_date]
        
        # 去除缺失值
        factor_values = factor_values.dropna()
        
        return factor_values
    
    def run(self, factor_name=None):
        """运行回测"""
        if factor_name is None:
            factor_name = self.config.FACTOR_NAME
        
        print(f"\n开始回测...")
        print(f"因子: {factor_name}")
        print(f"持有股票数量: {self.config.TOP_N}")
        print(f"最大换手率: {self.config.MAX_TURNOVER * 100}%")
        print(f"交易费率: {self.config.FEE_RATE * 10000}bp")
        print(f"日期范围: {self.dates[0]} 到 {self.dates[-1]}")
        print(f"总交易日: {len(self.dates)}")
        
        # 初始化账户
        self.cash[self.dates[0]] = self.initial_capital
        self.positions[self.dates[0]] = {}
        
        # 逐日回测
        for i, date in enumerate(self.dates):
            print(f"\r处理日期: {date.strftime('%Y-%m-%d')} ({i+1}/{len(self.dates)})", end='')
            
            # 获取当天因子值（使用前一个交易日）
            prev_factor = self.calculate_factor_scores(date, factor_name)
            
            if prev_factor.empty:
                # 如果没有因子数据，保持持仓不变
                if i > 0:
                    self.positions[date] = self.positions[self.dates[i-1]].copy()
                    self.cash[date] = self.cash[self.dates[i-1]]
                    self.portfolio_value[date] = self.portfolio_value[self.dates[i-1]]
                else:
                    self.positions[date] = {}
                    self.cash[date] = self.initial_capital
                    self.portfolio_value[date] = self.initial_capital
                continue
            
            # 获取可交易股票
            tradable_stocks = self.get_tradable_stocks(date, prev_factor)
            
            # 筛选可交易的因子值
            factor_scores = prev_factor[prev_factor.index.isin(tradable_stocks)]
            factor_scores = factor_scores.sort_values(ascending=False)
            
            # 如果可交易股票少于TOP_N，全部买入
            actual_n = min(self.config.TOP_N, len(factor_scores))
            top_stocks = factor_scores.head(actual_n).index.tolist()
            
            # 获取当前持仓
            current_positions = self.positions.get(self.dates[i-1] if i > 0 else self.dates[0], {})
            
            # 计算目标持仓
            target_positions = {stock: 0 for stock in top_stocks}
            
            # 计算换手
            current_stocks = set(current_positions.keys())
            target_stocks = set(target_positions.keys())
            
            # 需要卖出的股票
            to_sell = current_stocks - target_stocks
            # 需要买入的股票
            to_buy = target_stocks - current_stocks
            # 保持的股票
            to_hold = current_stocks & target_stocks
            
            # 控制换手率
            total_turnover = len(to_sell) / self.config.TOP_N if self.config.TOP_N > 0 else 0
            
            if total_turnover > self.config.MAX_TURNOVER:
                # 如果换手率过高，减少换手
                max_sell = int(self.config.TOP_N * self.config.MAX_TURNOVER)
                # 按因子值排序，卖出因子值最低的
                sell_scores = {stock: factor_scores.get(stock, -np.inf) for stock in to_sell}
                sorted_sell = sorted(sell_scores.items(), key=lambda x: x[1])
                to_sell = [s[0] for s in sorted_sell[:max_sell]]
                # 保留一些股票
                to_hold = current_stocks - set(to_sell)
                # 相应地调整买入
                to_buy = target_stocks - to_hold
                # 补充买入到actual_n
                remaining = actual_n - len(to_hold)
                if remaining > 0:
                    available_buy = [s for s in top_stocks if s not in to_hold and s not in to_sell]
                    to_buy = available_buy[:remaining]
            
            # 执行交易
            new_positions = {}
            cash = self.cash.get(self.dates[i-1] if i > 0 else self.dates[0], self.initial_capital)
            
            # 1. 卖出
            for stock in to_sell:
                shares = current_positions.get(stock, 0)
                if shares > 0:
                    price = self.data['twap'].loc[date, stock]
                    if not pd.isna(price):
                        # 检查是否涨停（不能卖出）
                        is_up, is_down = self.is_limit_trading(stock, date)
                        if not is_up:
                            # 计算卖出金额
                            sell_amount = shares * price * (1 - self.config.FEE_RATE)
                            cash += sell_amount
                            
                            # 记录交易
                            self.trade_records.append({
                                'date': date,
                                'stock': stock,
                                'action': 'sell',
                                'shares': shares,
                                'price': price,
                                'amount': shares * price,
                                'fee': shares * price * self.config.FEE_RATE,
                                'net_amount': sell_amount
                            })
            
            # 2. 买入
            if cash > 0 and len(to_buy) > 0:
                # 等权分配资金
                per_stock_amount = cash / len(to_buy)
                
                for stock in to_buy:
                    price = self.data['twap'].loc[date, stock]
                    if not pd.isna(price):
                        # 检查是否跌停（不能买入）
                        is_up, is_down = self.is_limit_trading(stock, date)
                        if not is_down:
                            # 计算可买入股数
                            shares = self.calculate_shares(per_stock_amount, price, stock)
                            if shares > 0:
                                buy_amount = shares * price * (1 + self.config.FEE_RATE)
                                if buy_amount <= cash:
                                    cash -= buy_amount
                                    new_positions[stock] = shares
                                    
                                    # 记录交易
                                    self.trade_records.append({
                                        'date': date,
                                        'stock': stock,
                                        'action': 'buy',
                                        'shares': shares,
                                        'price': price,
                                        'amount': shares * price,
                                        'fee': shares * price * self.config.FEE_RATE,
                                        'net_amount': buy_amount
                                    })
            
            # 3. 保留持仓
            for stock in to_hold:
                if stock in current_positions:
                    shares = current_positions[stock]
                    # 检查复权因子变化
                    if i > 0:
                        prev_date = self.dates[i-1]
                        old_factor = self.data['accum_factor'].loc[prev_date, stock] if stock in self.data['accum_factor'].columns else 1
                        new_factor = self.data['accum_factor'].loc[date, stock] if stock in self.data['accum_factor'].columns else 1
                        if not pd.isna(old_factor) and not pd.isna(new_factor) and old_factor != 0:
                            # 更新持仓数量（除权除息调整）
                            ratio = new_factor / old_factor
                            if ratio != 1 and ratio > 0:
                                shares = int(shares * ratio)
                    new_positions[stock] = shares
            
            # 更新持仓和现金
            self.positions[date] = new_positions
            self.cash[date] = cash
            
            # 计算总资产
            total_value = cash
            for stock, shares in new_positions.items():
                price = self.data['twap'].loc[date, stock] if stock in self.data['twap'].columns else 0
                if not pd.isna(price):
                    total_value += shares * price
            
            self.portfolio_value[date] = total_value
            
            # 记录每日持仓
            for stock, shares in new_positions.items():
                price = self.data['twap'].loc[date, stock] if stock in self.data['twap'].columns else 0
                if not pd.isna(price) and shares > 0:
                    self.daily_holdings.append({
                        'date': date,
                        'stock': stock,
                        'shares': shares,
                        'price': price,
                        'value': shares * price
                    })
        
        print(f"\n回测完成！")
        return self._generate_results()
    
    def _generate_results(self):
        """生成结果"""
        return {
            'positions': self.positions,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'trade_records': self.trade_records,
            'daily_holdings': self.daily_holdings
        }
import os
# ==================== 结果分析模块 ====================
class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, engine, benchmark_data, benchmark_stock):
        self.engine = engine
        self.benchmark = benchmark_data
        self.benchmark_stock = benchmark_stock
        self.results = None
        self.output_dir = None
        
    def set_output_dir(self, output_dir):
        """设置输出目录"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def calculate_metrics(self, returns):
        """计算绩效指标"""
        metrics = {}
        
        # 年化收益率
        days = len(returns)
        years = days / 252
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # 夏普比率（假设无风险利率3%）
        risk_free = 0.03
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # 盈亏比
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        profit_factor = positive_returns.sum() / abs(negative_returns.sum()) if len(negative_returns) > 0 else np.inf
        
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Trading Days': days
        }
        
        return metrics
    
    def analyze(self):
        """分析结果"""
        print("\n分析结果...")
        
        # 计算每日收益率
        portfolio_values = pd.Series(self.engine.portfolio_value).sort_index()
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        # 计算基准收益率
        benchmark_values = self.benchmark.set_index('trade_date')['close_adj']
        benchmark_values = benchmark_values.reindex(portfolio_values.index)
        benchmark_returns = benchmark_values.pct_change().dropna()
        
        # 对齐
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns[common_dates]
        benchmark_returns = benchmark_returns[common_dates]
        
        # 计算超额收益
        excess_returns = portfolio_returns - benchmark_returns
        
        # 计算绩效指标
        portfolio_metrics = self.calculate_metrics(portfolio_returns)
        benchmark_metrics = self.calculate_metrics(benchmark_returns)
        excess_metrics = self.calculate_metrics(excess_returns)
        
        self.results = {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'excess_returns': excess_returns,
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'portfolio_metrics': portfolio_metrics,
            'benchmark_metrics': benchmark_metrics,
            'excess_metrics': excess_metrics
        }
        
        return self.results
    
    def plot_results(self, save_path=None):
        """绘制结果图"""
        if self.results is None:
            self.analyze()
        
        # 创建输出目录
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. 净值曲线
        ax1 = axes[0, 0]
        portfolio_nav = self.results['portfolio_values'] / self.results['portfolio_values'].iloc[0]
        benchmark_nav = self.results['benchmark_values'] / self.results['benchmark_values'].iloc[0]
        ax1.plot(portfolio_nav.index, portfolio_nav, label='Strategy NAV', linewidth=2, color='blue')
        ax1.plot(benchmark_nav.index, benchmark_nav, label=f'Benchmark NAV ({self.benchmark_stock})', linewidth=2, color='orange')
        ax1.set_title('Net Asset Value Curve', fontsize=12)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('NAV')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 超额收益
        ax2 = axes[0, 1]
        excess_nav = (1 + self.results['excess_returns']).cumprod()
        ax2.plot(excess_nav.index, excess_nav, label='Excess Return', color='green', linewidth=2)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Excess Return Curve', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Excess NAV')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 回撤
        ax3 = axes[1, 0]
        cum_returns = (1 + self.results['portfolio_returns']).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns / running_max - 1) * 100
        ax3.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown Curve', fontsize=12)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 收益分布
        ax4 = axes[1, 1]
        ax4.hist(self.results['portfolio_returns'] * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Daily Return Distribution', fontsize=12)
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. 滚动夏普比率
        ax5 = axes[2, 0]
        rolling_sharpe = self.results['portfolio_returns'].rolling(60).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        ax5.plot(rolling_sharpe.index, rolling_sharpe, color='purple', linewidth=2)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('Rolling Sharpe Ratio (60-day)', fontsize=12)
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.grid(True, alpha=0.3)
        
        # 6. 绩效指标表格
        ax6 = axes[2, 1]
        ax6.axis('tight')
        ax6.axis('off')
        
        metrics_data = []
        for key, value in self.results['portfolio_metrics'].items():
            if isinstance(value, float):
                if 'Rate' in key or 'Return' in key or 'Drawdown' in key:
                    metrics_data.append([key, f'{value:.2%}'])
                else:
                    metrics_data.append([key, f'{value:.4f}'])
            else:
                metrics_data.append([key, str(value)])
        
        table = ax6.table(cellText=metrics_data, colLabels=['Metric', 'Value'], 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax6.set_title('Performance Metrics', fontsize=12, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, output_dir):
        """保存所有结果到输出目录"""
        if self.results is None:
            self.analyze()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存绩效指标
        metrics_df = pd.DataFrame({
            'Metric': list(self.results['portfolio_metrics'].keys()),
            'Strategy': list(self.results['portfolio_metrics'].values()),
            'Benchmark': list(self.results['benchmark_metrics'].values()),
            'Excess': list(self.results['excess_metrics'].values())
        })
        metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
        print(f"绩效指标已保存: {os.path.join(output_dir, 'performance_metrics.csv')}")
        
        # 保存净值数据
        nav_df = pd.DataFrame({
            'date': self.results['portfolio_values'].index,
            'portfolio_nav': self.results['portfolio_values'].values,
            'benchmark_nav': self.results['benchmark_values'].values
        })
        nav_df.to_csv(os.path.join(output_dir, 'nav_data.csv'), index=False)
        print(f"净值数据已保存: {os.path.join(output_dir, 'nav_data.csv')}")
        
        # 保存收益率数据
        returns_df = pd.DataFrame({
            'date': self.results['portfolio_returns'].index,
            'portfolio_return': self.results['portfolio_returns'].values,
            'benchmark_return': self.results['benchmark_returns'].values,
            'excess_return': self.results['excess_returns'].values
        })
        returns_df.to_csv(os.path.join(output_dir, 'returns_data.csv'), index=False)
        print(f"收益率数据已保存: {os.path.join(output_dir, 'returns_data.csv')}")
        
        # 绘制并保存图表
        plot_path = os.path.join(output_dir, 'backtest_results.png')
        self.plot_results(save_path=plot_path)
        plt.close()

# ==================== 主程序 ====================
def main():
    """主程序"""
    import os
    import sys
    
    print("="*80)
    print("量化回测系统")
    print("="*80)
    
    # 获取脚本所在目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 配置：脚本所在目录的上级目录下的 final_data 文件夹
    DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'final_data')
    DATA_PATH = os.path.normpath(DATA_PATH)
    
    # 结果输出目录：脚本所在目录下的 result 文件夹
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'result')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    BENCHMARK_STOCK = '000002.SZ'
    
    # 加载数据
    data_loader = DataLoader(DATA_PATH, BENCHMARK_STOCK)
    data = data_loader.load_data()
    
    print(f"\n基准股票: {data_loader.benchmark_stock}")
    
    # 配置回测
    config = Config()
    config.FACTOR_NAME = 'ret_20d'  # 使用20日收益率因子
    config.TOP_N = 200
    config.BENCHMARK_STOCK = data_loader.benchmark_stock
    
    # 运行回测
    engine = BacktestEngine(data, config)
    results = engine.run()
    
    # 分析结果
    analyzer = ResultAnalyzer(engine, data['benchmark'], data_loader.benchmark_stock)
    analyzer.set_output_dir(OUTPUT_DIR)
    metrics = analyzer.analyze()
    
    # 打印结果
    print("\n" + "="*80)
    print("回测结果")
    print("="*80)
    
    print(f"\n基准股票: {data_loader.benchmark_stock}")
    print("\n策略绩效:")
    for key, value in metrics['portfolio_metrics'].items():
        if 'Rate' in key or 'Return' in key or 'Drawdown' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\n基准绩效:")
    for key, value in metrics['benchmark_metrics'].items():
        if 'Rate' in key or 'Return' in key or 'Drawdown' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\n超额收益:")
    for key, value in metrics['excess_metrics'].items():
        if 'Rate' in key or 'Return' in key or 'Drawdown' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 保存结果
    analyzer.save_results(OUTPUT_DIR)
    
    # 保存交易记录
    if engine.trade_records:
        trade_df = pd.DataFrame(engine.trade_records)
        trade_df.to_csv(os.path.join(OUTPUT_DIR, 'trade_records.csv'), index=False)
        print(f"交易记录已保存: {os.path.join(OUTPUT_DIR, 'trade_records.csv')}")
        print(f"总交易次数: {len(trade_df)}")
    else:
        print("\n⚠️ 没有交易记录")
    
    # 保存每日持仓
    if engine.daily_holdings:
        holdings_df = pd.DataFrame(engine.daily_holdings)
        holdings_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_holdings.csv'), index=False)
        print(f"每日持仓已保存: {os.path.join(OUTPUT_DIR, 'daily_holdings.csv')}")
    
    # 保存每日净值
    nav_df = pd.DataFrame({
        'date': list(engine.portfolio_value.keys()),
        'total_value': list(engine.portfolio_value.values()),
        'cash': list(engine.cash.values())
    })
    nav_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_nav.csv'), index=False)
    print(f"每日净值已保存: {os.path.join(OUTPUT_DIR, 'daily_nav.csv')}")
    
    print("\n" + "="*80)
    print(f"所有结果已保存至: {OUTPUT_DIR}")
    print("回测完成！")
    print("="*80)

if __name__ == "__main__":
    main()