"""
因子组合优化 - 最小化最大回撤的权重配置
支持多种优化目标和约束条件
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
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.stats import skew, kurtosis
import time

# ==================== 导入回测框架 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from cross_sectional import BacktestEngine, ResultAnalyzer, Config
    print("✅ 成功从 cross_sectional.py 导入")
except ImportError:
    print("⚠️ 请确保 cross_sectional.py 文件存在")
    # 定义简单的Config类
    class Config:
        TOP_N = 200
        MAX_TURNOVER = 0.30
        FEE_RATE = 0.0014

# ==================== 配置参数 ====================
class OptimizeConfig:
    """优化配置"""
    # 约束条件
    MIN_WEIGHT = 0.001  # 最小权重
    MAX_WEIGHT = 0.25   # 最大权重
    WEIGHT_SUM = 1.0    # 权重和
    
    # 优化目标
    OBJECTIVE = 'maximize_sharpe'  # 'minimize_max_drawdown', 'maximize_sharpe', 'maximize_returns', 'minimize_volatility'
    
    # 优化算法
    ALGORITHM = 'differential_evolution'  # 'differential_evolution', 'dual_annealing', 'SLSQP'
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 回测参数
    TOP_N = 200
    FEE_RATE = 0.0014

# ==================== 因子回测模块 ====================
class FactorBacktest:
    """因子回测 - 计算每个因子的超额收益"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None
        self.factor_cols = []
        self.factor_returns = {}
        self.benchmark_returns = None
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        # 加载数据
        csv_file = self.data_path / "final_200stocks_with_benchmark_2020_2025.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        else:
            parquet_file = self.data_path / "final_200stocks_with_benchmark_2020_2025.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
            else:
                raise FileNotFoundError(f"数据文件不存在")
        
        # 识别因子列
        exclude_cols = ['trade_date', 'ts_code', 'symbol', 'name', 'area', 'industry', 'market',
                       'list_date', 'days_since_listed', 'year', 'month', 'quarter',
                       'open', 'high', 'low', 'close', 'vol', 'amount', 'twap',
                       'close_adj', 'twap_adj', 'open_adj', 'high_adj', 'low_adj', 'accum_factor']
        
        self.factor_cols = [col for col in df.columns if col not in exclude_cols]
        self.data = df
        
        print(f"数据加载完成，形状: {df.shape}")
        print(f"因子数量: {len(self.factor_cols)}")
        
        return df
    
    def calculate_factor_excess_returns(self, factor_name):
        """
        计算单个因子的超额收益
        
        Parameters:
        -----------
        factor_name: str, 因子名称
        
        Returns:
        --------
        excess_returns: Series, 每日超额收益
        """
        print(f"  计算因子超额收益: {factor_name}")
        
        # 准备数据
        factor_data = self.data.pivot(index='trade_date', columns='ts_code', values=factor_name)
        price_data = self.data.pivot(index='trade_date', columns='ts_code', values='close_adj')
        twap_data = self.data.pivot(index='trade_date', columns='ts_code', values='twap_adj')
        accum_data = self.data.pivot(index='trade_date', columns='ts_code', values='accum_factor')
        
        # 获取ST信息
        st_data = self.data[['trade_date', 'ts_code', 'name']].copy()
        st_data['is_st'] = st_data['name'].str.contains('ST|\\*ST', case=False, na=False)
        st_pivot = st_data.pivot(index='trade_date', columns='ts_code', values='is_st').fillna(False)
        
        # 获取基准数据（使用市场平均）
        benchmark_stock = '000002.SZ'
        benchmark_returns = self.data[self.data['ts_code'] == benchmark_stock].set_index('trade_date')['close_adj'].pct_change()
        
        # 获取日期
        dates = sorted(price_data.index)
        
        # 计算每日因子收益率（做多因子值高的股票）
        portfolio_returns = []
        
        for i, date in enumerate(dates):
            if i == 0:
                continue
            
            # 获取前一天的因子值
            prev_date = dates[i-1]
            factor_values = factor_data.loc[prev_date].dropna()
            
            if factor_values.empty:
                continue
            
            # 获取可交易股票
            tradable = []
            for stock in factor_values.index:
                # 检查ST
                if stock in st_pivot.columns and st_pivot.loc[date, stock]:
                    continue
                # 检查价格
                if stock in twap_data.columns:
                    price = twap_data.loc[date, stock]
                    if pd.isna(price) or price <= 0:
                        continue
                tradable.append(stock)
            
            # 选择因子值最高的TOP_N股票
            factor_scores = factor_values[factor_values.index.isin(tradable)]
            top_n = min(OptimizeConfig.TOP_N, len(factor_scores))
            top_stocks = factor_scores.nlargest(top_n).index.tolist()
            
            if not top_stocks:
                continue
            
            # 等权买入
            portfolio_return = 0
            for stock in top_stocks:
                if stock in twap_data.columns:
                    price_today = twap_data.loc[date, stock]
                    price_yesterday = twap_data.loc[prev_date, stock]
                    if not pd.isna(price_today) and not pd.isna(price_yesterday) and price_yesterday > 0:
                        stock_return = price_today / price_yesterday - 1
                        portfolio_return += stock_return / len(top_stocks)
            
            # 减去交易费用（双边）
            portfolio_return = portfolio_return * (1 - OptimizeConfig.FEE_RATE * 2)
            portfolio_returns.append(portfolio_return)
        
        # 构建超额收益序列
        excess_returns = pd.Series(portfolio_returns, index=dates[1:])
        
        # 减去基准收益
        benchmark_aligned = benchmark_returns.reindex(excess_returns.index).fillna(0)
        excess_returns = excess_returns - benchmark_aligned
        
        return excess_returns
    
    def calculate_all_factor_excess_returns(self, factor_list=None):
        """
        计算所有因子的超额收益
        
        Parameters:
        -----------
        factor_list: list, 因子列表（如果为None，使用所有因子）
        
        Returns:
        --------
        factor_returns_df: DataFrame, 所有因子的超额收益
        """
        if factor_list is None:
            factor_list = self.factor_cols
        
        print(f"\n计算 {len(factor_list)} 个因子的超额收益...")
        
        factor_returns_dict = {}
        for i, factor in enumerate(factor_list):
            print(f"  [{i+1}/{len(factor_list)}] {factor}")
            try:
                excess_returns = self.calculate_factor_excess_returns(factor)
                if len(excess_returns) > 0:
                    factor_returns_dict[factor] = excess_returns
            except Exception as e:
                print(f"    计算失败: {e}")
                continue
        
        self.factor_returns = pd.DataFrame(factor_returns_dict)
        
        print(f"\n成功计算 {len(self.factor_returns.columns)} 个因子的超额收益")
        print(f"日期范围: {self.factor_returns.index.min()} 到 {self.factor_returns.index.max()}")
        
        return self.factor_returns

# ==================== 优化求解器 ====================
class PortfolioOptimizer:
    """投资组合优化器 - 最小化最大回撤"""
    
    def __init__(self, factor_returns):
        """
        初始化优化器
        
        Parameters:
        -----------
        factor_returns: DataFrame, 各因子的超额收益
        """
        self.factor_returns = factor_returns
        self.n_factors = len(factor_returns.columns)
        self.factor_names = factor_returns.columns.tolist()
        
        # 优化结果
        self.optimal_weights = None
        self.optimal_metrics = {}
        self.optimization_history = []
        
    def calculate_portfolio_returns(self, weights):
        """
        计算投资组合收益
        
        Parameters:
        -----------
        weights: array, 权重向量
        
        Returns:
        --------
        portfolio_returns: Series, 投资组合每日收益
        """
        # 确保权重归一化
        weights = np.array(weights) / np.sum(weights)
        
        # 计算组合收益
        portfolio_returns = self.factor_returns.dot(weights)
        
        return portfolio_returns
    
    def calculate_portfolio_metrics(self, weights):
        """
        计算投资组合指标
        
        Parameters:
        -----------
        weights: array, 权重向量
        
        Returns:
        --------
        metrics: dict, 各项指标
        """
        portfolio_returns = self.calculate_portfolio_returns(weights)
        
        # 计算净值
        nav = (1 + portfolio_returns).cumprod()
        
        # 年化收益率
        days = len(portfolio_returns)
        years = days / 252
        total_return = nav.iloc[-1] - 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 年化波动率
        annual_vol = portfolio_returns.std() * np.sqrt(252) if len(portfolio_returns) > 0 else 0
        
        # 夏普比率（无风险利率3%）
        risk_free = 0.03
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        running_max = nav.expanding().max()
        drawdown = (nav / running_max - 1)
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
        
        # 盈亏比
        positive = portfolio_returns[portfolio_returns > 0]
        negative = portfolio_returns[portfolio_returns < 0]
        profit_factor = positive.sum() / abs(negative.sum()) if len(negative) > 0 else np.inf
        
        # 信息比率（相对等权）
        equal_weights = np.ones(self.n_factors) / self.n_factors
        equal_returns = self.calculate_portfolio_returns(equal_weights)
        excess_returns = portfolio_returns - equal_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'information_ratio': information_ratio,
            'nav': nav,
            'returns': portfolio_returns
        }
        
        return metrics
    
    def objective_minimize_max_drawdown(self, weights):
        """
        目标函数：最小化最大回撤
        
        Parameters:
        -----------
        weights: array, 权重向量
        
        Returns:
        --------
        objective: float, 目标函数值
        """
        # 确保权重非负且和为1
        weights = np.array(weights)
        weights = np.clip(weights, OptimizeConfig.MIN_WEIGHT, OptimizeConfig.MAX_WEIGHT)
        weights = weights / np.sum(weights)
        
        metrics = self.calculate_portfolio_metrics(weights)
        
        # 记录优化历史
        self.optimization_history.append({
            'weights': weights.copy(),
            'max_drawdown': metrics['max_drawdown'],
            'sharpe': metrics['sharpe_ratio'],
            'annual_return': metrics['annual_return']
        })
        
        # 最小化最大回撤（取负值因为scipy最小化）
        return -metrics['max_drawdown']
    
    def objective_maximize_sharpe(self, weights):
        """
        目标函数：最大化夏普比率
        
        Parameters:
        -----------
        weights: array, 权重向量
        
        Returns:
        --------
        objective: float, 目标函数值
        """
        weights = np.array(weights)
        weights = np.clip(weights, OptimizeConfig.MIN_WEIGHT, OptimizeConfig.MAX_WEIGHT)
        weights = weights / np.sum(weights)
        
        metrics = self.calculate_portfolio_metrics(weights)
        
        self.optimization_history.append({
            'weights': weights.copy(),
            'max_drawdown': metrics['max_drawdown'],
            'sharpe': metrics['sharpe_ratio'],
            'annual_return': metrics['annual_return']
        })
        
        # 最大化夏普比率（取负值）
        return -metrics['sharpe_ratio']
    
    def objective_maximize_returns(self, weights):
        """目标函数：最大化年化收益率"""
        weights = np.array(weights)
        weights = np.clip(weights, OptimizeConfig.MIN_WEIGHT, OptimizeConfig.MAX_WEIGHT)
        weights = weights / np.sum(weights)
        
        metrics = self.calculate_portfolio_metrics(weights)
        
        self.optimization_history.append({
            'weights': weights.copy(),
            'max_drawdown': metrics['max_drawdown'],
            'sharpe': metrics['sharpe_ratio'],
            'annual_return': metrics['annual_return']
        })
        
        return -metrics['annual_return']
    
    def objective_minimize_volatility(self, weights):
        """目标函数：最小化波动率"""
        weights = np.array(weights)
        weights = np.clip(weights, OptimizeConfig.MIN_WEIGHT, OptimizeConfig.MAX_WEIGHT)
        weights = weights / np.sum(weights)
        
        metrics = self.calculate_portfolio_metrics(weights)
        
        self.optimization_history.append({
            'weights': weights.copy(),
            'max_drawdown': metrics['max_drawdown'],
            'sharpe': metrics['sharpe_ratio'],
            'annual_return': metrics['annual_return']
        })
        
        return metrics['annual_volatility']
    
    def optimize(self):
        """
        执行优化
        
        Returns:
        --------
        result: dict, 优化结果
        """
        print(f"\n{'='*60}")
        print("开始投资组合优化")
        print(f"{'='*60}")
        print(f"因子数量: {self.n_factors}")
        print(f"优化目标: {OptimizeConfig.OBJECTIVE}")
        print(f"优化算法: {OptimizeConfig.ALGORITHM}")
        print(f"权重范围: [{OptimizeConfig.MIN_WEIGHT}, {OptimizeConfig.MAX_WEIGHT}]")
        
        # 初始权重（等权）
        initial_weights = np.ones(self.n_factors) / self.n_factors
        
        # 权重边界
        bounds = [(OptimizeConfig.MIN_WEIGHT, OptimizeConfig.MAX_WEIGHT)] * self.n_factors
        
        # 约束条件：权重和为1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # 选择目标函数
        if OptimizeConfig.OBJECTIVE == 'minimize_max_drawdown':
            objective = self.objective_minimize_max_drawdown
        elif OptimizeConfig.OBJECTIVE == 'maximize_sharpe':
            objective = self.objective_maximize_sharpe
        elif OptimizeConfig.OBJECTIVE == 'maximize_returns':
            objective = self.objective_maximize_returns
        elif OptimizeConfig.OBJECTIVE == 'minimize_volatility':
            objective = self.objective_minimize_volatility
        else:
            raise ValueError(f"不支持的目标函数: {OptimizeConfig.OBJECTIVE}")
        
        start_time = time.time()
        
        # 选择优化算法
        if OptimizeConfig.ALGORITHM == 'differential_evolution':
            # 差分进化算法 - 适合非凸问题
            result = differential_evolution(
                objective,
                bounds,
                strategy='best1bin',
                maxiter=1000,
                popsize=15,
                tol=1e-6,
                seed=OptimizeConfig.RANDOM_SEED,
                workers=1  # 单进程
            )
        elif OptimizeConfig.ALGORITHM == 'dual_annealing':
            # 模拟退火 - 适合非凸问题
            result = dual_annealing(
                objective,
                bounds,
                maxiter=1000,
                seed=OptimizeConfig.RANDOM_SEED
            )
        else:
            # SLSQP - 局部优化
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
        
        elapsed_time = time.time() - start_time
        
        if result.success:
            optimal_weights = result.x
            optimal_weights = optimal_weights / np.sum(optimal_weights)  # 确保和为1
            
            print(f"\n✅ 优化成功!")
            print(f"耗时: {elapsed_time:.2f} 秒")
            print(f"迭代次数: {result.nit if hasattr(result, 'nit') else 'N/A'}")
            
            # 计算最优组合的指标
            metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            self.optimal_weights = optimal_weights
            self.optimal_metrics = metrics
            
            # 打印结果
            self.print_optimization_results(optimal_weights, metrics)
            
            return {
                'weights': optimal_weights,
                'metrics': metrics,
                'success': True,
                'message': result.message
            }
        else:
            print(f"\n❌ 优化失败: {result.message}")
            return {
                'weights': None,
                'metrics': None,
                'success': False,
                'message': result.message
            }
    
    def print_optimization_results(self, weights, metrics):
        """打印优化结果"""
        print(f"\n{'='*60}")
        print("优化结果")
        print(f"{'='*60}")
        
        print("\n最优权重:")
        # 显示权重最大的前10个因子
        weight_df = pd.DataFrame({
            'factor': self.factor_names,
            'weight': weights
        }).sort_values('weight', ascending=False)
        
        for i, row in weight_df.head(10).iterrows():
            print(f"  {row['factor']:20s}: {row['weight']:.4f} ({row['weight']*100:.2f}%)")
        if len(weight_df) > 10:
            print(f"  ... 还有 {len(weight_df) - 10} 个因子")
        
        print(f"\n投资组合指标:")
        print(f"  总收益率: {metrics['total_return']:.2%}")
        print(f"  年化收益率: {metrics['annual_return']:.2%}")
        print(f"  年化波动率: {metrics['annual_volatility']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  胜率: {metrics['win_rate']:.2%}")
        print(f"  盈亏比: {metrics['profit_factor']:.2f}")
        print(f"  信息比率: {metrics['information_ratio']:.4f}")
    
    def plot_optimization_results(self, save_path=None):
        """绘制优化结果"""
        if self.optimal_weights is None:
            print("请先运行优化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 权重分布
        ax1 = axes[0, 0]
        weight_df = pd.DataFrame({
            'factor': self.factor_names,
            'weight': self.optimal_weights
        }).sort_values('weight', ascending=False)
        
        colors = ['green' if w > 0.02 else 'blue' for w in weight_df['weight']]
        ax1.barh(weight_df['factor'].head(20), weight_df['weight'].head(20), color=colors[:20])
        ax1.set_title('Top 20 Factor Weights', fontsize=12)
        ax1.set_xlabel('Weight')
        ax1.set_ylabel('Factor')
        
        # 2. 净值曲线
        ax2 = axes[0, 1]
        nav = self.optimal_metrics['nav']
        # 等权组合
        equal_weights = np.ones(self.n_factors) / self.n_factors
        equal_metrics = self.calculate_portfolio_metrics(equal_weights)
        
        ax2.plot(nav.index, nav, label='Optimal Portfolio', linewidth=2, color='blue')
        ax2.plot(nav.index, equal_metrics['nav'], label='Equal Weight', linewidth=2, color='orange', alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative Returns', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('NAV')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 回撤曲线
        ax3 = axes[1, 0]
        nav_opt = self.optimal_metrics['nav']
        running_max = nav_opt.expanding().max()
        drawdown = (nav_opt / running_max - 1) * 100
        
        ax3.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown Curve', fontsize=12)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 优化历史
        ax4 = axes[1, 1]
        if self.optimization_history:
            history_df = pd.DataFrame(self.optimization_history)
            ax4.scatter(history_df['sharpe'], history_df['max_drawdown'], alpha=0.5, s=10)
            ax4.scatter(history_df['sharpe'].iloc[-1], history_df['max_drawdown'].iloc[-1], 
                       color='red', s=100, marker='*', label='Optimal')
            ax4.set_xlabel('Sharpe Ratio')
            ax4.set_ylabel('Max Drawdown')
            ax4.set_title('Optimization History', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        plt.show()
        return fig

# ==================== 主程序 ====================
def main():
    """主程序"""
    print("="*80)
    print("因子组合优化 - 最小化最大回撤")
    print("="*80)
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, '..', 'final_data')
    if not os.path.exists(DATA_PATH):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(script_dir, '..', 'final_data')
    
    DATA_PATH = os.path.normpath(DATA_PATH)
    OUTPUT_PATH = os.path.join(current_dir, 'optimization_results')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print(f"数据路径: {DATA_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    
    # 1. 加载数据并计算因子超额收益
    factor_backtest = FactorBacktest(DATA_PATH)
    data = factor_backtest.load_data()
    
    # 选择一部分因子进行优化（避免计算量过大）
    # 可以调整这里选择因子的数量
    selected_factors = factor_backtest.factor_cols[:20]  # 取前20个因子
    print(f"\n选择 {len(selected_factors)} 个因子进行优化")
    
    # 计算因子超额收益
    factor_returns = factor_backtest.calculate_all_factor_excess_returns(selected_factors)
    
    if factor_returns.empty:
        print("❌ 没有可用的因子超额收益数据")
        return
    
    # 2. 执行优化
    optimizer = PortfolioOptimizer(factor_returns)
    result = optimizer.optimize()
    
    if not result['success']:
        print("❌ 优化失败")
        return
    
    # 3. 绘制结果
    plot_path = os.path.join(OUTPUT_PATH, 'optimization_results.png')
    optimizer.plot_optimization_results(save_path=plot_path)
    
    # 4. 保存结果
    # 保存权重
    weight_df = pd.DataFrame({
        'factor': optimizer.factor_names,
        'weight': optimizer.optimal_weights
    }).sort_values('weight', ascending=False)
    weight_df.to_csv(os.path.join(OUTPUT_PATH, 'optimal_weights.csv'), index=False)
    
    # 保存指标
    metrics_df = pd.DataFrame([{
        'metric': k,
        'value': v
    } for k, v in optimizer.optimal_metrics.items() if not isinstance(v, (pd.Series, pd.DataFrame))])
    metrics_df.to_csv(os.path.join(OUTPUT_PATH, 'optimal_metrics.csv'), index=False)
    
    # 保存净值曲线
    nav_df = pd.DataFrame({
        'date': optimizer.optimal_metrics['nav'].index,
        'nav': optimizer.optimal_metrics['nav'].values
    })
    nav_df.to_csv(os.path.join(OUTPUT_PATH, 'optimal_nav.csv'), index=False)
    
    print(f"\n✅ 所有结果已保存至: {OUTPUT_PATH}")
    print("="*80)

if __name__ == "__main__":
    main()