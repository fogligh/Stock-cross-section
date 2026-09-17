# core/engine.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .registry import FactorRegistry


class FactorEngine:
    """
    因子计算引擎
    
    功能：
    1. 单股票因子计算
    2. 多股票批量计算
    3. 因子正交化
    4. 因子标准化
    5. 因子中性化
    """
    
    def __init__(self, data_source: Optional[Callable] = None):
        """
        参数:
            data_source: 数据源函数，签名 func(stock, end_date, count) -> DataFrame
                         如果不提供，需要手动传入df
        """
        self.data_source = data_source
        self._cache = {}
    
    # ==================== 1. 单因子计算 ====================
    
    def calc_factor(self, 
                    df: pd.DataFrame,
                    factor_name: str,
                    params: Optional[Dict] = None) -> pd.Series:
        """
        计算单个因子
        
        参数:
            df: 输入数据（含OHLCV）
            factor_name: 因子名称
            params: 因子参数
        
        返回:
            因子值序列
        """
        func = FactorRegistry.get_func(factor_name)
        params = params or {}
        
        result = func(df, **params)
        
        # 统一处理返回值
        if isinstance(result, pd.Series):
            return result
        elif isinstance(result, pd.DataFrame):
            # 如果返回多列，取第一列或合并
            if len(result.columns) == 1:
                return result.iloc[:, 0]
            else:
                # 多列情况，返回DataFrame
                return result
        elif isinstance(result, np.ndarray):
            return pd.Series(result, index=df.index)
        else:
            raise TypeError(f"因子 {factor_name} 返回类型不支持: {type(result)}")
    
    # ==================== 2. 批量计算 ====================
    
    def calc_factors(self,
                     df: pd.DataFrame,
                     factor_configs: List[Dict]) -> pd.DataFrame:
        """
        批量计算多个因子
        
        参数:
            df: 输入数据
            factor_configs: 因子配置列表
                [
                    {"name": "MACD", "params": {"fast": 12, "slow": 26}},
                    {"name": "KDJ", "params": {"n": 9}},
                ]
        
        返回:
            包含所有因子的DataFrame
        """
        results = {}
        
        for config in factor_configs:
            name = config['name']
            params = config.get('params', {})
            alias = config.get('alias', name)
            
            try:
                result = self.calc_factor(df, name, params)
                
                if isinstance(result, pd.Series):
                    results[alias] = result
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        results[f"{alias}_{col}"] = result[col]
                        
            except Exception as e:
                print(f"因子 {name} 计算失败: {e}")
                continue
        
        return pd.DataFrame(results, index=df.index)
    
    # ==================== 3. 多股票批量计算 ====================
    
    def calc_factors_multi_stock(self,
                                  stock_list: List[str],
                                  end_date: str,
                                  count: int,
                                  factor_configs: List[Dict]) -> pd.DataFrame:
        """
        多股票批量计算因子（面板数据）
        
        返回:
            DataFrame，MultiIndex (date, stock)
        """
        all_results = []
        
        for stock in stock_list:
            try:
                if self.data_source:
                    df = self.data_source(stock, end_date, count)
                else:
                    raise ValueError("未提供data_source")
                
                if df is None or len(df) < 30:
                    continue
                
                factors = self.calc_factors(df, factor_configs)
                factors['stock'] = stock
                factors['date'] = factors.index
                all_results.append(factors)
                
            except Exception as e:
                print(f"股票 {stock} 计算失败: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        
        # 合并
        result = pd.concat(all_results, ignore_index=True)
        result = result.set_index(['date', 'stock'])
        
        return result
    
    # ==================== 4. 因子标准化 ====================
    
    @staticmethod
    def standardize(factor: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        因子标准化
        
        参数:
            factor: 因子值
            method: 'zscore' 或 'rank' 或 'minmax'
        """
        if method == 'zscore':
            return (factor - factor.mean()) / factor.std()
        elif method == 'rank':
            return factor.rank(pct=True)
        elif method == 'minmax':
            return (factor - factor.min()) / (factor.max() - factor.min())
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    # ==================== 5. 因子去极值 ====================
    
    @staticmethod
    def winsorize(factor: pd.Series, 
                  method: str = 'mad',
                  n: float = 3) -> pd.Series:
        """
        因子去极值
        
        参数:
            method: 'mad' 或 'sigma' 或 'quantile'
            n: 倍数或分位数
        """
        if method == 'mad':
            # MAD法
            median = factor.median()
            mad = (factor - median).abs().median()
            upper = median + n * 1.4826 * mad
            lower = median - n * 1.4826 * mad
            return factor.clip(lower, upper)
        
        elif method == 'sigma':
            # 3σ法
            mean = factor.mean()
            std = factor.std()
            upper = mean + n * std
            lower = mean - n * std
            return factor.clip(lower, upper)
        
        elif method == 'quantile':
            # 分位数法
            lower = factor.quantile(n)
            upper = factor.quantile(1 - n)
            return factor.clip(lower, upper)
        
        else:
            raise ValueError(f"不支持的去极值方法: {method}")
    
    # ==================== 6. 因子中性化 ====================
    
    @staticmethod
    def neutralize(factor: pd.Series,
                   industry: Optional[pd.Series] = None,
                   market_cap: Optional[pd.Series] = None) -> pd.Series:
        """
        因子中性化（行业中性化 + 市值中性化）
        
        参数:
            factor: 因子值
            industry: 行业哑变量（DataFrame，列为行业）
            market_cap: 市值序列
        """
        import statsmodels.api as sm
        
        X = pd.DataFrame(index=factor.index)
        
        if industry is not None:
            if isinstance(industry, pd.Series):
                # 转为哑变量
                industry_dummies = pd.get_dummies(industry, prefix='ind')
                X = pd.concat([X, industry_dummies], axis=1)
            else:
                X = pd.concat([X, industry], axis=1)
        
        if market_cap is not None:
            # 市值取对数
            X['log_mcap'] = np.log(market_cap)
        
        if X.empty:
            return factor
        
        # 添加常数项
        X = sm.add_constant(X)
        
        # OLS回归
        valid_idx = factor.notna() & X.notna().all(axis=1)
        y = factor[valid_idx]
        X_valid = X[valid_idx]
        
        if len(y) < 10:
            return factor
        
        try:
            model = sm.OLS(y, X_valid).fit()
            residual = model.resid
            result = pd.Series(np.nan, index=factor.index)
            result[valid_idx] = residual
            return result
        except Exception as e:
            print(f"中性化失败: {e}")
            return factor
    
    # ==================== 7. 因子处理流水线 ====================
    
    def process_factor(self,
                       factor: pd.Series,
                       winsorize_method: str = 'mad',
                       standardize_method: str = 'zscore',
                       industry: Optional[pd.Series] = None,
                       market_cap: Optional[pd.Series] = None) -> pd.Series:
        """
        因子处理流水线：去极值 → 中性化 → 标准化
        """
        # 1. 去极值
        factor = self.winsorize(factor, method=winsorize_method)
        
        # 2. 中性化
        if industry is not None or market_cap is not None:
            factor = self.neutralize(factor, industry, market_cap)
        
        # 3. 标准化
        factor = self.standardize(factor, method=standardize_method)
        
        return factor
    
    # ==================== 8. IC计算 ====================
    
    @staticmethod
    def calc_ic(factor: pd.Series, 
                forward_return: pd.Series,
                method: str = 'spearman') -> float:
        """
        计算IC（信息系数）
        """
        from scipy.stats import spearmanr, pearsonr
        
        valid = factor.notna() & forward_return.notna()
        if valid.sum() < 10:
            return np.nan
        
        f = factor[valid]
        r = forward_return[valid]
        
        if method == 'spearman':
            ic, _ = spearmanr(f, r)
        else:
            ic, _ = pearsonr(f, r)
        
        return ic
    
    @staticmethod
    def calc_ic_series(factor_panel: pd.DataFrame,
                       return_panel: pd.DataFrame,
                       method: str = 'spearman') -> pd.Series:
        """
        计算IC序列（时间序列）
        
        参数:
            factor_panel: MultiIndex (date, stock) 的因子面板
            return_panel: MultiIndex (date, stock) 的收益面板
        """
        ic_list = []
        dates = factor_panel.index.get_level_values(0).unique()
        
        for date in dates:
            try:
                f = factor_panel.loc[date]
                r = return_panel.loc[date]
                
                valid = f.notna() & r.notna()
                if valid.sum() < 10:
                    continue
                
                ic = FactorEngine.calc_ic(f[valid], r[valid], method)
                ic_list.append({'date': date, 'IC': ic})
            except:
                continue
        
        return pd.DataFrame(ic_list).set_index('date')['IC']
    
    # ==================== 9. 因子相关性 ====================
    
    @staticmethod
    def calc_correlation(factor_panel: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子相关性矩阵
        """
        return factor_panel.corr(method='spearman')
    
    # ==================== 10. 分层回测 ====================
    
    @staticmethod
    def stratified_backtest(factor: pd.Series,
                            forward_return: pd.Series,
                            n_groups: int = 10) -> pd.DataFrame:
        """
        分层回测
        
        参数:
            factor: 因子值（横截面）
            forward_return: 未来收益
            n_groups: 分组数
        
        返回:
            各分组的平均收益
        """
        df = pd.DataFrame({'factor': factor, 'return': forward_return}).dropna()
        
        if len(df) < n_groups:
            return pd.DataFrame()
        
        df['group'] = pd.qcut(df['factor'], n_groups, labels=False, duplicates='drop')
        
        group_returns = df.groupby('group')['return'].agg(['mean', 'std', 'count'])
        group_returns['t_stat'] = group_returns['mean'] / (group_returns['std'] / np.sqrt(group_returns['count']))
        
        return group_returns