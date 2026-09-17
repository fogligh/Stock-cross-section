# core/validator.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import datetime


class FactorValidator:
    """
    因子校验器
    
    功能：
    1. 数据质量校验
    2. 因子有效性校验
    3. 过拟合检测
    4. 未来函数检测
    """
    
    def __init__(self, factor_name: str = "unknown"):
        self.factor_name = factor_name
        self.validation_results = {}
    
    # ==================== 1. 数据质量校验 ====================
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        检查数据质量
        """
        result = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'missing_values': {},
            'nan_ratio': {},
            'inf_count': {},
            'duplicate_index': df.index.duplicated().sum(),
        }
        
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                result['missing_values'][col] = df[col].isna().sum()
                result['nan_ratio'][col] = df[col].isna().sum() / len(df)
                result['inf_count'][col] = np.isinf(df[col]).sum()
        
        return result
    
    # ==================== 2. 因子值校验 ====================
    
    def check_factor_values(self, factor: pd.Series) -> Dict:
        """
        检查因子值是否合理
        """
        result = {
            'name': self.factor_name,
            'count': len(factor),
            'nan_count': factor.isna().sum(),
            'nan_ratio': factor.isna().sum() / len(factor),
            'inf_count': np.isinf(factor).sum(),
            'mean': factor.mean(),
            'std': factor.std(),
            'min': factor.min(),
            'max': factor.max(),
            'skew': factor.skew(),
            'kurtosis': factor.kurtosis(),
        }
        
        # 合理性判断
        warnings = []
        if result['nan_ratio'] > 0.3:
            warnings.append(f"NaN比例过高: {result['nan_ratio']:.2%}")
        if result['std'] == 0:
            warnings.append("标准差为0，因子无变化")
        if abs(result['skew']) > 10:
            warnings.append(f"偏度过大: {result['skew']:.2f}")
        if result['inf_count'] > 0:
            warnings.append(f"存在无穷值: {result['inf_count']}")
        
        result['warnings'] = warnings
        result['passed'] = len(warnings) == 0
        
        return result
    
    # ==================== 3. 未来函数检测 ====================
    
    def check_future_function(self, 
                              df: pd.DataFrame,
                              factor_func: Callable,
                              test_length: int = 50) -> Dict:
        """
        未来函数检测
        
        方法：截取数据到t时刻，计算因子值，与完整数据计算的值对比
        如果不同，说明存在未来函数
        """
        if len(df) < test_length * 2:
            return {'error': '数据长度不足'}
        
        # 完整数据计算结果
        full_factor = factor_func(df)
        if isinstance(full_factor, pd.DataFrame):
            full_factor = full_factor.iloc[:, 0]
        
        # 截取数据计算结果
        truncated_df = df.iloc[:-test_length]
        truncated_factor = factor_func(truncated_df)
        if isinstance(truncated_factor, pd.DataFrame):
            truncated_factor = truncated_factor.iloc[:, 0]
        
        # 对比共同部分
        common_idx = truncated_factor.index
        full_values = full_factor.loc[common_idx]
        truncated_values = truncated_factor
        
        # 计算差异
        diff = (full_values - truncated_values).abs()
        max_diff = diff.max()
        
        # 判断
        has_future_function = max_diff > 1e-6
        
        result = {
            'has_future_function': has_future_function,
            'max_diff': max_diff,
            'test_length': test_length,
            'common_length': len(common_idx),
        }
        
        if has_future_function:
            result['warning'] = f"⚠️ 检测到未来函数！最大差异: {max_diff}"
        else:
            result['message'] = "✅ 无未来函数"
        
        return result
    
    # ==================== 4. IC有效性校验 ====================
    
    def check_ic_validity(self, 
                          ic_series: pd.Series,
                          ic_threshold: float = 0.02,
                          ir_threshold: float = 0.3) -> Dict:
        """
        IC有效性校验
        
        参数:
            ic_series: IC时间序列
            ic_threshold: IC均值阈值
            ir_threshold: IR阈值
        """
        if len(ic_series) < 20:
            return {'error': 'IC序列长度不足'}
        
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ir = mean_ic / std_ic if std_ic > 0 else 0
        
        # t检验
        from scipy.stats import ttest_1samp
        t_stat, p_value = ttest_1samp(ic_series.dropna(), 0)
        
        # IC胜率
        ic_win_rate = (ic_series > 0).mean() if mean_ic > 0 else (ic_series < 0).mean()
        
        result = {
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'ir': ir,
            't_stat': t_stat,
            'p_value': p_value,
            'ic_win_rate': ic_win_rate,
            'ic_positive_ratio': (ic_series > 0).mean(),
            'passed': (abs(mean_ic) > ic_threshold) and (abs(ir) > ir_threshold),
        }
        
        if result['passed']:
            result['message'] = f"✅ 因子有效 (IC={mean_ic:.4f}, IR={ir:.3f})"
        else:
            result['message'] = f"❌ 因子无效 (IC={mean_ic:.4f}, IR={ir:.3f})"
        
        return result
    
    # ==================== 5. 过拟合检测 ====================
    
    def check_overfitting(self,
                          factor_train: pd.Series,
                          factor_test: pd.Series,
                          return_train: pd.Series,
                          return_test: pd.Series) -> Dict:
        """
        过拟合检测
        
        对比训练集和测试集的IC
        """
        from scipy.stats import spearmanr
        
        def _calc_ic(factor, ret):
            valid = factor.notna() & ret.notna()
            if valid.sum() < 10:
                return np.nan
            ic, _ = spearmanr(factor[valid], ret[valid])
            return ic
        
        ic_train = _calc_ic(factor_train, return_train)
        ic_test = _calc_ic(factor_test, return_test)
        
        ic_decay = ic_train - ic_test if not (np.isnan(ic_train) or np.isnan(ic_test)) else np.nan
        
        result = {
            'ic_train': ic_train,
            'ic_test': ic_test,
            'ic_decay': ic_decay,
            'overfitting': ic_decay > 0.03 if not np.isnan(ic_decay) else False,
        }
        
        if result['overfitting']:
            result['message'] = f"⚠️ 存在过拟合 (训练IC={ic_train:.4f}, 测试IC={ic_test:.4f}, 衰减={ic_decay:.4f})"
        else:
            result['message'] = f"✅ 无过拟合 (训练IC={ic_train:.4f}, 测试IC={ic_test:.4f})"
        
        return result
    
    # ==================== 6. 因子稳定性检验 ====================
    
    def check_stability(self, ic_series: pd.Series, window: int = 20) -> Dict:
        """
        因子稳定性检验（滚动IC）
        """
        if len(ic_series) < window * 2:
            return {'error': '数据长度不足'}
        
        rolling_ic = ic_series.rolling(window).mean()
        rolling_std = ic_series.rolling(window).std()
        
        # 计算后半段与前半段的IC差异
        half = len(ic_series) // 2
        first_half_ic = ic_series[:half].mean()
        second_half_ic = ic_series[half:].mean()
        
        result = {
            'rolling_ic_mean': rolling_ic.mean(),
            'rolling_ic_std': rolling_std.mean(),
            'first_half_ic': first_half_ic,
            'second_half_ic': second_half_ic,
            'ic_change': second_half_ic - first_half_ic,
            'stable': abs(second_half_ic - first_half_ic) < 0.02,
        }
        
        return result
    
    # ==================== 7. 生成完整校验报告 ====================
    
    def generate_report(self,
                        df: pd.DataFrame,
                        factor: pd.Series,
                        ic_series: Optional[pd.Series] = None,
                        factor_func: Optional[Callable] = None) -> str:
        """
        生成完整的校验报告
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"因子校验报告: {self.factor_name}")
        lines.append("=" * 70)
        
        # 1. 数据质量
        lines.append("\n【1. 数据质量】")
        data_quality = self.check_data_quality(df)
        lines.append(f"  总行数: {data_quality['total_rows']}")
        lines.append(f"  重复索引: {data_quality['duplicate_index']}")
        for col, ratio in data_quality['nan_ratio'].items():
            if ratio > 0:
                lines.append(f"  {col} NaN比例: {ratio:.2%}")
        
        # 2. 因子值
        lines.append("\n【2. 因子值统计】")
        factor_check = self.check_factor_values(factor)
        lines.append(f"  均值: {factor_check['mean']:.6f}")
        lines.append(f"  标准差: {factor_check['std']:.6f}")
        lines.append(f"  偏度: {factor_check['skew']:.4f}")
        lines.append(f"  峰度: {factor_check['kurtosis']:.4f}")
        lines.append(f"  NaN比例: {factor_check['nan_ratio']:.2%}")
        if factor_check['warnings']:
            for w in factor_check['warnings']:
                lines.append(f"  ⚠️ {w}")
        lines.append(f"  校验结果: {'✅ 通过' if factor_check['passed'] else '❌ 未通过'}")
        
        # 3. 未来函数检测
        if factor_func is not None:
            lines.append("\n【3. 未来函数检测】")
            ff_check = self.check_future_function(df, factor_func)
            if 'error' not in ff_check:
                lines.append(f"  检测结果: {ff_check.get('message', ff_check.get('warning', '未知'))}")
        
        # 4. IC有效性
        if ic_series is not None and len(ic_series) > 20:
            lines.append("\n【4. IC有效性】")
            ic_check = self.check_ic_validity(ic_series)
            if 'error' not in ic_check:
                lines.append(f"  均值IC: {ic_check['mean_ic']:.6f}")
                lines.append(f"  IR: {ic_check['ir']:.4f}")
                lines.append(f"  IC胜率: {ic_check['ic_win_rate']:.2%}")
                lines.append(f"  t统计量: {ic_check['t_stat']:.4f}")
                lines.append(f"  p值: {ic_check['p_value']:.6f}")
                lines.append(f"  {ic_check['message']}")
        
        lines.append("\n" + "=" * 70)
        
        report = "\n".join(lines)
        self.validation_results[self.factor_name] = report
        return report