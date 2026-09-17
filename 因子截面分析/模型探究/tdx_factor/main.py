# main.py

import pandas as pd
import numpy as np
from core.registry import FactorRegistry
from factors import momentum, trend, volatility, volume


# ==================== 1. 加载数据 ====================
def load_data():
    """模拟数据"""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 0.5
    low = close - np.random.rand(n) * 0.5
    open_price = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


# ==================== 2. 单因子计算 ====================
df = load_data()

# 查看已注册因子
print("已注册因子:", FactorRegistry.list_all())
print()

# 查看MACD因子信息
print(FactorRegistry.describe("MACD"))
print()

# 计算MACD
macd_func = FactorRegistry.get_func("MACD")
macd_result = macd_func(df)
print("MACD计算结果:")
print(macd_result.tail())
print()

# 计算KDJ（自定义参数）
kdj_func = FactorRegistry.get_func("KDJ")
kdj_result = kdj_func(df, n=9, m1=3, m2=3)
print("KDJ计算结果:")
print(kdj_result.tail())
print()


# ==================== 3. 批量计算 ====================
factor_names = ["MACD", "KDJ", "RSI", "BOLL", "ATR", "CCI"]
factor_params = {
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "KDJ": {"n": 9, "m1": 3, "m2": 3},
    "RSI": {"n1": 6, "n2": 12, "n3": 24},
    "BOLL": {"n": 20, "p": 2},
    "ATR": {"n": 14},
    "CCI": {"n": 14},
}

all_factors = FactorRegistry.batch_calc(df, factor_names, factor_params)
print("批量计算结果:")
print(all_factors.tail())
print(f"\n因子总数: {all_factors.shape[1]}")


# ==================== 4. 按分类查询 ====================
print("\n按分类查询:")
print(f"动量类: {FactorRegistry.list_by_category('momentum')}")
print(f"趋势类: {FactorRegistry.list_by_category('trend')}")
print(f"波动率类: {FactorRegistry.list_by_category('volatility')}")
print(f"量价类: {FactorRegistry.list_by_category('volume')}")

print("\n按标签查询:")
print(f"经典指标: {FactorRegistry.list_by_tags(['经典'])}")
print(f"超买超卖: {FactorRegistry.list_by_tags(['超买超卖'])}")