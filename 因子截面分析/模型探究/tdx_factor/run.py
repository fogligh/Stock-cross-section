# run.py

from tdx_factor import FactorRegistry, FactorEngine, FactorValidator
from tdx_factor.storage import FactorDatabase, FactorCache
import pandas as pd
import numpy as np


# 1. 加载数据
def load_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    open_price = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


# 2. 计算因子
df = load_data()
engine = FactorEngine()

configs = [
    {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
    {"name": "KDJ", "params": {"n": 9, "m1": 3, "m2": 3}},
    {"name": "RSI"},
    {"name": "BOLL"},
    {"name": "ATR"},
    {"name": "DPO"},
    {"name": "CCI"},
]

factors = engine.calc_factors(df, configs)
print(f"计算因子数: {factors.shape[1]}")
print(factors.tail())


# 3. 校验因子
validator = FactorValidator("MACD")
macd_factor = engine.calc_factor(df, "MACD")
report = validator.generate_report(df, macd_factor['DIF'])
print(report)


# 4. 存入数据库
db = FactorDatabase("factors.db")
for col in factors.columns:
    latest_value = factors[col].iloc[-1]
    if not pd.isna(latest_value):
        db.save_factor(
            trade_date=df.index[-1].strftime('%Y-%m-%d'),
            stock_code='000001.SZ',
            factor_name=col,
            value=float(latest_value)
        )

print("✅ 因子已入库")


# 5. 查询
result = db.query_factor("MACD_DIF")
print(result.tail())