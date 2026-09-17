# factors/volatility.py

import numpy as np
import pandas as pd
from core.decorators import tdx_factor


# ==================== ATR ====================
@tdx_factor(
    name="ATR",
    category="volatility",
    description="平均真实波幅（ATR）",
    params={"n": 14},
    tags=["波动率", "经典"],
    required_columns=['close', 'high', 'low'],
    min_length=30,
)
def atr(df, n=14):
    """
    通达信公式：
        TR:=MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(LOW-REF(CLOSE,1)));
        ATR:MA(TR,N);
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(n).mean()
    
    return pd.DataFrame({
        'TR': tr,
        'ATR': atr_val,
        'ATR_PCT': atr_val / close
    }, index=df.index)


# ==================== BOLL ====================
@tdx_factor(
    name="BOLL",
    category="volatility",
    description="布林带（Bollinger Bands）",
    params={"n": 20, "p": 2},
    tags=["波动率", "经典"],
    required_columns=['close'],
    min_length=60,
)
def boll(df, n=20, p=2):
    """
    通达信公式：
        MID:=MA(CLOSE,N);
        UPPER:MID+P*STD(CLOSE,N);
        LOWER:MID-P*STD(CLOSE,N);
    """
    close = df['close']
    
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    
    upper = mid + p * std
    lower = mid - p * std
    
    return pd.DataFrame({
        'BOLL_MID': mid,
        'BOLL_UPPER': upper,
        'BOLL_LOWER': lower,
        'BOLL_WIDTH': (upper - lower) / mid,
        'BOLL_POS': (close - lower) / (upper - lower).replace(0, np.nan)
    }, index=df.index)


# ==================== 历史波动率 ====================
@tdx_factor(
    name="HV",
    category="volatility",
    description="历史波动率（Historical Volatility）",
    params={"n": 20, "annualize": True},
    tags=["波动率"],
    required_columns=['close'],
    min_length=30,
)
def hv(df, n=20, annualize=True):
    """
    历史波动率 = 收益率标准差 × sqrt(252)
    """
    close = df['close']
    
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(n).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return pd.DataFrame({
        'HV': vol,
        'RET': ret
    }, index=df.index)


# ==================== 真实波幅 ====================
@tdx_factor(
    name="TR",
    category="volatility",
    description="真实波幅（True Range）",
    params={},
    tags=["波动率"],
    required_columns=['close', 'high', 'low'],
    min_length=30,
)
def tr(df):
    """真实波幅"""
    close = df['close']
    high = df['high']
    low = df['low']
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr_val = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return pd.DataFrame({
        'TR': tr_val,
        'TR_PCT': tr_val / close
    }, index=df.index)


# ==================== 波动率变化 ====================
@tdx_factor(
    name="VOL_CHANGE",
    category="volatility",
    description="波动率变化率",
    params={"n": 20},
    tags=["波动率"],
    required_columns=['close'],
    min_length=60,
)
def vol_change(df, n=20):
    """
    波动率变化：当前波动率 / 历史波动率
    """
    close = df['close']
    ret = close.pct_change()
    
    short_vol = ret.rolling(n).std()
    long_vol = ret.rolling(n * 3).std()
    
    vol_ratio = short_vol / long_vol.replace(0, np.nan)
    
    return pd.DataFrame({
        'SHORT_VOL': short_vol,
        'LONG_VOL': long_vol,
        'VOL_RATIO': vol_ratio
    }, index=df.index)