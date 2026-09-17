# factors/momentum.py

import numpy as np
import pandas as pd
from core.decorators import tdx_factor


# ==================== 示例1：MACD ====================
@tdx_factor(
    name="MACD",
    category="momentum",
    description="指数平滑异同移动平均（MACD）",
    params={"fast": 12, "slow": 26, "signal": 9},
    tags=["趋势", "动量", "经典"],
    required_columns=['close'],
    min_length=60,
    param_validators={
        'fast': lambda x: isinstance(x, int) and 2 <= x <= 100,
        'slow': lambda x: isinstance(x, int) and 2 <= x <= 200,
        'signal': lambda x: isinstance(x, int) and 2 <= x <= 50,
    }
)
def macd(df, fast=12, slow=26, signal=9):
    """
    MACD指标
    
    通达信公式：
        DIF:EMA(CLOSE,SHORT)-EMA(CLOSE,LONG);
        DEA:EMA(DIF,MID);
        MACD:(DIF-DEA)*2,COLORSTICK;
    """
    close = df['close']
    
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_bar = 2 * (dif - dea)
    
    return pd.DataFrame({
        'DIF': dif,
        'DEA': dea,
        'MACD': macd_bar
    }, index=df.index)


# ==================== 示例2：KDJ ====================
@tdx_factor(
    name="KDJ",
    category="momentum",
    description="随机指标KDJ",
    params={"n": 9, "m1": 3, "m2": 3},
    tags=["超买超卖", "经典"],
    required_columns=['close', 'high', 'low'],
    min_length=30,
    param_validators={
        'n': lambda x: isinstance(x, int) and 2 <= x <= 100,
    }
)
def kdj(df, n=9, m1=3, m2=3):
    """
    KDJ指标
    
    通达信公式：
        RSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
        K:SMA(RSV,M1,1);
        D:SMA(K,M2,1);
        J:3*K-2*D;
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    llv = low.rolling(n).min()
    hhv = high.rolling(n).max()
    
    rsv = (close - llv) / (hhv - llv).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    
    # SMA(X, N, M) = (M*X + (N-M)*Y') / N
    # 等价于 ewm(alpha=M/N, adjust=False)
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return pd.DataFrame({
        'K': k,
        'D': d,
        'J': j
    }, index=df.index)


# ==================== 示例3：DPO ====================
@tdx_factor(
    name="DPO",
    category="trend",
    description="区间震荡线（DPO）",
    params={"n": 20, "m": 6},
    tags=["趋势", "震荡"],
    required_columns=['close'],
    min_length=60,
    param_validators={
        'n': lambda x: isinstance(x, int) and 5 <= x <= 100,
    }
)
def dpo(df, n=20, m=6):
    """
    DPO指标
    
    通达信公式：
        MA20:=MA(CLOSE,N);
        DPO:CLOSE-MA20,COLORWHITE;
        MADPO:MA(DPO,M),COLORYELLOW;
    """
    close = df['close']
    
    ma_n = close.rolling(n).mean()
    dpo_line = close - ma_n.shift(n // 2)  # 通达信中通常偏移N/2
    madpo = dpo_line.rolling(m).mean()
    
    return pd.DataFrame({
        'DPO': dpo_line,
        'MADPO': madpo
    }, index=df.index)


# ==================== 示例4：RSI ====================
@tdx_factor(
    name="RSI",
    category="momentum",
    description="相对强弱指标（RSI）",
    params={"n1": 6, "n2": 12, "n3": 24},
    tags=["超买超卖", "经典"],
    required_columns=['close'],
    min_length=60,
)
def rsi(df, n1=6, n2=12, n3=24):
    """
    RSI指标
    
    通达信公式：
        LC:=REF(CLOSE,1);
        RSI1:SMA(MAX(CLOSE-LC,0),N1,1)/SMA(ABS(CLOSE-LC),N1,1)*100;
        RSI2:SMA(MAX(CLOSE-LC,0),N2,1)/SMA(ABS(CLOSE-LC),N2,1)*100;
        RSI3:SMA(MAX(CLOSE-LC,0),N3,1)/SMA(ABS(CLOSE-LC),N3,1)*100;
    """
    close = df['close']
    lc = close.shift(1)
    
    diff = close - lc
    gain = diff.clip(lower=0)
    abs_diff = diff.abs()
    
    def _rsi(gain_series, abs_series, n):
        # SMA(X, N, 1) 等价于 ewm(alpha=1/N)
        sma_gain = gain_series.ewm(alpha=1/n, adjust=False).mean()
        sma_abs = abs_series.ewm(alpha=1/n, adjust=False).mean()
        return sma_gain / sma_abs.replace(0, np.nan) * 100
    
    rsi1 = _rsi(gain, abs_diff, n1)
    rsi2 = _rsi(gain, abs_diff, n2)
    rsi3 = _rsi(gain, abs_diff, n3)
    
    return pd.DataFrame({
        'RSI1': rsi1,
        'RSI2': rsi2,
        'RSI3': rsi3
    }, index=df.index)


# ==================== 示例5：布林带 ====================
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
    布林带
    
    通达信公式：
        MID:=MA(CLOSE,N);
        UPPER:MID+P*STD(CLOSE,N);
        LOWER:MID-P*STD(CLOSE,N);
    """
    close = df['close']
    
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)  # 通达信用总体标准差
    
    upper = mid + p * std
    lower = mid - p * std
    
    return pd.DataFrame({
        'BOLL_MID': mid,
        'BOLL_UPPER': upper,
        'BOLL_LOWER': lower,
        'BOLL_WIDTH': (upper - lower) / mid,
        'BOLL_POS': (close - lower) / (upper - lower).replace(0, np.nan)
    }, index=df.index)


# ==================== 示例6：ATR ====================
@tdx_factor(
    name="ATR",
    category="volatility",
    description="平均真实波幅（ATR）",
    params={"n": 14},
    tags=["波动率"],
    required_columns=['close', 'high', 'low'],
    min_length=30,
)
def atr(df, n=14):
    """
    ATR指标
    
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
        'ATR': atr_val
    }, index=df.index)


# ==================== 示例7：复杂因子（多步骤） ====================
@tdx_factor(
    name="CCI",
    category="momentum",
    description="顺势指标（CCI）",
    params={"n": 14},
    tags=["超买超卖"],
    required_columns=['close', 'high', 'low'],
    min_length=30,
)
def cci(df, n=14):
    """
    CCI指标
    
    通达信公式：
        TP:=(HIGH+LOW+CLOSE)/3;
        CCI:(TP-MA(TP,N))/(0.015*AVEDEV(TP,N));
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    tp = (high + low + close) / 3
    ma_tp = tp.rolling(n).mean()
    
    # AVEDEV = 平均绝对偏差
    def _avedev(series, n):
        return series.rolling(n).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))),
            raw=True
        )
    
    avedev = _avedev(tp, n)
    
    cci_val = (tp - ma_tp) / (0.015 * avedev).replace(0, np.nan)
    
    return pd.DataFrame({
        'TP': tp,
        'CCI': cci_val
    }, index=df.index)


# ==================== 示例8：复杂因子（含未来函数检测） ====================
@tdx_factor(
    name="OBV",
    category="volume",
    description="能量潮（OBV）",
    params={},
    tags=["量价"],
    required_columns=['close', 'volume'],
    min_length=30,
)
def obv(df):
    """
    OBV指标
    
    通达信公式：
        OBV:SUM(IF(CLOSE>REF(CLOSE,1),VOL,IF(CLOSE<REF(CLOSE,1),-VOL,0)),0);
    """
    close = df['close']
    volume = df['volume']
    
    prev_close = close.shift(1)
    
    direction = np.where(
        close > prev_close, 1,
        np.where(close < prev_close, -1, 0)
    )
    
    obv_val = (direction * volume).cumsum()
    
    return pd.DataFrame({
        'OBV': obv_val
    }, index=df.index)