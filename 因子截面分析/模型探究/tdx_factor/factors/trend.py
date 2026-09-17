# factors/trend.py

import numpy as np
import pandas as pd
from core.decorators import tdx_factor


# ==================== DPO ====================
@tdx_factor(
    name="DPO",
    category="trend",
    description="区间震荡线（Detrended Price Oscillator）",
    params={"n": 20, "m": 6},
    tags=["趋势", "震荡"],
    required_columns=['close'],
    min_length=60,
)
def dpo(df, n=20, m=6):
    """
    通达信公式：
        MA20:=MA(CLOSE,N);
        DPO:CLOSE-MA20,COLORWHITE;
        MADPO:MA(DPO,M),COLORYELLOW;
    """
    close = df['close']
    
    ma_n = close.rolling(n).mean()
    dpo_line = close - ma_n.shift(n // 2)
    madpo = dpo_line.rolling(m).mean()
    
    return pd.DataFrame({
        'DPO': dpo_line,
        'MADPO': madpo
    }, index=df.index)


# ==================== BBI ====================
@tdx_factor(
    name="BBI",
    category="trend",
    description="多空指标（Bull and Bear Index）",
    params={"n1": 3, "n2": 6, "n3": 12, "n4": 24},
    tags=["趋势", "多空"],
    required_columns=['close'],
    min_length=30,
)
def bbi(df, n1=3, n2=6, n3=12, n4=24):
    """
    通达信公式：
        BBI:(MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,12)+MA(CLOSE,24))/4;
    """
    close = df['close']
    
    ma1 = close.rolling(n1).mean()
    ma2 = close.rolling(n2).mean()
    ma3 = close.rolling(n3).mean()
    ma4 = close.rolling(n4).mean()
    
    bbi_line = (ma1 + ma2 + ma3 + ma4) / 4
    
    return pd.DataFrame({
        'BBI': bbi_line
    }, index=df.index)


# ==================== TRIX ====================
@tdx_factor(
    name="TRIX",
    category="trend",
    description="三重指数平滑平均线（TRIX）",
    params={"n": 12, "m": 9},
    tags=["趋势", "长线"],
    required_columns=['close'],
    min_length=60,
)
def trix(df, n=12, m=9):
    """
    通达信公式：
        TR:=EMA(EMA(EMA(CLOSE,N),N),N);
        TRIX:(TR-REF(TR,1))/REF(TR,1)*100;
        TRMA:MA(TRIX,M);
    """
    close = df['close']
    
    ema1 = close.ewm(span=n, adjust=False).mean()
    ema2 = ema1.ewm(span=n, adjust=False).mean()
    ema3 = ema2.ewm(span=n, adjust=False).mean()
    
    trix_line = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trma = trix_line.rolling(m).mean()
    
    return pd.DataFrame({
        'TRIX': trix_line,
        'TRMA': trma
    }, index=df.index)


# ==================== DMA ====================
@tdx_factor(
    name="DMA",
    category="trend",
    description="平行线差指标（DMA）",
    params={"n1": 10, "n2": 50, "m": 10},
    tags=["趋势"],
    required_columns=['close'],
    min_length=60,
)
def dma(df, n1=10, n2=50, m=10):
    """
    通达信公式：
        DIF:MA(CLOSE,N1)-MA(CLOSE,N2);
        DIFMA:MA(DIF,M);
    """
    close = df['close']
    
    ma1 = close.rolling(n1).mean()
    ma2 = close.rolling(n2).mean()
    
    dif = ma1 - ma2
    difma = dif.rolling(m).mean()
    
    return pd.DataFrame({
        'DMA_DIF': dif,
        'DMA_DIFMA': difma
    }, index=df.index)


# ==================== EXPMA ====================
@tdx_factor(
    name="EXPMA",
    category="trend",
    description="指数平均线（EXPMA/EMA）",
    params={"n1": 12, "n2": 50},
    tags=["趋势", "均线"],
    required_columns=['close'],
    min_length=60,
)
def expma(df, n1=12, n2=50):
    """
    通达信公式：
        EXPMA1:EMA(CLOSE,N1);
        EXPMA2:EMA(CLOSE,N2);
    """
    close = df['close']
    
    expma1 = close.ewm(span=n1, adjust=False).mean()
    expma2 = close.ewm(span=n2, adjust=False).mean()
    
    return pd.DataFrame({
        'EXPMA1': expma1,
        'EXPMA2': expma2
    }, index=df.index)


# ==================== SAR ====================
@tdx_factor(
    name="SAR",
    category="trend",
    description="抛物线转向指标（SAR）",
    params={"n": 4, "step": 0.02, "max_step": 0.2},
    tags=["趋势", "止损"],
    required_columns=['high', 'low'],
    min_length=30,
)
def sar(df, n=4, step=0.02, max_step=0.2):
    """
    SAR（简化版）
    通达信标准SAR需要迭代计算
    """
    high = df['high'].values
    low = df['low'].values
    length = len(df)
    
    sar_values = np.zeros(length)
    sar_values[:] = np.nan
    
    if length < n:
        return pd.Series(sar_values, index=df.index, name='SAR')
    
    # 初始化
    is_long = True
    af = step
    ep = high[0]
    sar_val = low[0]
    
    for i in range(1, length):
        sar_val = sar_val + af * (ep - sar_val)
        
        if is_long:
            if low[i] < sar_val:
                is_long = False
                sar_val = ep
                ep = low[i]
                af = step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            if high[i] > sar_val:
                is_long = True
                sar_val = ep
                ep = high[i]
                af = step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
        
        sar_values[i] = sar_val
    
    return pd.Series(sar_values, index=df.index, name='SAR')