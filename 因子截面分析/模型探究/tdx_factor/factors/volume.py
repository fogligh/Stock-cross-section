# factors/volume.py

import numpy as np
import pandas as pd
from core.decorators import tdx_factor


# ==================== OBV ====================
@tdx_factor(
    name="OBV",
    category="volume",
    description="能量潮（On Balance Volume）",
    params={},
    tags=["量价", "经典"],
    required_columns=['close', 'volume'],
    min_length=30,
)
def obv(df):
    """
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
    
    return pd.DataFrame({'OBV': obv_val}, index=df.index)


# ==================== VOL-MA ====================
@tdx_factor(
    name="VOL_MA",
    category="volume",
    description="成交量均线",
    params={"n1": 5, "n2": 10, "n3": 20},
    tags=["量价", "均线"],
    required_columns=['volume'],
    min_length=30,
)
def vol_ma(df, n1=5, n2=10, n3=20):
    """
    通达信公式：
        MAVOL1:MA(VOLUME,N1);
        MAVOL2:MA(VOLUME,N2);
        MAVOL3:MA(VOLUME,N3);
    """
    volume = df['volume']
    
    return pd.DataFrame({
        'VOL_MA5': volume.rolling(n1).mean(),
        'VOL_MA10': volume.rolling(n2).mean(),
        'VOL_MA20': volume.rolling(n3).mean(),
        'VOL_RATIO': volume / volume.rolling(n3).mean(),
    }, index=df.index)


# ==================== VR ====================
@tdx_factor(
    name="VR",
    category="volume",
    description="成交量变异率（Volume Ratio）",
    params={"n": 26},
    tags=["量价", "超买超卖"],
    required_columns=['close', 'volume'],
    min_length=60,
)
def vr(df, n=26):
    """
    通达信公式：
        VR:SUM(IF(CLOSE>REF(CLOSE,1),VOL,0),N)/
          SUM(IF(CLOSE<REF(CLOSE,1),VOL,0),N)*100;
    """
    close = df['close']
    volume = df['volume']
    
    prev_close = close.shift(1)
    
    up_vol = np.where(close > prev_close, volume, 0)
    down_vol = np.where(close < prev_close, volume, 0)
    
    up_vol_series = pd.Series(up_vol, index=df.index)
    down_vol_series = pd.Series(down_vol, index=df.index)
    
    vr_val = up_vol_series.rolling(n).sum() / down_vol_series.rolling(n).sum().replace(0, np.nan) * 100
    
    return pd.DataFrame({'VR': vr_val}, index=df.index)


# ==================== MFI ====================
@tdx_factor(
    name="MFI",
    category="volume",
    description="资金流量指标（Money Flow Index）",
    params={"n": 14},
    tags=["量价", "超买超卖"],
    required_columns=['close', 'high', 'low', 'volume'],
    min_length=30,
)
def mfi(df, n=14):
    """
    MFI = 100 - 100/(1+PMF/NMF)
    PMF: 正资金流
    NMF: 负资金流
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    tp = (high + low + close) / 3
    mf = tp * volume
    
    prev_tp = tp.shift(1)
    
    pos_mf = pd.Series(np.where(tp > prev_tp, mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(tp < prev_tp, mf, 0), index=df.index)
    
    pmf = pos_mf.rolling(n).sum()
    nmf = neg_mf.rolling(n).sum()
    
    mfi_val = 100 - 100 / (1 + pmf / nmf.replace(0, np.nan))
    
    return pd.DataFrame({'MFI': mfi_val}, index=df.index)


# ==================== 量价配合 ====================
@tdx_factor(
    name="VP_CORR",
    category="volume",
    description="量价相关性",
    params={"n": 20},
    tags=["量价"],
    required_columns=['close', 'volume'],
    min_length=30,
)
def vp_corr(df, n=20):
    """
    量价滚动相关系数
    """
    close = df['close']
    volume = df['volume']
    
    ret = close.pct_change()
    
    corr = ret.rolling(n).corr(volume)
    
    return pd.DataFrame({'VP_CORR': corr}, index=df.index)


# ==================== 主力资金 ====================
@tdx_factor(
    name="MAIN_FORCE",
    category="volume",
    description="主力资金流入（简化版）",
    params={"n": 20},
    tags=["量价", "资金"],
    required_columns=['close', 'high', 'low', 'volume'],
    min_length=30,
)
def main_force(df, n=20):
    """
    简化版主力资金
    上涨日的成交量 - 下跌日的成交量
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # 价格位置
    price_position = (close - low) / (high - low).replace(0, np.nan)
    
    # 资金流
    money_flow = price_position * volume
    
    # 净流入
    main_flow = money_flow - (1 - price_position) * volume
    main_flow_ma = main_flow.rolling(n).mean()
    
    return pd.DataFrame({
        'MAIN_FLOW': main_flow,
        'MAIN_FLOW_MA': main_flow_ma
    }, index=df.index)