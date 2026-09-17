# core/decorators.py

import functools
import time
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime

# ==================== 全局注册中心 ====================
FACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}


# ==================== 装饰器1：因子注册 ====================
def register_factor(name: str, 
                    category: str = "unknown",
                    description: str = "",
                    params: Optional[Dict] = None,
                    tags: Optional[List[str]] = None,
                    version: str = "1.0.0"):
    """
    因子注册装饰器
    
    用法：
        @register_factor(name="MACD", category="momentum", 
                        description="指数平滑异同移动平均",
                        params={"fast": 12, "slow": 26, "signal": 9})
        def macd_factor(df, fast=12, slow=26, signal=9):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # 注册元信息
        FACTOR_REGISTRY[name] = {
            'name': name,
            'category': category,
            'description': description,
            'params': params or {},
            'tags': tags or [],
            'version': version,
            'func': func,
            'registered_at': datetime.now().isoformat(),
        }
        
        wrapper._factor_name = name
        wrapper._factor_meta = FACTOR_REGISTRY[name]
        
        return wrapper
    return decorator


# ==================== 装饰器2：参数校验 ====================
def validate_params(**validators):
    """
    参数校验装饰器
    
    用法：
        @validate_params(period=lambda x: isinstance(x, int) and x > 0)
        def my_factor(df, period=20):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 合并参数
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"[{func.__name__}] 参数 {param_name}={value} 校验失败"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==================== 装饰器3：数据校验 ====================
def validate_data(required_columns: List[str], 
                  min_length: int = 30,
                  allow_nan: bool = False):
    """
    数据校验装饰器
    
    用法：
        @validate_data(required_columns=['close', 'volume'], min_length=60)
        def my_factor(df):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(df, *args, **kwargs):
            if df is None or len(df) < min_length:
                raise ValueError(
                    f"[{func.__name__}] 数据长度不足: {len(df) if df is not None else 0} < {min_length}"
                )
            
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(f"[{func.__name__}] 缺少列: {missing}")
            
            if not allow_nan and df[required_columns].isna().any().any():
                raise ValueError(f"[{func.__name__}] 数据包含NaN")
            
            return func(df, *args, **kwargs)
        return wrapper
    return decorator


# ==================== 装饰器4：缓存 ====================
def cache_result(cache_key_func: Optional[Callable] = None):
    """
    结果缓存装饰器
    
    用法：
        @cache_result(cache_key_func=lambda df, period: f"ma_{period}_{df.index[-1]}")
        def my_factor(df, period=20):
            ...
    """
    _cache = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(df, *args, **kwargs):
            if cache_key_func:
                key = cache_key_func(df, *args, **kwargs)
            else:
                key = f"{func.__name__}_{hash(str(df.index[-1]))}_{args}_{kwargs}"
            
            if key not in _cache:
                _cache[key] = func(df, *args, **kwargs)
            return _cache[key]
        
        wrapper._cache = _cache
        return wrapper
    return decorator


# ==================== 装饰器5：性能计时 ====================
def timeit(func: Callable) -> Callable:
    """性能计时装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] 耗时: {elapsed:.4f}秒")
        return result
    return wrapper


# ==================== 装饰器6：异常捕获 ====================
def safe_calc(default_value=np.nan):
    """
    异常捕获装饰器
    
    用法：
        @safe_calc(default_value=0)
        def my_factor(df):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"[{func.__name__}] 计算失败: {e}")
                # 返回默认值
                if len(args) > 0 and isinstance(args[0], pd.DataFrame):
                    return pd.Series(default_value, index=args[0].index)
                return default_value
        return wrapper
    return decorator


# ==================== 装饰器7：组合装饰器（推荐） ====================
def tdx_factor(name: str,
               category: str = "unknown",
               description: str = "",
               params: Optional[Dict] = None,
               tags: Optional[List[str]] = None,
               required_columns: Optional[List[str]] = None,
               min_length: int = 30,
               param_validators: Optional[Dict] = None,
               use_cache: bool = True,
               verbose: bool = True):
    """
    通达信因子一站式装饰器（组合装饰器）
    
    整合了：注册、参数校验、数据校验、缓存、异常捕获、计时
    
    用法：
        @tdx_factor(
            name="MACD",
            category="momentum",
            description="指数平滑异同移动平均",
            params={"fast": 12, "slow": 26, "signal": 9},
            required_columns=['close'],
            min_length=60,
            param_validators={
                'fast': lambda x: isinstance(x, int) and x > 0,
                'slow': lambda x: isinstance(x, int) and x > 0,
            }
        )
        def macd(df, fast=12, slow=26, signal=9):
            ...
            return pd.DataFrame({'DIF': dif, 'DEA': dea, 'MACD': macd})
    """
    def decorator(func: Callable) -> Callable:
        # 应用各个装饰器
        wrapped = func
        
        # 1. 数据校验
        if required_columns:
            wrapped = validate_data(
                required_columns=required_columns,
                min_length=min_length
            )(wrapped)
        
        # 2. 参数校验
        if param_validators:
            wrapped = validate_params(**param_validators)(wrapped)
        
        # 3. 缓存
        if use_cache:
            wrapped = cache_result()(wrapped)
        
        # 4. 异常捕获
        wrapped = safe_calc()(wrapped)
        
        # 5. 注册
        wrapped = register_factor(
            name=name,
            category=category,
            description=description,
            params=params,
            tags=tags
        )(wrapped)
        
        return wrapped
    return decorator