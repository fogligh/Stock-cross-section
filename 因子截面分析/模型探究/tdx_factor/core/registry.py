# core/registry.py

from typing import Dict, List, Callable, Any
import pandas as pd
import numpy as np
from .decorators import FACTOR_REGISTRY


class FactorRegistry:
    """因子注册中心"""
    
    @staticmethod
    def list_all() -> List[str]:
        """列出所有已注册因子"""
        return list(FACTOR_REGISTRY.keys())
    
    @staticmethod
    def get(name: str) -> Dict[str, Any]:
        """获取因子元信息"""
        if name not in FACTOR_REGISTRY:
            raise KeyError(f"因子 {name} 未注册")
        return FACTOR_REGISTRY[name]
    
    @staticmethod
    def get_func(name: str) -> Callable:
        """获取因子函数"""
        return FactorRegistry.get(name)['func']
    
    @staticmethod
    def list_by_category(category: str) -> List[str]:
        """按分类列出因子"""
        return [
            name for name, meta in FACTOR_REGISTRY.items()
            if meta['category'] == category
        ]
    
    @staticmethod
    def list_by_tags(tags: List[str]) -> List[str]:
        """按标签列出因子"""
        return [
            name for name, meta in FACTOR_REGISTRY.items()
            if any(tag in meta['tags'] for tag in tags)
        ]
    
    @staticmethod
    def describe(name: str) -> str:
        """打印因子描述"""
        meta = FactorRegistry.get(name)
        lines = [
            f"因子名称: {meta['name']}",
            f"分类: {meta['category']}",
            f"描述: {meta['description']}",
            f"参数: {meta['params']}",
            f"标签: {meta['tags']}",
            f"版本: {meta['version']}",
            f"注册时间: {meta['registered_at']}",
        ]
        return "\n".join(lines)
    
    @staticmethod
    def batch_calc(df: pd.DataFrame, 
                   factor_names: List[str],
                   factor_params: Dict[str, Dict] = None) -> pd.DataFrame:
        """
        批量计算多个因子
        
        参数:
            df: 输入数据
            factor_names: 因子名称列表
            factor_params: 各因子的参数 {name: {param: value}}
        
        返回:
            DataFrame，包含所有因子值
        """
        results = {}
        
        for name in factor_names:
            try:
                func = FactorRegistry.get_func(name)
                params = (factor_params or {}).get(name, {})
                result = func(df, **params)
                
                # 统一处理返回值
                if isinstance(result, pd.Series):
                    results[name] = result
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        results[f"{name}_{col}"] = result[col]
                elif isinstance(result, np.ndarray):
                    results[name] = pd.Series(result, index=df.index)
                    
            except Exception as e:
                print(f"因子 {name} 计算失败: {e}")
                continue
        
        return pd.DataFrame(results, index=df.index)