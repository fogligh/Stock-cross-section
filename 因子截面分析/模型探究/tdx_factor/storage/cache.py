# storage/cache.py

import pandas as pd
import numpy as np
import pickle
import os
import hashlib
from typing import Any, Optional, Dict
from datetime import datetime, timedelta


class FactorCache:
    """
    因子缓存
    
    支持：
    1. 内存缓存
    2. 磁盘缓存
    3. 自动过期
    """
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 max_memory_items: int = 100,
                 default_ttl: int = 3600):
        """
        参数:
            cache_dir: 磁盘缓存目录
            max_memory_items: 内存缓存最大条目数
            default_ttl: 默认过期时间（秒）
        """
        self.cache_dir = cache_dir
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        
        # 内存缓存
        self._memory_cache: Dict[str, Dict] = {}
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    # ==================== 1. 生成缓存键 ====================
    
    @staticmethod
    def _make_key(factor_name: str, 
                  stock: str, 
                  date: str, 
                  params: Optional[Dict] = None) -> str:
        """生成缓存键"""
        params_str = str(sorted(params.items())) if params else ""
        raw = f"{factor_name}_{stock}_{date}_{params_str}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    # ==================== 2. 内存缓存 ====================
    
    def get_memory(self, key: str) -> Optional[Any]:
        """从内存获取"""
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if datetime.now() < entry['expire_at']:
                entry['hits'] += 1
                return entry['value']
            else:
                del self._memory_cache[key]
        return None
    
    def set_memory(self, key: str, value: Any, ttl: Optional[int] = None):
        """写入内存"""
        # LRU淘汰
        if len(self._memory_cache) >= self.max_memory_items:
            # 删除最少使用的
            sorted_keys = sorted(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k]['hits']
            )
            for k in sorted_keys[:self.max_memory_items // 4]:
                del self._memory_cache[k]
        
        ttl = ttl or self.default_ttl
        self._memory_cache[key] = {
            'value': value,
            'expire_at': datetime.now() + timedelta(seconds=ttl),
            'hits': 0,
            'created_at': datetime.now()
        }
    
    # ==================== 3. 磁盘缓存 ====================
    
    def get_disk(self, key: str) -> Optional[Any]:
        """从磁盘获取"""
        file_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
            
            if datetime.now() < entry['expire_at']:
                return entry['value']
            else:
                os.remove(file_path)
                return None
        except Exception:
            return None
    
    def set_disk(self, key: str, value: Any, ttl: Optional[int] = None):
        """写入磁盘"""
        ttl = ttl or self.default_ttl * 24  # 磁盘缓存时间更长
        
        file_path = os.path.join(self.cache_dir, f"{key}.pkl")
        entry = {
            'value': value,
            'expire_at': datetime.now() + timedelta(seconds=ttl),
            'created_at': datetime.now()
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            print(f"磁盘缓存写入失败: {e}")
    
    # ==================== 4. 统一接口 ====================
    
    def get(self, 
            factor_name: str,
            stock: str,
            date: str,
            params: Optional[Dict] = None,
            use_disk: bool = True) -> Optional[Any]:
        """
        获取缓存
        """
        key = self._make_key(factor_name, stock, date, params)
        
        # 先查内存
        value = self.get_memory(key)
        if value is not None:
            return value
        
        # 再查磁盘
        if use_disk:
            value = self.get_disk(key)
            if value is not None:
                # 回写到内存
                self.set_memory(key, value)
                return value
        
        return None
    
    def set(self,
            factor_name: str,
            stock: str,
            date: str,
            value: Any,
            params: Optional[Dict] = None,
            ttl: Optional[int] = None,
            use_disk: bool = True):
        """
        写入缓存
        """
        key = self._make_key(factor_name, stock, date, params)
        
        # 写内存
        self.set_memory(key, value, ttl)
        
        # 写磁盘
        if use_disk:
            self.set_disk(key, value, ttl)
    
    # ==================== 5. 清理 ====================
    
    def clear_memory(self):
        """清空内存缓存"""
        self._memory_cache.clear()
    
    def clear_disk(self):
        """清空磁盘缓存"""
        for f in os.listdir(self.cache_dir):
            if f.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, f))
    
    def clear_all(self):
        """清空所有缓存"""
        self.clear_memory()
        self.clear_disk()
    
    def clean_expired(self):
        """清理过期缓存"""
        now = datetime.now()
        
        # 内存
        expired_keys = [
            k for k, v in self._memory_cache.items()
            if now >= v['expire_at']
        ]
        for k in expired_keys:
            del self._memory_cache[k]
        
        # 磁盘
        for f in os.listdir(self.cache_dir):
            if f.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, f)
                try:
                    with open(file_path, 'rb') as fp:
                        entry = pickle.load(fp)
                    if now >= entry['expire_at']:
                        os.remove(file_path)
                except Exception:
                    os.remove(file_path)
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        memory_size = sum(
            len(pickle.dumps(v['value'])) 
            for v in self._memory_cache.values()
        )
        
        disk_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        disk_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in disk_files
        )
        
        return {
            'memory_items': len(self._memory_cache),
            'memory_size_mb': memory_size / 1024 / 1024,
            'disk_items': len(disk_files),
            'disk_size_mb': disk_size / 1024 / 1024,
        }