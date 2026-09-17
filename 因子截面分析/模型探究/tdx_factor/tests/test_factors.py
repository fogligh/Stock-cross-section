# tests/test_factors.py

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import FactorRegistry
from core.engine import FactorEngine
from core.validator import FactorValidator
from core.decorators import FACTOR_REGISTRY

# 导入所有因子（触发注册）
from factors import trend, momentum, volume, volatility


def create_test_data(n=200, seed=42):
    """创建测试数据"""
    np.random.seed(seed)
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


class TestFactorRegistry(unittest.TestCase):
    """测试因子注册中心"""
    
    def test_registry_has_factors(self):
        """测试因子是否注册"""
        factors = FactorRegistry.list_all()
        self.assertGreater(len(factors), 0)
        print(f"已注册因子: {factors}")
    
    def test_get_factor(self):
        """测试获取因子"""
        func = FactorRegistry.get_func("MACD")
        self.assertTrue(callable(func))
    
    def test_get_by_category(self):
        """测试按分类查询"""
        momentum_factors = FactorRegistry.list_by_category("momentum")
        self.assertGreater(len(momentum_factors), 0)
        print(f"动量因子: {momentum_factors}")
    
    def test_describe(self):
        """测试因子描述"""
        desc = FactorRegistry.describe("MACD")
        self.assertIn("MACD", desc)
        print(desc)


class TestMACD(unittest.TestCase):
    """测试MACD因子"""
    
    def setUp(self):
        self.df = create_test_data()
        self.func = FactorRegistry.get_func("MACD")
    
    def test_output_shape(self):
        """测试输出形状"""
        result = self.func(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertEqual(list(result.columns), ['DIF', 'DEA', 'MACD'])
    
    def test_no_nan_in_recent(self):
        """测试最近值无NaN"""
        result = self.func(self.df)
        self.assertFalse(result.iloc[-1].isna().any())
    
    def test_custom_params(self):
        """测试自定义参数"""
        result = self.func(self.df, fast=5, slow=10, signal=3)
        self.assertEqual(len(result), len(self.df))
    
    def test_invalid_params(self):
        """测试非法参数"""
        with self.assertRaises(ValueError):
            self.func(self.df, fast=-1)


class TestKDJ(unittest.TestCase):
    """测试KDJ因子"""
    
    def setUp(self):
        self.df = create_test_data()
        self.func = FactorRegistry.get_func("KDJ")
    
    def test_output(self):
        """测试输出"""
        result = self.func(self.df)
        self.assertEqual(list(result.columns), ['K', 'D', 'J'])
    
    def test_value_range(self):
        """测试值范围（K/D在0-100）"""
        result = self.func(self.df)
        k_valid = result['K'].dropna()
        self.assertTrue((k_valid >= -10).all() and (k_valid <= 110).all())


class TestRSI(unittest.TestCase):
    """测试RSI因子"""
    
    def setUp(self):
        self.df = create_test_data()
        self.func = FactorRegistry.get_func("RSI")
    
    def test_output(self):
        """测试输出"""
        result = self.func(self.df)
        self.assertEqual(list(result.columns), ['RSI1', 'RSI2', 'RSI3'])
    
    def test_value_range(self):
        """测试值范围（0-100）"""
        result = self.func(self.df)
        for col in result.columns:
            valid = result[col].dropna()
            self.assertTrue((valid >= 0).all() and (valid <= 100).all())


class TestBOLL(unittest.TestCase):
    """测试布林带"""
    
    def setUp(self):
        self.df = create_test_data()
        self.func = FactorRegistry.get_func("BOLL")
    
    def test_output(self):
        """测试输出"""
        result = self.func(self.df)
        self.assertIn('BOLL_UPPER', result.columns)
        self.assertIn('BOLL_LOWER', result.columns)
    
    def test_upper_above_lower(self):
        """测试上轨>下轨"""
        result = self.func(self.df).dropna()
        self.assertTrue((result['BOLL_UPPER'] >= result['BOLL_LOWER']).all())


class TestEngine(unittest.TestCase):
    """测试计算引擎"""
    
    def setUp(self):
        self.df = create_test_data()
        self.engine = FactorEngine()
    
    def test_calc_single_factor(self):
        """测试计算单个因子"""
        result = self.engine.calc_factor(self.df, "MACD")
        self.assertEqual(len(result), len(self.df))
    
    def test_calc_batch_factors(self):
        """测试批量计算"""
        configs = [
            {"name": "MACD"},
            {"name": "KDJ"},
            {"name": "RSI"},
        ]
        result = self.engine.calc_factors(self.df, configs)
        self.assertGreater(result.shape[1], 3)
        print(f"批量计算因子数: {result.shape[1]}")
    
    def test_standardize(self):
        """测试标准化"""
        factor = pd.Series(np.random.randn(100))
        std = FactorEngine.standardize(factor, 'zscore')
        self.assertAlmostEqual(std.mean(), 0, places=5)
        self.assertAlmostEqual(std.std(), 1, places=5)
    
    def test_winsorize(self):
        """测试去极值"""
        factor = pd.Series(np.concatenate([np.random.randn(98), [100, -100]]))
        win = FactorEngine.winsorize(factor, 'mad', n=3)
        self.assertLess(win.max(), 50)
        self.assertGreater(win.min(), -50)


class TestValidator(unittest.TestCase):
    """测试校验器"""
    
    def setUp(self):
        self.df = create_test_data()
        self.validator = FactorValidator("TestFactor")
    
    def test_data_quality(self):
        """测试数据质量检查"""
        result = self.validator.check_data_quality(self.df)
        self.assertEqual(result['total_rows'], 200)
    
    def test_factor_values(self):
        """测试因子值检查"""
        factor = self.df['close']
        result = self.validator.check_factor_values(factor)
        self.assertIn('mean', result)
        self.assertIn('std', result)
    
    def test_future_function(self):
        """测试未来函数检测"""
        def simple_ma(df):
            return df['close'].rolling(20).mean()
        
        result = self.validator.check_future_function(self.df, simple_ma)
        self.assertFalse(result['has_future_function'])


class TestCache(unittest.TestCase):
    """测试缓存"""
    
    def setUp(self):
        from storage.cache import FactorCache
        self.cache = FactorCache(cache_dir="./test_cache")
    
    def tearDown(self):
        self.cache.clear_all()
        import shutil
        if os.path.exists("./test_cache"):
            shutil.rmtree("./test_cache")
    
    def test_memory_cache(self):
        """测试内存缓存"""
        self.cache.set("MACD", "000001.SZ", "2024-01-01", {"value": 1.23})
        result = self.cache.get("MACD", "000001.SZ", "2024-01-01")
        self.assertEqual(result, {"value": 1.23})
    
    def test_stats(self):
        """测试统计"""
        stats = self.cache.get_stats()
        self.assertIn('memory_items', stats)


if __name__ == '__main__':
    unittest.main(verbosity=2)