# __init__.py (项目根目录)

from .core.registry import FactorRegistry
from .core.engine import FactorEngine
from .core.validator import FactorValidator

# 导入所有因子，触发注册
from .factors import trend, momentum, volume, volatility

__version__ = "1.0.0"
__all__ = [
    'FactorRegistry',
    'FactorEngine',
    'FactorValidator',
]