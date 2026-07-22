"""
使用Dask进行完整的数据处理
从原始数据生成回测所需的所有数据文件
"""

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np
import random
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')
import time

# ==================== 获取脚本所在目录 ====================
def get_script_dir():
    """获取当前脚本所在的目录"""
    try:
        # 如果在Jupyter中运行
        from IPython import get_ipython
        if get_ipython() is not None:
            # 获取当前工作目录
            return Path.cwd()
    except:
        pass
    
    # 获取脚本文件路径
    try:
        script_path = Path(__file__).resolve()
        return script_path.parent
    except NameError:
        # 如果在交互式环境中，使用当前工作目录
        return Path.cwd()

# 获取脚本所在目录
SCRIPT_DIR = get_script_dir()
print(f"脚本所在目录: {SCRIPT_DIR}")

# ==================== 配置 ====================
class DataConfig:
    """数据配置"""
    # 路径配置 - 相对于脚本所在目录
    DATA_DIR = SCRIPT_DIR / ".." / "data"
    OUTPUT_DIR = SCRIPT_DIR / ".." / "final_data"
    
    # 基准股票
    BENCHMARK_STOCK = '000001.SZ'
    
    # 时间范围
    START_DATE = '2020-01-01'
    END_DATE = '2025-12-31'
    
    # 采样配置
    SAMPLE_SIZE = 200  # 随机抽取股票数量
    RANDOM_SEED = 42
    
    # Dask配置
    CHUNK_SIZE = '256MB'

# 创建配置实例
config = DataConfig()

# 标准化路径（转换为绝对路径）
config.DATA_DIR = config.DATA_DIR.resolve()
config.OUTPUT_DIR = config.OUTPUT_DIR.resolve()

print(f"数据目录: {config.DATA_DIR}")
print(f"输出目录: {config.OUTPUT_DIR}")
print(f"数据目录是否存在: {config.DATA_DIR.exists()}")

# 确保输出目录存在
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

# ==================== 1. 数据读取 ====================
print("\n" + "="*80)
print("步骤1: 读取原始数据")
print("="*80)

start_time = time.time()

# 检查数据文件是否存在
daily_file = config.DATA_DIR / "stk_daily.csv"
basic_file = config.DATA_DIR / "stk_basic.csv"

if not daily_file.exists():
    print(f"❌ 错误: 数据文件不存在 - {daily_file}")
    print(f"当前工作目录: {Path.cwd()}")
    print(f"脚本目录: {SCRIPT_DIR}")
    raise FileNotFoundError(f"找不到文件: {daily_file}")

if not basic_file.exists():
    print(f"❌ 错误: 数据文件不存在 - {basic_file}")
    raise FileNotFoundError(f"找不到文件: {basic_file}")

print(f"✅ 找到数据文件: {daily_file}")
print(f"✅ 找到数据文件: {basic_file}")

# 读取stk_daily.csv
print("\n读取 stk_daily.csv...")
daily_ddf = dd.read_csv(
    daily_file,
    dtype={'ts_code': 'object', 'trade_date': 'object'},
    blocksize=config.CHUNK_SIZE
)

# 转换日期
daily_ddf['trade_date'] = dd.to_datetime(daily_ddf['trade_date'], dayfirst=True)

# 读取stk_basic.csv
print("读取 stk_basic.csv...")
basic_df = pd.read_csv(basic_file)
basic_df['list_date'] = pd.to_datetime(basic_df['list_date'], dayfirst=True)

# 获取数据行数
try:
    total_rows = len(daily_ddf)
    print(f"原始数据行数: {total_rows:,}")
except:
    print("原始数据行数: 计算中...")
    
print(f"股票基础信息: {len(basic_df)} 只")

# ==================== 2. 数据过滤 ====================
print("\n" + "="*80)
print("步骤2: 过滤数据（日期范围 + 处理基准）")
print("="*80)

# 过滤日期范围
print(f"筛选日期范围: {config.START_DATE} 到 {config.END_DATE}")
date_filtered = daily_ddf[
    (daily_ddf['trade_date'] >= config.START_DATE) & 
    (daily_ddf['trade_date'] <= config.END_DATE)
]

# 获取所有股票列表
print("获取所有股票列表...")
all_stocks_list = date_filtered['ts_code'].unique().compute()
print(f"所有股票数量: {len(all_stocks_list)}")

# 检查基准股票是否存在
benchmark_exists = config.BENCHMARK_STOCK in all_stocks_list
if benchmark_exists:
    print(f"✅ 基准股票 {config.BENCHMARK_STOCK} 存在于数据中")
else:
    print(f"⚠️ 基准股票 {config.BENCHMARK_STOCK} 不存在于数据中")
    # 尝试查找相似的股票
    similar_stocks = [s for s in all_stocks_list if s.startswith('000001')]
    if similar_stocks:
        config.BENCHMARK_STOCK = similar_stocks[0]
        print(f"使用相似股票 {config.BENCHMARK_STOCK} 作为基准")
    else:
        # 使用第一个股票作为基准
        config.BENCHMARK_STOCK = all_stocks_list[0]
        print(f"使用第一个股票 {config.BENCHMARK_STOCK} 作为基准")

# 剔除基准股票用于随机抽取
other_filtered = date_filtered[date_filtered['ts_code'] != config.BENCHMARK_STOCK]

# 获取其他股票列表
other_stocks = other_filtered['ts_code'].unique().compute()
print(f"其他股票数量: {len(other_stocks)}")

# ==================== 3. 计算复权因子和价格 ====================
print("\n" + "="*80)
print("步骤3: 计算复权因子和后复权价格")
print("="*80)

def process_stock_data(df_chunk):
    """处理每只股票的数据"""
    results = []
    
    for ts_code, group in df_chunk.groupby('ts_code'):
        group = group.sort_values('trade_date')
        if len(group) < 2:
            continue
        
        # 计算每日收益率
        group['daily_ret'] = group['close'] / group['close'].shift(1) - 1
        
        # 计算累计复权因子
        cum_ret = (1 + group['daily_ret']).fillna(1).cumprod()
        group['accum_factor'] = cum_ret
        group.loc[group.index[0], 'accum_factor'] = 1.0
        
        # 计算TWAP
        group['twap'] = (group['open'] + group['high'] + group['low'] + group['close']) / 4
        
        # 后复权价格
        group['close_adj'] = group['close'] * group['accum_factor']
        group['twap_adj'] = group['twap'] * group['accum_factor']
        group['open_adj'] = group['open'] * group['accum_factor']
        group['high_adj'] = group['high'] * group['accum_factor']
        group['low_adj'] = group['low'] * group['accum_factor']
        
        results.append(group)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

print("计算复权因子...")

# 由于数据量可能较大，分批次处理
batch_size = 500  # 每批处理的股票数量
other_stocks_list = other_stocks.tolist()
all_processed = []

for i in range(0, len(other_stocks_list), batch_size):
    batch_stocks = other_stocks_list[i:i+batch_size]
    print(f"  处理第 {i//batch_size + 1}/{(len(other_stocks_list)-1)//batch_size + 1} 批...")
    
    # 筛选该批股票
    batch_ddf = other_filtered[other_filtered['ts_code'].isin(batch_stocks)]
    
    # 转换并处理
    try:
        batch_df = batch_ddf.compute()
        if not batch_df.empty:
            processed_batch = process_stock_data(batch_df)
            if not processed_batch.empty:
                all_processed.append(processed_batch)
    except Exception as e:
        print(f"  处理第 {i//batch_size + 1} 批时出错: {e}")
        continue

if all_processed:
    processed = pd.concat(all_processed, ignore_index=True)
else:
    processed = pd.DataFrame()

print(f"处理后数据行数: {len(processed):,}")

if processed.empty:
    print("❌ 错误: 处理后数据为空，请检查数据源")
    exit()

# ==================== 4. 计算技术面因子 ====================
print("\n" + "="*80)
print("步骤4: 计算技术面因子")
print("="*80)

def calculate_factors_for_stock(group):
    """计算单只股票的所有因子"""
    group = group.sort_values('trade_date').copy()
    
    if len(group) < 50:
        return group
    
    # ---- 收益率因子 ----
    group['ret_1d'] = group['close_adj'].pct_change()
    group['ret_5d'] = group['close_adj'].pct_change(5)
    group['ret_10d'] = group['close_adj'].pct_change(10)
    group['ret_20d'] = group['close_adj'].pct_change(20)
    group['ret_60d'] = group['close_adj'].pct_change(60)
    
    # ---- 均线因子 ----
    group['ma_5'] = group['close_adj'].rolling(5).mean()
    group['ma_10'] = group['close_adj'].rolling(10).mean()
    group['ma_20'] = group['close_adj'].rolling(20).mean()
    group['ma_60'] = group['close_adj'].rolling(60).mean()
    
    group['ma_5_ratio'] = group['close_adj'] / group['ma_5'] - 1
    group['ma_10_ratio'] = group['close_adj'] / group['ma_10'] - 1
    group['ma_20_ratio'] = group['close_adj'] / group['ma_20'] - 1
    group['ma_60_ratio'] = group['close_adj'] / group['ma_60'] - 1
    
    group['ma_5_10_cross'] = group['ma_5'] - group['ma_10']
    group['ma_10_20_cross'] = group['ma_10'] - group['ma_20']
    group['ma_20_60_cross'] = group['ma_20'] - group['ma_60']
    
    group['ma_5_slope'] = group['ma_5'] / group['ma_5'].shift(5) - 1
    group['ma_10_slope'] = group['ma_10'] / group['ma_10'].shift(10) - 1
    
    # ---- 成交量因子 ----
    group['volume_ma_5'] = group['vol'].rolling(5).mean()
    group['volume_ma_10'] = group['vol'].rolling(10).mean()
    group['volume_ma_20'] = group['vol'].rolling(20).mean()
    
    group['volume_ratio_5'] = group['vol'] / group['volume_ma_5']
    group['volume_ratio_10'] = group['vol'] / group['volume_ma_10']
    group['volume_ratio_20'] = group['vol'] / group['volume_ma_20']
    group['volume_ma_5_ratio'] = group['volume_ma_5'] / group['volume_ma_20']
    
    # ---- 价格形态因子 ----
    group['high_low_ratio'] = group['high_adj'] / group['low_adj']
    group['close_open_ratio'] = group['close_adj'] / group['open_adj']
    group['high_close_ratio'] = group['high_adj'] / group['close_adj'] - 1
    group['low_close_ratio'] = group['close_adj'] / group['low_adj'] - 1
    group['high_low_spread'] = (group['high_adj'] - group['low_adj']) / group['close_adj']
    
    # ---- 动量因子 ----
    group['momentum_10'] = group['close_adj'] / group['close_adj'].shift(10) - 1
    group['momentum_20'] = group['close_adj'] / group['close_adj'].shift(20) - 1
    group['momentum_30'] = group['close_adj'] / group['close_adj'].shift(30) - 1
    
    # ---- 波动率因子 ----
    group['volatility_5'] = group['ret_1d'].rolling(5).std()
    group['volatility_10'] = group['ret_1d'].rolling(10).std()
    group['volatility_20'] = group['ret_1d'].rolling(20).std()
    
    group['sharpe_10'] = group['ret_1d'].rolling(10).mean() / (group['ret_1d'].rolling(10).std() + 1e-8)
    group['sharpe_20'] = group['ret_1d'].rolling(20).mean() / (group['ret_1d'].rolling(20).std() + 1e-8)
    
    # ---- 回撤因子 ----
    group['cummax'] = group['close_adj'].expanding().max()
    group['drawdown'] = group['close_adj'] / group['cummax'] - 1
    group['max_drawdown_20'] = group['drawdown'].rolling(20).min()
    group['max_drawdown_60'] = group['drawdown'].rolling(60).min()
    
    # ---- 振幅因子 ----
    group['amplitude'] = (group['high_adj'] - group['low_adj']) / group['close_adj']
    group['amplitude_ma_5'] = group['amplitude'].rolling(5).mean()
    group['amplitude_ma_20'] = group['amplitude'].rolling(20).mean()
    
    # ---- RSI ----
    delta = group['close_adj'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    group['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ---- 布林带 ----
    bb_middle = group['close_adj'].rolling(20).mean()
    bb_std = group['close_adj'].rolling(20).std()
    group['bb_upper'] = bb_middle + 2 * bb_std
    group['bb_lower'] = bb_middle - 2 * bb_std
    group['bb_position'] = (group['close_adj'] - group['bb_lower']) / (group['bb_upper'] - group['bb_lower'] + 1e-8)
    group['bb_width'] = (group['bb_upper'] - group['bb_lower']) / (bb_middle + 1e-8)
    
    return group

print("计算技术面因子...")
factor_results = []
stock_groups = processed.groupby('ts_code')
total_stocks = len(stock_groups)

for idx, (ts_code, group) in enumerate(stock_groups, 1):
    if idx % 50 == 0:
        print(f"  处理第 {idx}/{total_stocks} 只股票: {ts_code}")
    
    try:
        factor_group = calculate_factors_for_stock(group)
        if not factor_group.empty:
            factor_results.append(factor_group)
    except Exception as e:
        print(f"  处理 {ts_code} 时出错: {e}")
        continue

full_data = pd.concat(factor_results, ignore_index=True)
print(f"因子计算完成，数据行数: {len(full_data):,}")

if full_data.empty:
    print("❌ 错误: 因子计算后数据为空")
    exit()

# ==================== 5. 数据清洗 ====================
print("\n" + "="*80)
print("步骤5: 数据清洗")
print("="*80)

# 定义滚动因子（需要前向填充）
rolling_factors = [
    'ma_5', 'ma_10', 'ma_20', 'ma_60',
    'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_60_ratio',
    'ma_5_10_cross', 'ma_10_20_cross', 'ma_20_60_cross',
    'ma_5_slope', 'ma_10_slope',
    'volume_ma_5', 'volume_ma_10', 'volume_ma_20',
    'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20', 'volume_ma_5_ratio',
    'volatility_5', 'volatility_10', 'volatility_20',
    'sharpe_10', 'sharpe_20',
    'amplitude_ma_5', 'amplitude_ma_20',
    'max_drawdown_20', 'max_drawdown_60',
    'bb_upper', 'bb_lower', 'bb_position', 'bb_width'
]

lag_factors = [
    'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
    'momentum_10', 'momentum_20', 'momentum_30'
]

print("开始清洗...")
cleaned_groups = []
stock_groups = full_data.groupby('ts_code')
total_stocks = len(stock_groups)

for idx, (ts_code, group) in enumerate(stock_groups, 1):
    if idx % 50 == 0:
        print(f"  清洗第 {idx}/{total_stocks} 只股票: {ts_code}")
    
    group = group.sort_values('trade_date').copy()
    
    # 识别因子列
    exclude_cols = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount',
                   'twap', 'accum_factor', 'close_adj', 'twap_adj', 'open_adj', 'high_adj', 'low_adj']
    factor_cols = [col for col in group.columns if col not in exclude_cols]
    
    # 填充缺失值
    for col in factor_cols:
        is_rolling = any(f in col for f in rolling_factors)
        is_lag = any(f in col for f in lag_factors)
        
        if is_rolling or is_lag:
            group[col] = group[col].ffill()
        else:
            group[col] = group[col].ffill().bfill()
    
    # 处理极端值（3倍标准差）
    for col in factor_cols:
        if group[col].dtype in ['float64', 'float32']:
            mean_val = group[col].mean()
            std_val = group[col].std()
            if std_val > 0 and not pd.isna(std_val):
                upper_bound = mean_val + 3 * std_val
                lower_bound = mean_val - 3 * std_val
                group[col] = group[col].clip(lower_bound, upper_bound)
    
    # 删除仍有缺失值的行
    group = group.dropna(subset=factor_cols, how='any')
    
    if not group.empty:
        cleaned_groups.append(group)

cleaned_data = pd.concat(cleaned_groups, ignore_index=True)
print(f"清洗后数据形状: {cleaned_data.shape}")
print(f"清洗后缺失值: {cleaned_data.isnull().sum().sum():,}")

if cleaned_data.empty:
    print("❌ 错误: 清洗后数据为空")
    exit()

# ==================== 6. 随机抽取200只股票 ====================
print("\n" + "="*80)
print("步骤6: 随机抽取200只股票")
print("="*80)

# 获取所有股票（排除基准）
available_stocks = cleaned_data[cleaned_data['ts_code'] != config.BENCHMARK_STOCK]['ts_code'].unique()
print(f"可用股票数量: {len(available_stocks)}")

# 随机抽取
sample_size = min(config.SAMPLE_SIZE, len(available_stocks))
selected_stocks = np.random.choice(available_stocks, size=sample_size, replace=False)
selected_stocks = sorted(selected_stocks)  # 排序后已经是list

print(f"抽取股票数量: {len(selected_stocks)}")
print(f"抽取的股票列表 (前20只):")
for i, stock in enumerate(selected_stocks[:20], 1):
    print(f"  {i:3d}. {stock}")

# ==================== 7. 提取基准股票并合并 ====================
print("\n" + "="*80)
print("步骤7: 提取基准股票并合并")
print("="*80)

# 获取基准股票数据
benchmark_data = cleaned_data[cleaned_data['ts_code'] == config.BENCHMARK_STOCK]

if benchmark_data.empty:
    print(f"⚠️ 基准股票 {config.BENCHMARK_STOCK} 不在清洗后的数据中")
    # 尝试从原始处理数据中获取
    benchmark_data = processed[processed['ts_code'] == config.BENCHMARK_STOCK]
    if benchmark_data.empty:
        print("❌ 错误: 无法获取基准股票数据")
        # 使用第一个可用股票作为基准
        # 从cleaned_data中获取第一个股票
        first_stock = cleaned_data['ts_code'].iloc[0]
        config.BENCHMARK_STOCK = first_stock
        benchmark_data = cleaned_data[cleaned_data['ts_code'] == first_stock]
        print(f"使用 {config.BENCHMARK_STOCK} 作为基准")
    else:
        print(f"从原始数据中获取基准股票 {config.BENCHMARK_STOCK}")

# 确保基准股票数据存在
if benchmark_data.empty:
    # 如果还是为空，从cleaned_data中获取第一个股票
    first_stock = cleaned_data['ts_code'].iloc[0]
    config.BENCHMARK_STOCK = first_stock
    benchmark_data = cleaned_data[cleaned_data['ts_code'] == first_stock]
    print(f"使用 {config.BENCHMARK_STOCK} 作为基准")

print(f"基准股票: {config.BENCHMARK_STOCK}, 数据量: {len(benchmark_data)}")

# 筛选随机股票数据
selected_data = cleaned_data[cleaned_data['ts_code'].isin(selected_stocks)]

# 合并基准和随机股票
# selected_stocks已经是list，直接使用
final_stocks_list = [config.BENCHMARK_STOCK] + selected_stocks
final_data = pd.concat([benchmark_data, selected_data], ignore_index=True)

print(f"最终数据包含 {len(final_stocks_list)} 只股票")
print(f"  - 基准股票: {config.BENCHMARK_STOCK}")
print(f"  - 随机股票: {len(selected_stocks)} 只")
print(f"最终数据形状: {final_data.shape}")

# ==================== 8. 合并股票基础信息 ====================
print("\n" + "="*80)
print("步骤8: 合并股票基础信息")
print("="*80)

# 合并基础信息
final_data = final_data.merge(
    basic_df[['ts_code', 'symbol', 'name', 'area', 'industry', 'market', 'list_date']],
    on='ts_code',
    how='left'
)

# 计算上市天数
final_data['days_since_listed'] = (final_data['trade_date'] - final_data['list_date']).dt.days

# 添加时间维度
final_data['year'] = final_data['trade_date'].dt.year
final_data['month'] = final_data['trade_date'].dt.month
final_data['quarter'] = final_data['trade_date'].dt.quarter

# ==================== 9. 最终数据清洗 ====================
print("\n" + "="*80)
print("步骤9: 最终数据清洗")
print("="*80)

# 处理缺失值
print(f"删除前数据形状: {final_data.shape}")
print(f"删除前缺失值: {final_data.isnull().sum().sum():,}")

# 删除任何包含缺失值的行
final_data = final_data.dropna()

print(f"删除后数据形状: {final_data.shape}")
print(f"删除后缺失值: {final_data.isnull().sum().sum():,}")

if final_data.empty:
    print("❌ 错误: 最终数据为空")
    exit()

# ==================== 10. 重新排列列顺序 ====================
print("\n" + "="*80)
print("步骤10: 重新排列列顺序")
print("="*80)

base_cols = ['trade_date', 'ts_code', 'symbol', 'name', 'area', 'industry', 'market', 
             'list_date', 'days_since_listed', 'year', 'month', 'quarter']

price_cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'twap', 
              'close_adj', 'twap_adj', 'open_adj', 'high_adj', 'low_adj', 'accum_factor']

# 获取所有因子列
factor_cols = [col for col in final_data.columns if col not in base_cols + price_cols]

# 重新排序
column_order = base_cols + price_cols + factor_cols
available_cols = [col for col in column_order if col in final_data.columns]
final_data = final_data[available_cols]

print(f"最终数据列数: {len(final_data.columns)}")
print(f"因子数量: {len(factor_cols)}")

# ==================== 11. 保存数据 ====================
print("\n" + "="*80)
print("步骤11: 保存最终数据")
print("="*80)

# 准备输出文件名
prefix = f"final_{len(selected_stocks)}stocks_with_benchmark_{config.START_DATE[:4]}_{config.END_DATE[:4]}"
output_dir = config.OUTPUT_DIR

# 保存CSV
csv_path = output_dir / f"{prefix}.csv"
print(f"保存CSV: {csv_path}")
final_data.to_csv(csv_path, index=False)
print(f"  ✅ CSV保存完成，大小: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")

# 保存Parquet
parquet_path = output_dir / f"{prefix}.parquet"
print(f"保存Parquet: {parquet_path}")
try:
    final_data.to_parquet(parquet_path, engine='fastparquet', index=False)
    print(f"  ✅ Parquet保存完成，大小: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")
except:
    try:
        final_data.to_parquet(parquet_path, engine='pyarrow', index=False, compression='snappy')
        print(f"  ✅ Parquet (pyarrow) 保存完成")
    except Exception as e:
        print(f"  ❌ Parquet保存失败: {e}")

# 保存股票列表
stock_list_path = output_dir / f"{prefix}_stock_list.txt"
with open(stock_list_path, 'w', encoding='utf-8') as f:
    f.write("最终数据股票列表\n")
    f.write("="*70 + "\n")
    f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据范围: {config.START_DATE} 到 {config.END_DATE}\n")
    f.write(f"总股票数量: {len(final_stocks_list)}\n")
    f.write(f"  - 基准股票: {config.BENCHMARK_STOCK}\n")
    f.write(f"  - 随机股票: {len(selected_stocks)} 只\n")
    f.write("="*70 + "\n\n")
    
    # 基准股票
    f.write(f"基准股票: {config.BENCHMARK_STOCK}\n")
    benchmark_info = final_data[final_data['ts_code'] == config.BENCHMARK_STOCK]
    if not benchmark_info.empty:
        info = benchmark_info.iloc[0]
        name = info.get('name', 'N/A')
        industry = info.get('industry', 'N/A')
        count = len(benchmark_info)
        f.write(f"  {config.BENCHMARK_STOCK}  {name:10s}  ({industry:10s})  数据量: {count:5d} 行 ★基准★\n")
    else:
        f.write(f"  ⚠️ 基准股票数据不存在\n")
    f.write("\n" + "-"*70 + "\n\n")
    
    # 随机股票
    f.write("随机股票:\n")
    valid_count = 0
    for i, stock in enumerate(selected_stocks, 1):
        stock_info = final_data[final_data['ts_code'] == stock]
        if not stock_info.empty:
            info = stock_info.iloc[0]
            name = info.get('name', 'N/A')
            industry = info.get('industry', 'N/A')
            count = len(stock_info)
            f.write(f"  {i:4d}. {stock}  {name:10s}  ({industry:10s})  数据量: {count:5d} 行\n")
            valid_count += 1
        else:
            f.write(f"  {i:4d}. {stock}  ⚠️ 数据不存在\n")
    
    f.write("\n" + "-"*70 + "\n")
    f.write(f"有效股票: {valid_count}/{len(selected_stocks)} 只\n")

print(f"  ✅ 股票列表: {stock_list_path}")

# 保存股票列表（仅包含有效股票）
valid_stocks = final_data['ts_code'].unique().tolist()
valid_stocks_path = output_dir / f"{prefix}_valid_stocks.txt"
with open(valid_stocks_path, 'w', encoding='utf-8') as f:
    f.write("最终数据有效股票列表\n")
    f.write("="*70 + "\n")
    f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"有效股票数量: {len(valid_stocks)}\n")
    f.write("="*70 + "\n\n")
    
    for i, stock in enumerate(valid_stocks, 1):
        stock_info = final_data[final_data['ts_code'] == stock]
        if not stock_info.empty:
            info = stock_info.iloc[0]
            name = info.get('name', 'N/A')
            industry = info.get('industry', 'N/A')
            count = len(stock_info)
            is_benchmark = "★基准★" if stock == config.BENCHMARK_STOCK else ""
            f.write(f"  {i:4d}. {stock}  {name:10s}  ({industry:10s})  数据量: {count:5d} 行 {is_benchmark}\n")

print(f"  ✅ 有效股票列表: {valid_stocks_path}")

# ==================== 12. 统计信息 ====================
print("\n" + "="*80)
print("数据统计摘要")
print("="*80)

# 获取实际有效的股票列表
actual_stocks = final_data['ts_code'].unique().tolist()
actual_benchmark = config.BENCHMARK_STOCK if config.BENCHMARK_STOCK in actual_stocks else None
actual_random = [s for s in actual_stocks if s != actual_benchmark]

print(f"\n最终数据统计:")
print(f"  - 数据行数: {final_data.shape[0]:,}")
print(f"  - 数据列数: {final_data.shape[1]}")
print(f"  - 实际股票数量: {len(actual_stocks)}")
print(f"    * 基准股票: {actual_benchmark} (存在: {actual_benchmark is not None})")
print(f"    * 随机股票: {len(actual_random)} 只")
print(f"  - 计划股票数量: {len(final_stocks_list)}")
print(f"  - 缺失股票: {len(final_stocks_list) - len(actual_stocks)} 只")
print(f"  - 日期范围: {final_data['trade_date'].min()} 到 {final_data['trade_date'].max()}")
print(f"  - 缺失值: {final_data.isnull().sum().sum():,}")

# 检查哪些股票缺失
missing_stocks = set(final_stocks_list) - set(actual_stocks)
if missing_stocks:
    print(f"\n⚠️ 缺失的股票 ({len(missing_stocks)}只):")
    for stock in list(missing_stocks)[:10]:
        print(f"    {stock}")
    if len(missing_stocks) > 10:
        print(f"    ... 还有 {len(missing_stocks) - 10} 只")

# 按年份统计
print(f"\n各年份数据量:")
for year in range(2020, 2026):
    year_data = final_data[final_data['year'] == year]
    print(f"  {year}年: {len(year_data):,} 行")

# 行业分布
print(f"\n行业分布:")
industry_counts = final_data.groupby('industry')['ts_code'].nunique().sort_values(ascending=False)
for industry, count in industry_counts.head(10).items():
    print(f"  {industry}: {count} 只")

# 因子列表
print(f"\n因子列表 (共{len(factor_cols)}个):")
for i, col in enumerate(factor_cols[:20], 1):
    print(f"  {i:3d}. {col}")
if len(factor_cols) > 20:
    print(f"  ... 还有 {len(factor_cols) - 20} 个因子")

# ==================== 完成 ====================
elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("处理完成！")
print("="*80)
print(f"\n总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
print(f"输出目录: {output_dir}")
print(f"\n输出文件:")
print(f"  ✅ {csv_path.name} ({csv_path.stat().st_size / 1024 / 1024:.2f} MB)")
print(f"  ✅ {parquet_path.name} ({parquet_path.stat().st_size / 1024 / 1024:.2f} MB)")
print(f"  ✅ {stock_list_path.name}")
print(f"  ✅ {valid_stocks_path.name}")
print("="*80)

# 显示输出文件的完整路径
print(f"\n完整路径:")
print(f"  {csv_path}")
print(f"  {parquet_path}")
print(f"  {stock_list_path}")
print(f"  {valid_stocks_path}")