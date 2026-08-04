import pandas as pd
import numpy as np
import os
from datetime import datetime

def extract_sample_stocks():
    """
    从原始股票数据中随机抽取100只股票，筛选指定日期范围的数据
    """
    # 1. 设置文件路径
    input_file = r'D:\a111111\回测模拟\data\stk_daily.csv'
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extracted_data')
    output_file = os.path.join(output_dir, 'stk_100.csv')
    
    # 2. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 3. 读取原始数据
    print(f"读取文件: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"原始数据形状: {df.shape}")
        print(f"原始数据列: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件出错: {e}")
        return
    
    # 4. 转换日期格式（月/日/年 格式）
    print("\n转换日期格式...")
    print(f"trade_date 示例值: {df['trade_date'].head(3).tolist()}")
    
    try:
        # 尝试月/日/年格式
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%m/%d/%Y')
        print("使用格式: 月/日/年 (%m/%d/%Y)")
    except:
        try:
            # 尝试日/月/年格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%d/%m/%Y')
            print("使用格式: 日/月/年 (%d/%m/%Y)")
        except:
            try:
                # 自动推断
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                print("使用格式: 自动推断")
            except Exception as e:
                print(f"日期格式转换失败: {e}")
                return
    
    print(f"日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
    
    # 5. 筛选日期范围
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 1, 12)
    
    print(f"\n筛选日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    
    df_date_filtered = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
    print(f"日期筛选后数据形状: {df_date_filtered.shape}")
    
    if df_date_filtered.empty:
        print("警告: 在指定日期范围内没有数据")
        return
    
    # 6. 获取所有股票代码
    if 'ts_code' in df_date_filtered.columns:
        all_stocks = df_date_filtered['ts_code'].unique()
        print(f"\n日期范围内共有 {len(all_stocks)} 只股票")
        
        if len(all_stocks) <= 100:
            selected_stocks = all_stocks
            print(f"股票总数少于100只，全部选取: {len(selected_stocks)} 只")
        else:
            np.random.seed(42)
            selected_stocks = np.random.choice(all_stocks, size=100, replace=False)
            print(f"随机抽取 100 只股票")
            print(f"示例股票代码: {selected_stocks[:5].tolist()}")
        
        # 7. 筛选选定股票的数据
        df_selected = df_date_filtered[df_date_filtered['ts_code'].isin(selected_stocks)]
        
        # 8. 按股票代码和日期排序
        df_selected = df_selected.sort_values(['ts_code', 'trade_date'])
        
        # 9. 重置索引
        df_selected = df_selected.reset_index(drop=True)
        
        print(f"\n最终数据形状: {df_selected.shape}")
        print(f"实际股票数量: {df_selected['ts_code'].nunique()}")
        
        # 10. 显示统计信息
        print(f"\n数据统计:")
        print(f"  日期范围: {df_selected['trade_date'].min()} 到 {df_selected['trade_date'].max()}")
        print(f"  股票代码示例: {df_selected['ts_code'].head(5).tolist()}")
        print(f"  每只股票平均记录数: {len(df_selected) / df_selected['ts_code'].nunique():.1f}")
        
        stock_counts = df_selected.groupby('ts_code').size()
        print(f"  每只股票最少记录数: {stock_counts.min()}")
        print(f"  每只股票最多记录数: {stock_counts.max()}")
        
        # 11. 保存到CSV文件（保存时保留日期列为字符串格式）
        print(f"\n保存文件到: {output_file}")
        # 将日期转换为字符串再保存，避免格式问题
        df_selected['trade_date'] = df_selected['trade_date'].dt.strftime('%Y-%m-%d')
        df_selected.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 数据保存成功！")
        
        # 12. 显示前几行预览
        print("\n数据预览（前5行）:")
        print(df_selected.head(5))
        
        # 13. 显示日期分布
        print(f"\n日期分布（按年-月统计）:")
        # 重新转换为日期类型用于分组
        df_selected['trade_date_temp'] = pd.to_datetime(df_selected['trade_date'])
        date_stats = df_selected.groupby(df_selected['trade_date_temp'].dt.to_period('M')).size()
        print(date_stats.head(10))
        # 删除临时列
        df_selected = df_selected.drop('trade_date_temp', axis=1)
        
        return df_selected
    
    else:
        print("错误: 没有 ts_code 列")
        return


def validate_extracted_data(file_path):
    """验证提取的数据（修复日期类型问题）"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    
    print("\n" + "="*60)
    print("数据验证报告")
    print("="*60)
    
    # 1. 基本统计
    print(f"总记录数: {len(df):,}")
    print(f"股票数量: {df['ts_code'].nunique()}")
    print(f"日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
    
    # 2. 列检查
    print(f"\n列名: {df.columns.tolist()}")
    
    # 3. 缺失值检查
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n缺失值统计:")
        print(missing[missing > 0])
    
    # 4. 每只股票的记录数
    print(f"\n每只股票记录数统计:")
    counts = df.groupby('ts_code').size()
    print(f"  最小: {counts.min()}")
    print(f"  最大: {counts.max()}")
    print(f"  平均: {counts.mean():.1f}")
    
    # 5. 检查日期连续性（修复：先将日期转为datetime）
    print(f"\n日期连续性检查（第一只股票）:")
    first_stock = df['ts_code'].iloc[0]
    stock_data = df[df['ts_code'] == first_stock].sort_values('trade_date')
    
    # 将日期转换为datetime类型
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
    
    print(f"  股票: {first_stock}")
    print(f"  记录数: {len(stock_data)}")
    if len(stock_data) > 1:
        date_diff = stock_data['trade_date'].diff().dropna()
        # 计算日期间隔（天数）
        date_diff_days = date_diff.dt.days
        print(f"  日期间隔平均: {date_diff_days.mean():.1f} 天")
        print(f"  日期间隔最大: {date_diff_days.max()} 天")
        print(f"  日期间隔最小: {date_diff_days.min()} 天")
        
        # 检查是否有缺失的交易日
        expected_dates = pd.date_range(start=stock_data['trade_date'].min(), 
                                       end=stock_data['trade_date'].max(), 
                                       freq='D')
        missing_dates = set(expected_dates) - set(stock_data['trade_date'])
        if len(missing_dates) > 0:
            # 只统计交易日（周一到周五）
            weekdays = [d for d in expected_dates if d.weekday() < 5]
            missing_weekdays = set(weekdays) - set(stock_data['trade_date'])
            print(f"  缺失交易日数: {len(missing_weekdays)}")
    
    return df


if __name__ == "__main__":
    # 运行主函数
    result = extract_sample_stocks()
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    
    # 显示文件保存位置
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extracted_data')
    output_file = os.path.join(output_dir, 'stk_100.csv')
    if os.path.exists(output_file):
        print(f"\n📁 输出文件: {output_file}")
        file_size = os.path.getsize(output_file) / 1024
        if file_size > 1024:
            print(f"📊 文件大小: {file_size / 1024:.2f} MB")
        else:
            print(f"📊 文件大小: {file_size:.2f} KB")
        
        # 验证数据（重新读取验证）
        print("\n验证数据...")
        validate_extracted_data(output_file)