# storage/database.py

import sqlite3
import pandas as pd
import numpy as np
import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FactorDatabase:
    """
    因子数据库
    
    功能：
    1. 因子值存储（单条/批量）
    2. 因子元信息管理
    3. 因子查询（按名称/日期/股票）
    4. 因子面板查询（多股票多日期）
    5. 数据导出
    6. 数据库维护
    """
    
    def __init__(self, db_path: str = "factor.db", echo: bool = False):
        """
        参数:
            db_path: 数据库文件路径（SQLite）
            echo: 是否打印SQL语句
        """
        self.db_path = db_path
        self.echo = echo
        
        # 创建目录
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # 连接数据库
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # 性能优化
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = -64000")  # 64MB
        self.conn.execute("PRAGMA temp_store = MEMORY")
        
        # 初始化表结构
        self._init_tables()
    
    # ==================== 1. 初始化表结构 ====================
    
    def _init_tables(self):
        """初始化所有表"""
        cursor = self.conn.cursor()
        
        # 表1：因子值表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                factor_value REAL,
                factor_value_str TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(trade_date, stock_code, factor_name)
            )
        """)
        
        # 索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_factor_date 
            ON factor_values(trade_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_factor_stock 
            ON factor_values(stock_code)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_factor_name 
            ON factor_values(factor_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_factor_date_name 
            ON factor_values(trade_date, factor_name)
        """)
        
        # 表2：因子元信息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_meta (
                factor_name TEXT PRIMARY KEY,
                category TEXT,
                description TEXT,
                params TEXT,
                tags TEXT,
                version TEXT,
                registered_at TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 表3：因子计算日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_name TEXT NOT NULL,
                stock_code TEXT,
                trade_date TEXT,
                status TEXT,
                message TEXT,
                elapsed_ms REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 表4：因子统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_name TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                stock_count INTEGER,
                mean REAL,
                std REAL,
                min REAL,
                max REAL,
                median REAL,
                ic REAL,
                ir REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(factor_name, trade_date)
            )
        """)
        
        self.conn.commit()
        
        if self.echo:
            print(f"✅ 数据库初始化完成: {self.db_path}")
    
    # ==================== 2. 因子值写入 ====================
    
    def save_factor(self, 
                    trade_date: str,
                    stock_code: str,
                    factor_name: str,
                    value: float) -> bool:
        """
        保存单个因子值
        
        参数:
            trade_date: 交易日期 'YYYY-MM-DD'
            stock_code: 股票代码 '000001.SZ'
            factor_name: 因子名称
            value: 因子值
        """
        try:
            cursor = self.conn.cursor()
            
            # 处理NaN和Inf
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                value = None
            
            cursor.execute("""
                INSERT OR REPLACE INTO factor_values 
                (trade_date, stock_code, factor_name, factor_value, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (trade_date, stock_code, factor_name, value))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ 保存失败 [{factor_name}/{stock_code}/{trade_date}]: {e}")
            return False
    
    def save_factors_batch(self, 
                           trade_date: str,
                           stock_code: str,
                           factors_dict: Dict[str, float]) -> int:
        """
        批量保存单只股票的所有因子
        
        参数:
            trade_date: 交易日期
            stock_code: 股票代码
            factors_dict: {factor_name: value}
        
        返回:
            成功保存的数量
        """
        if not factors_dict:
            return 0
        
        try:
            cursor = self.conn.cursor()
            
            data = []
            for name, value in factors_dict.items():
                # 处理NaN和Inf
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    value = None
                
                data.append((trade_date, stock_code, name, value))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO factor_values 
                (trade_date, stock_code, factor_name, factor_value, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, data)
            
            self.conn.commit()
            return len(data)
            
        except Exception as e:
            print(f"❌ 批量保存失败 [{stock_code}/{trade_date}]: {e}")
            return 0
    
    def save_panel(self,
                   factor_panel: pd.DataFrame,
                   factor_name: str) -> int:
        """
        保存因子面板数据（多股票多日期）
        
        参数:
            factor_panel: MultiIndex (date, stock) 或 DataFrame
            factor_name: 因子名称
        
        返回:
            保存数量
        """
        if factor_panel.empty:
            return 0
        
        try:
            cursor = self.conn.cursor()
            
            data = []
            
            # 处理MultiIndex
            if isinstance(factor_panel.index, pd.MultiIndex):
                for (date, stock), value in factor_panel.iloc[:, 0].items():
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                    
                    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                        value = None
                    
                    data.append((date_str, stock, factor_name, value))
            else:
                # 单列：index为日期，columns为股票
                for date, row in factor_panel.iterrows():
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                    
                    for stock, value in row.items():
                        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                            value = None
                        
                        data.append((date_str, stock, factor_name, value))
            
            # 批量写入（分批，避免SQL变量过多）
            batch_size = 5000
            total = 0
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                cursor.executemany("""
                    INSERT OR REPLACE INTO factor_values 
                    (trade_date, stock_code, factor_name, factor_value, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, batch)
                total += len(batch)
            
            self.conn.commit()
            return total
            
        except Exception as e:
            print(f"❌ 面板保存失败 [{factor_name}]: {e}")
            return 0
    
    def save_multi_factors(self,
                           trade_date: str,
                           all_factors: pd.DataFrame) -> int:
        """
        保存某日所有股票的所有因子
        
        参数:
            trade_date: 交易日期
            all_factors: DataFrame，index为股票代码，columns为因子名
        
        返回:
            保存数量
        """
        if all_factors.empty:
            return 0
        
        try:
            cursor = self.conn.cursor()
            
            data = []
            for stock, row in all_factors.iterrows():
                for factor_name, value in row.items():
                    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                        value = None
                    
                    data.append((trade_date, stock, factor_name, value))
            
            # 批量写入
            batch_size = 5000
            total = 0
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                cursor.executemany("""
                    INSERT OR REPLACE INTO factor_values 
                    (trade_date, stock_code, factor_name, factor_value, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, batch)
                total += len(batch)
            
            self.conn.commit()
            return total
            
        except Exception as e:
            print(f"❌ 多因子保存失败 [{trade_date}]: {e}")
            return 0
    
    # ==================== 3. 因子查询 ====================
    
    def query_factor(self,
                     factor_name: str,
                     stock_code: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        查询因子值
        
        参数:
            factor_name: 因子名称
            stock_code: 股票代码（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
        
        返回:
            DataFrame
        """
        sql = "SELECT * FROM factor_values WHERE factor_name = ?"
        params = [factor_name]
        
        if stock_code:
            sql += " AND stock_code = ?"
            params.append(stock_code)
        
        if start_date:
            sql += " AND trade_date >= ?"
            params.append(start_date)
        
        if end_date:
            sql += " AND trade_date <= ?"
            params.append(end_date)
        
        sql += " ORDER BY trade_date, stock_code"
        
        return pd.read_sql(sql, self.conn, params=params)
    
    def query_factor_panel(self,
                           factor_name: str,
                           stock_list: Optional[List[str]] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        查询因子面板（日期 × 股票）
        
        返回:
            DataFrame，index为日期，columns为股票代码
        """
        sql = "SELECT trade_date, stock_code, factor_value FROM factor_values WHERE factor_name = ?"
        params = [factor_name]
        
        if stock_list:
            placeholders = ','.join(['?'] * len(stock_list))
            sql += f" AND stock_code IN ({placeholders})"
            params.extend(stock_list)
        
        if start_date:
            sql += " AND trade_date >= ?"
            params.append(start_date)
        
        if end_date:
            sql += " AND trade_date <= ?"
            params.append(end_date)
        
        df = pd.read_sql(sql, self.conn, params=params)
        
        if df.empty:
            return pd.DataFrame()
        
        # 转为面板
        panel = df.pivot(index='trade_date', columns='stock_code', values='factor_value')
        panel.index = pd.to_datetime(panel.index)
        
        return panel
    
    def query_multi_factors(self,
                            factor_names: List[str],
                            trade_date: str,
                            stock_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        查询某日多个因子
        
        返回:
            DataFrame，index为股票代码，columns为因子名
        """
        if not factor_names:
            return pd.DataFrame()
        
        placeholders = ','.join(['?'] * len(factor_names))
        sql = f"""
            SELECT stock_code, factor_name, factor_value 
            FROM factor_values 
            WHERE factor_name IN ({placeholders}) AND trade_date = ?
        """
        params = factor_names + [trade_date]
        
        if stock_list:
            stock_placeholders = ','.join(['?'] * len(stock_list))
            sql += f" AND stock_code IN ({stock_placeholders})"
            params.extend(stock_list)
        
        df = pd.read_sql(sql, self.conn, params=params)
        
        if df.empty:
            return pd.DataFrame()
        
        # 转为宽表
        panel = df.pivot(index='stock_code', columns='factor_name', values='factor_value')
        
        return panel
    
    def query_latest(self, 
                     factor_name: str,
                     n_days: int = 1) -> pd.DataFrame:
        """
        查询最近N天的因子值
        """
        sql = """
            SELECT * FROM factor_values 
            WHERE factor_name = ? 
            AND trade_date >= (
                SELECT MAX(trade_date) FROM factor_values WHERE factor_name = ?
            )
            ORDER BY trade_date DESC, stock_code
        """
        
        df = pd.read_sql(sql, self.conn, params=[factor_name, factor_name])
        
        if df.empty:
            return df
        
        # 获取最近N个交易日
        dates = sorted(df['trade_date'].unique(), reverse=True)[:n_days]
        return df[df['trade_date'].isin(dates)]
    
    def query_date_range(self) -> Dict[str, str]:
        """查询数据库中已有的日期范围"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM factor_values")
        row = cursor.fetchone()
        
        return {
            'min_date': row[0],
            'max_date': row[1]
        }
    
    # ==================== 4. 因子元信息管理 ====================
    
    def save_meta(self, meta: Dict[str, Any]) -> bool:
        """
        保存因子元信息
        
        参数:
            meta: 来自 FactorRegistry 的元信息字典
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO factor_meta
                (factor_name, category, description, params, tags, version, registered_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                meta.get('name'),
                meta.get('category'),
                meta.get('description'),
                json.dumps(meta.get('params', {}), ensure_ascii=False),
                json.dumps(meta.get('tags', []), ensure_ascii=False),
                meta.get('version', '1.0.0'),
                meta.get('registered_at', datetime.now().isoformat()),
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ 元信息保存失败: {e}")
            return False
    
    def get_meta(self, factor_name: str) -> Optional[Dict]:
        """获取因子元信息"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM factor_meta WHERE factor_name = ?", (factor_name,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return {
            'factor_name': row['factor_name'],
            'category': row['category'],
            'description': row['description'],
            'params': json.loads(row['params']) if row['params'] else {},
            'tags': json.loads(row['tags']) if row['tags'] else [],
            'version': row['version'],
            'registered_at': row['registered_at'],
            'updated_at': row['updated_at'],
        }
    
    def list_all_factors(self) -> pd.DataFrame:
        """列出所有因子"""
        return pd.read_sql("SELECT * FROM factor_meta ORDER BY category, factor_name", self.conn)
    
    # ==================== 5. 因子统计 ====================
    
    def save_stats(self,
                   factor_name: str,
                   trade_date: str,
                   stats: Dict[str, float]) -> bool:
        """保存因子每日统计"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO factor_stats
                (factor_name, trade_date, stock_count, mean, std, min, max, median, ic, ir, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                factor_name,
                trade_date,
                stats.get('stock_count'),
                stats.get('mean'),
                stats.get('std'),
                stats.get('min'),
                stats.get('max'),
                stats.get('median'),
                stats.get('ic'),
                stats.get('ir'),
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ 统计保存失败: {e}")
            return False
    
    def query_stats(self,
                    factor_name: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """查询因子统计"""
        sql = "SELECT * FROM factor_stats WHERE factor_name = ?"
        params = [factor_name]
        
        if start_date:
            sql += " AND trade_date >= ?"
            params.append(start_date)
        
        if end_date:
            sql += " AND trade_date <= ?"
            params.append(end_date)
        
        sql += " ORDER BY trade_date"
        
        return pd.read_sql(sql, self.conn, params=params)
    
    # ==================== 6. 日志 ====================
    
    def log(self,
            factor_name: str,
            status: str,
            stock_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            message: Optional[str] = None,
            elapsed_ms: Optional[float] = None) -> bool:
        """记录计算日志"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO factor_log
                (factor_name, stock_code, trade_date, status, message, elapsed_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (factor_name, stock_code, trade_date, status, message, elapsed_ms))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ 日志保存失败: {e}")
            return False
    
    # ==================== 7. 导出 ====================
    
    def export_to_csv(self,
                      factor_name: str,
                      output_path: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> bool:
        """导出因子到CSV"""
        try:
            df = self.query_factor(factor_name, start_date=start_date, end_date=end_date)
            
            if df.empty:
                print(f"⚠️ 无数据可导出: {factor_name}")
                return False
            
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 已导出 {len(df)} 行到 {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False
    
    def export_to_excel(self,
                        factor_names: List[str],
                        output_path: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> bool:
        """导出多个因子到Excel（每个因子一个Sheet）"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for name in factor_names:
                    df = self.query_factor(name, start_date=start_date, end_date=end_date)
                    if not df.empty:
                        sheet_name = name[:31]  # Excel限制31字符
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"✅ 已导出 {len(factor_names)} 个因子到 {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False
    
    def export_panel_to_parquet(self,
                                 factor_name: str,
                                 output_path: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> bool:
        """导出面板到Parquet（高效压缩）"""
        try:
            panel = self.query_factor_panel(
                factor_name,
                start_date=start_date,
                end_date=end_date
            )
            
            if panel.empty:
                print(f"⚠️ 无数据可导出: {factor_name}")
                return False
            
            panel.to_parquet(output_path, compression='snappy')
            print(f"✅ 已导出面板 {panel.shape} 到 {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False
    
    # ==================== 8. 数据库维护 ====================
    
    def get_table_info(self) -> Dict[str, int]:
        """获取表信息"""
        cursor = self.conn.cursor()
        
        info = {}
        for table in ['factor_values', 'factor_meta', 'factor_log', 'factor_stats']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            info[table] = cursor.fetchone()[0]
        
        return info
    
    def get_db_size(self) -> float:
        """获取数据库大小（MB）"""
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path) / 1024 / 1024
        return 0
    
    def delete_factor(self, 
                      factor_name: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> int:
        """删除因子数据"""
        try:
            cursor = self.conn.cursor()
            
            sql = "DELETE FROM factor_values WHERE factor_name = ?"
            params = [factor_name]
            
            if start_date:
                sql += " AND trade_date >= ?"
                params.append(start_date)
            
            if end_date:
                sql += " AND trade_date <= ?"
                params.append(end_date)
            
            cursor.execute(sql, params)
            deleted = cursor.rowcount
            self.conn.commit()
            
            print(f"✅ 删除 {deleted} 行 [{factor_name}]")
            return deleted
            
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return 0
    
    def vacuum(self):
        """压缩数据库"""
        self.conn.execute("VACUUM")
        self.conn.commit()
        print("✅ 数据库压缩完成")
    
    def backup(self, backup_path: str):
        """备份数据库"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            print(f"✅ 数据库已备份到 {backup_path}")
            return True
        except Exception as e:
            print(f"❌ 备份失败: {e}")
            return False
    
    # ==================== 9. 上下文管理 ====================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
            self.conn = None


# ==================== 便捷函数 ====================

def create_database(db_path: str = "factor.db") -> FactorDatabase:
    """创建数据库实例"""
    return FactorDatabase(db_path)


def sync_registry_to_db(db: FactorDatabase):
    """
    将 FactorRegistry 中所有因子元信息同步到数据库
    """
    from core.decorators import FACTOR_REGISTRY
    
    count = 0
    for name, meta in FACTOR_REGISTRY.items():
        if db.save_meta(meta):
            count += 1
    
    print(f"✅ 已同步 {count} 个因子元信息到数据库")
    return count


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import numpy as np
    
    # 1. 创建数据库
    db = FactorDatabase("test_factor.db", echo=True)
    
    # 2. 保存单个因子
    db.save_factor(
        trade_date='2024-01-15',
        stock_code='000001.SZ',
        factor_name='MACD_DIF',
        value=0.1234
    )
    
    # 3. 批量保存
    factors = {
        'MACD_DIF': 0.1234,
        'MACD_DEA': 0.0987,
        'KDJ_K': 65.2,
        'KDJ_D': 58.3,
        'RSI1': 55.6,
    }
    db.save_factors_batch('2024-01-15', '000001.SZ', factors)
    
    # 4. 保存面板
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
    
    panel_data = pd.DataFrame(
        np.random.randn(10, 3),
        index=pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock']),
        columns=['value']
    )
    db.save_panel(panel_data, 'TEST_FACTOR')
    
    # 5. 查询
    df = db.query_factor('MACD_DIF', stock_code='000001.SZ')
    print(df)
    
    # 6. 查询面板
    panel = db.query_factor_panel('TEST_FACTOR')
    print(panel)
    
    # 7. 获取统计
    info = db.get_table_info()
    print(f"表信息: {info}")
    print(f"数据库大小: {db.get_db_size():.2f} MB")
    
    # 8. 关闭
    db.close()