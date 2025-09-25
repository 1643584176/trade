"""
时间处理工具类
提供统一的时间转换和处理方法，确保项目中时间处理的一致性
"""

import pandas as pd
from datetime import datetime, timedelta
import MetaTrader5 as mt5


class TimeUtils:
    """时间处理工具类"""
    
    @staticmethod
    def mt5_timestamp_to_datetime(timestamp):
        """
        将MT5时间戳转换为datetime对象（UTC时间）
        
        Args:
            timestamp (int): MT5时间戳
            
        Returns:
            datetime: UTC时间的datetime对象
        """
        return pd.to_datetime(timestamp, unit='s')
    
    @staticmethod
    def mt5_timestamp_to_beijing_time(timestamp):
        """
        将MT5时间戳直接转换为北京时间
        
        Args:
            timestamp (int): MT5时间戳
            
        Returns:
            datetime: 北京时间的datetime对象
        """
        # MT5时间戳是UTC时间，北京时间是UTC+8
        utc_time = pd.to_datetime(timestamp, unit='s')
        beijing_time = utc_time + timedelta(hours=8)
        return beijing_time
    
    @staticmethod
    def mt5_time_to_beijing_time(mt5_time):
        """
        将MT5时间转换为北京时间
        
        Args:
            mt5_time (datetime): MT5时间
            
        Returns:
            datetime: 北京时间
        """
        # MT5时间通常是UTC+2或UTC+3（取决于夏令时），北京时间是UTC+8
        # 根据项目经验，MT5时间比北京时间少5小时
        beijing_time = mt5_time + timedelta(hours=5)
        return beijing_time
    
    @staticmethod
    def datetime_to_string(dt, include_milliseconds=False):
        """
        将datetime对象转换为字符串格式
        
        Args:
            dt (datetime): datetime对象
            include_milliseconds (bool): 是否包含毫秒
            
        Returns:
            str: 格式化的时间字符串
        """
        if include_milliseconds:
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 显示到毫秒
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')  # 精确到秒
    
    @staticmethod
    def get_time_diff_seconds(time1, time2):
        """
        计算两个时间之间的差值（秒）
        
        Args:
            time1 (datetime): 时间1
            time2 (datetime): 时间2
            
        Returns:
            float: 时间差（秒）
        """
        return abs((time1 - time2).total_seconds())
    
    @staticmethod
    def is_valid_time_range(mt5_time, max_diff_hours=24):
        """
        检查MT5时间是否在合理范围内
        
        Args:
            mt5_time (datetime): MT5时间
            max_diff_hours (int): 最大时间差（小时）
            
        Returns:
            bool: 时间是否有效
        """
        current_time = datetime.now()
        time_diff = abs((current_time - mt5_time).total_seconds())
        return time_diff <= (max_diff_hours * 3600)

    @staticmethod
    def get_latest_m1_time(symbol="XAUUSD"):
        """
        获取指定品种的最新M1 K线时间
        
        Args:
            symbol (str): 交易品种，默认为"XAUUSD"
            
        Returns:
            tuple: (MT5时间, 北京时间) 或 (None, None) 如果获取失败
        """
        try:
            # 初始化MT5连接（如果尚未连接）
            if not mt5.initialize():
                print("MT5初始化失败")
                return None, None
            
            # 确保品种在市场观察列表中
            mt5.symbol_select(symbol, True)
            
            # 获取最新的M1 K线数据
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            
            if rates is None or len(rates) == 0:
                print("未获取到M1数据")
                return None, None
            
            # 获取MT5时间（UTC时间）
            mt5_time = TimeUtils.mt5_timestamp_to_datetime(rates[0]['time'])
            
            # 计算北京时间
            beijing_time = TimeUtils.mt5_time_to_beijing_time(mt5_time)
            
            return mt5_time, beijing_time
            
        except Exception as e:
            print(f"获取最新M1时间时发生错误: {str(e)}")
            return None, None
        finally:
            # 注意：不关闭MT5连接，因为可能其他地方还需要使用
            pass


def main():
    """测试时间工具类"""
    print("时间工具类测试")
    print("=" * 50)
    
    # 测试获取MT5最新M1时间
    mt5_time, beijing_time = TimeUtils.get_latest_m1_time("XAUUSD")
    
    if mt5_time is not None and beijing_time is not None:
        print(f"MT5最新M1时间: {TimeUtils.datetime_to_string(mt5_time, True)}")
        print(f"对应北京时间: {TimeUtils.datetime_to_string(beijing_time, True)}")
        
        # 计算时间差
        time_diff = TimeUtils.get_time_diff_seconds(mt5_time, beijing_time)
        print(f"时间差: {time_diff:.0f} 秒 ({time_diff/3600:.1f} 小时)")
    else:
        print("无法获取MT5最新M1时间")


if __name__ == "__main__":
    main()