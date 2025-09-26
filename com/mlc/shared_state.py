"""
全局共享状态模块
用于在多个货币对策略之间共享风险控制状态
"""

import threading
import MetaTrader5 as mt5
from datetime import datetime, date

class SharedState:
    def __init__(self):
        # 初始化MT5连接以获取账户信息
        if not mt5.initialize():
            print("MT5初始化失败，使用默认账户信息")
            self.initial_balance = 10000.0  # 默认初始资金10000美元
        else:
            # 获取账户信息
            account_info = mt5.account_info()
            if account_info is None:
                print("无法获取账户信息，使用默认账户信息")
                self.initial_balance = 10000.0
            else:
                self.initial_balance = float(account_info.balance)
            mt5.shutdown()  # 初始化完成后关闭连接
        
        self.current_balance = self.initial_balance  # 当前余额
        self.equity = self.initial_balance  # 当前净值
        
        # 风险控制参数
        self.daily_loss_limit = self.initial_balance * 0.05  # 每日最大亏损5%
        self.total_loss_limit = self.initial_balance * 0.10  # 总最大亏损10%
        self.min_equity_limit = self.initial_balance * 0.90   # 最低净值限制90%
        
        # 交易统计
        self.daily_trades = 0  # 当日交易次数
        self.daily_pnl = 0.0   # 当日盈亏
        self.total_pnl = 0.0   # 总盈亏
        
        # 日期跟踪
        self.last_reset_date = date.today()
        
        # 线程锁
        self._lock = threading.Lock()
    
    def update_account_info(self):
        """
        更新账户信息
        从MT5获取最新的账户余额和净值
        """
        try:
            # 确保MT5已初始化
            if not mt5.initialize():
                print("MT5初始化失败，无法更新账户信息")
                return
            
            # 获取账户信息
            account_info = mt5.account_info()
            if account_info is not None:
                with self._lock:
                    self.current_balance = float(account_info.balance)
                    self.equity = float(account_info.equity)
            else:
                print("无法获取账户信息")
            
            # 注意：这里不关闭MT5连接，因为其他地方可能还需要使用
            # mt5.shutdown()
                
        except Exception as e:
            print(f"更新账户信息时发生错误: {str(e)}")
    
    def check_daily_reset(self, current_date=None):
        """
        检查是否需要重置每日统计
        
        Args:
            current_date (date): 当前日期，默认为今天
            
        Returns:
            bool: 是否需要重置
        """
        if current_date is None:
            current_date = date.today()
            
        if current_date > self.last_reset_date:
            with self._lock:
                self.daily_trades = 0
                self.daily_pnl = 0.0
                self.last_reset_date = current_date
            return True
        return False
    
    def update_daily_stats(self, trades_count=0, profit=0.0):
        """
        更新每日交易统计
        
        Args:
            trades_count (int): 交易次数变化
            profit (float): 盈亏变化
        """
        with self._lock:
            self.daily_trades += trades_count
            self.daily_pnl += profit
            self.total_pnl += profit
            self.current_balance += profit
            # equity将在各策略中实时更新
    
    def can_open_position(self):
        """
        检查是否可以开仓（满足风控条件）
        
        Returns:
            bool: 是否可以开仓
        """
        # 更新账户信息
        self.update_account_info()
        
        with self._lock:
            # 检查各种风控条件
            if self.daily_pnl <= -self.daily_loss_limit:
                return False  # 达到每日最大亏损限制
                
            if self.current_balance - self.initial_balance <= -self.total_loss_limit:
                return False  # 达到总最大亏损限制
                
            if self.equity < self.min_equity_limit:
                return False  # 净值低于最低限制
                
            return True
    
    def get_global_stats(self):
        """
        获取全局状态统计信息
        
        Returns:
            dict: 包含所有全局状态信息的字典
        """
        # 更新账户信息
        self.update_account_info()
        
        with self._lock:
            return {
                'current_balance': self.current_balance,
                'equity': self.equity,
                'daily_trades': self.daily_trades,
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'initial_balance': self.initial_balance,
                'daily_loss_limit': self.daily_loss_limit,
                'total_loss_limit': self.total_loss_limit,
                'min_equity_limit': self.min_equity_limit
            }

# 创建全局共享状态实例
shared_state = SharedState()