"""
运行所有货币对策略的主程序
同时运行GBPUSD、EURUSD和XAUUSD策略，它们共享全局状态以满足FTMO总体风险控制要求
"""

import threading
import time
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入配置加载器
from com.mlc.utils.config_loader import config

# 导入共享状态
from shared_state import shared_state

def run_gbpusd_strategy():
    """运行GBPUSD策略"""
    try:
        from GBPUSD.GBPUSD真实运行 import run_strategy as gbpusd_run_strategy
        gbpusd_run_strategy()
    except Exception as e:
        print(f"GBPUSD策略执行过程中发生错误: {str(e)}")

def run_eurusd_strategy():
    """运行EURUSD策略"""
    try:
        from EURUSD.EURUSD真实运行 import run_strategy as eurusd_run_strategy
        eurusd_run_strategy()
    except Exception as e:
        print(f"EURUSD策略执行过程中发生错误: {str(e)}")

def run_usdjpy_strategy():
    """运行USDJPY策略"""
    try:
        from USDJPY.USDJPY真实运行 import run_strategy as usdjpy_run_strategy
        usdjpy_run_strategy()
    except Exception as e:
        print(f"USDJPY策略执行过程中发生错误: {str(e)}")

def run_audusd_strategy():
    """运行AUDUSD策略"""
    try:
        from AUDUSD.AUDUSD真实运行 import run_strategy as audusd_run_strategy
        audusd_run_strategy()
    except Exception as e:
        print(f"AUDUSD策略执行过程中发生错误: {str(e)}")

def run_xauusd_strategy():
    """运行XAUUSD策略"""
    try:
        from XAUUSD.XAUUSD真实运行 import FTMORealTimeTrader
        trader = FTMORealTimeTrader()
        trader.run()
    except Exception as e:
        print(f"XAUUSD策略执行过程中发生错误: {str(e)}")



def monitor_global_state():
    """监控全局状态"""
    while True:
        try:
            # 获取全局状态
            stats = shared_state.get_global_stats()
            
            # 打印全局状态
            print(f"[全局状态监控] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  当前余额: {stats['current_balance']:.2f}")
            print(f"  当前净值: {stats['equity']:.2f}")
            print(f"  当日交易次数: {stats['daily_trades']}")
            print(f"  当日盈亏: {stats['daily_pnl']:.2f}")
            print(f"  总盈亏: {stats['total_pnl']:.2f}")
            print("-" * 50)
            
            # 检查风险控制条件
            if stats['daily_pnl'] <= -shared_state.daily_loss_limit:
                print(f"警告: 达到每日最大亏损限制 {shared_state.daily_loss_limit:.2f}")
                
            if stats['current_balance'] - shared_state.initial_balance <= -shared_state.total_loss_limit:
                print(f"警告: 达到总最大亏损限制 {shared_state.total_loss_limit:.2f}")
                
            if stats['equity'] < shared_state.min_equity_limit:
                print(f"警告: 净值低于最低限制 {shared_state.min_equity_limit:.2f}")
            
            # 每30秒打印一次状态
            time.sleep(30)
            
        except Exception as e:
            print(f"[全局状态监控] 监控全局状态时发生错误: {str(e)}")
            time.sleep(30)

def main():
    """主函数"""
    print("启动多货币对交易策略协调器...")
    print(f"初始资金: {shared_state.initial_balance:.2f}")
    print(f"每日最大亏损: {shared_state.daily_loss_limit:.2f} (5%)")
    print(f"总最大亏损: {shared_state.total_loss_limit:.2f} (10%)")
    print(f"最低净值限制: {shared_state.min_equity_limit:.2f} (90%)")
    print("=" * 50)
    
    # 创建并启动GBPUSD策略线程
    gbpusd_thread = threading.Thread(target=run_gbpusd_strategy, name="GBPUSD_Thread")
    gbpusd_thread.daemon = True  # 设置为守护线程
    
    # 创建并启动EURUSD策略线程
    eurusd_thread = threading.Thread(target=run_eurusd_strategy, name="EURUSD_Thread")
    eurusd_thread.daemon = True  # 设置为守护线程


    # 创建并启动USDJPY策略线程
    usdjpy_thread = threading.Thread(target=run_usdjpy_strategy, name="USDJPY_Thread")
    usdjpy_thread.daemon = True  # 设置为守护线程
    
    # 创建并启动AUDUSD策略线程
    audusd_thread = threading.Thread(target=run_audusd_strategy, name="AUDUSD_Thread")
    audusd_thread.daemon = True  # 设置为守护线程
    
    # # 创建并启动XAUUSD策略线程
    # xauusd_thread = threading.Thread(target=run_xauusd_strategy, name="XAUUSD_Thread")
    # xauusd_thread.daemon = True  # 设置为守护线程
    
    # 创建并启动全局状态监控线程
    monitor_thread = threading.Thread(target=monitor_global_state, name="Monitor_Thread")
    monitor_thread.daemon = True  # 设置为守护线程
    
    # 启动所有线程
    gbpusd_thread.start()
    eurusd_thread.start()
    usdjpy_thread.start()
    audusd_thread.start()
    # xauusd_thread.start()
    monitor_thread.start()
    
    print("所有策略已启动，正在运行中...")
    print("按 Ctrl+C 停止运行")
    print("=" * 50)
    
    try:
        # 主线程持续运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断程序执行")
        print("正在关闭所有策略...")
    except Exception as e:
        print(f"主程序执行过程中发生错误: {str(e)}")
    finally:
        print("程序已退出")

if __name__ == "__main__":
    main()