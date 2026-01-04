import MetaTrader5 as mt5
import datetime
from typing import Dict, Optional


def get_mt5_account_stats() -> Optional[Dict]:
    """
    查询MT5账户的核心财务数据：
    - 原始初始余额（账户开立本金）、当前本金、剩余余额
    - 持仓盈亏、今日盈亏（平仓+持仓）、累计盈亏（平仓+持仓）
    返回：包含所有数据的字典，失败时返回None
    """
    # 1. 初始化并连接MT5
    if not mt5.initialize():
        print(f"MT5连接失败，错误代码：{mt5.last_error()}")
        mt5.shutdown()
        return None

    # 2. 获取账户基础信息
    account_info = mt5.account_info()
    if account_info is None:
        print(f"获取账户信息失败，错误代码：{mt5.last_error()}")
        mt5.shutdown()
        return None

    # 基础数据（当前本金、剩余余额、持仓浮动盈亏）
    current_balance = account_info.balance  # 当前本金（扣除已平仓盈亏）
    equity = account_info.equity  # 剩余余额（当前本金+持仓盈亏）
    position_profit = account_info.profit  # 持仓浮动盈亏

    # 3. 统计平仓订单盈亏（今日/累计）
    from_date = datetime.datetime(2000, 1, 1)
    to_date = datetime.datetime.now()
    history_deals = mt5.history_deals_get(from_date, to_date)

    total_closed_profit = 0.0  # 累计平仓盈亏
    today_closed_profit = 0.0  # 今日平仓盈亏
    today = datetime.datetime.now().date()

    if history_deals is not None and len(history_deals) > 0:
        for deal in history_deals:
            # 仅统计平仓订单（排除挂单、开仓等无效订单）
            if deal.entry == mt5.DEAL_ENTRY_OUT:
                total_closed_profit += deal.profit
                # 今日平仓订单
                deal_time = datetime.datetime.fromtimestamp(deal.time)
                if deal_time.date() == today:
                    today_closed_profit += deal.profit
    else:
        print("无历史平仓订单，平仓盈亏默认0")

    # 关键修正：计算原始初始余额（账户开立时的本金）
    original_initial_balance = current_balance - total_closed_profit
    # 今日总盈亏 = 今日平仓盈亏 + 持仓浮动盈亏
    today_total_profit = today_closed_profit + position_profit
    # 累计总盈亏 = 累计平仓盈亏 + 持仓浮动盈亏
    total_total_profit = total_closed_profit + position_profit

    # 整理最终数据
    stats = {
        "原始初始余额（开立本金）": round(original_initial_balance, 2),
        "当前本金（balance）": round(current_balance, 2),
        "剩余余额（equity）": round(equity, 2),
        "持仓盈亏（浮动）": round(position_profit, 2),
        "今日平仓盈亏": round(today_closed_profit, 2),
        "今日总盈亏（平仓+持仓）": round(today_total_profit, 2),
        "累计平仓盈亏": round(total_closed_profit, 2),
        "累计总盈亏（平仓+持仓）": round(total_total_profit, 2)
    }

    # 关闭连接
    mt5.shutdown()
    return stats


def check_and_close_positions_by_profit_threshold(threshold_percentage=4.5):
    """
    检查账户持仓盈亏百分比，如果大于等于指定百分比则平仓
    
    参数:
        threshold_percentage (float): 平仓阈值百分比，默认4.5%
    返回:
        bool: 是否执行了平仓操作
    """
    import math
    
    account_stats = get_mt5_account_stats()
    if not account_stats:
        return False
    
    current_balance = account_stats["当前本金（balance）"]
    position_profit = account_stats["持仓盈亏（浮动）"]
    
    # 计算持仓盈亏占当前本金的百分比
    if current_balance != 0:
        profit_percentage = abs(position_profit) / current_balance * 100
        
        # 只有在亏损（position_profit < 0）且比例大于等于4.5%时才平仓
        if position_profit < 0 and profit_percentage >= threshold_percentage:
            return True
        else:
            return False
    else:
        return False


def continuous_monitoring(interval=10):
    """
    持续监控账户，每interval秒检查一次
    
    参数:
        interval (int): 检查间隔时间，单位秒，默认10秒
    """
    import time
    import os
    import sys
    
    # 添加一键操作目录到Python路径，以便导入关闭持仓脚本
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yijian_dir = os.path.join(script_dir, "一键操作")
    sys.path.insert(0, yijian_dir)
    
    print(f"开始持续监控，每{interval}秒检查一次账户...")
    
    try:
        while True:
            account_stats = get_mt5_account_stats()
            if account_stats:
                current_balance = account_stats["当前本金（balance）"]
                position_profit = account_stats["持仓盈亏（浮动）"]
                equity = account_stats["剩余余额（equity）"]
                
                # 计算持仓盈亏占当前本金的百分比
                if current_balance != 0:
                    profit_percentage = abs(position_profit) / current_balance * 100
                    
                    # 输出一行简洁的日志
                    status = "需平仓" if position_profit < 0 and profit_percentage >= 4.5 else "正常"  # 只有在亏损且比例>=4.5%时才需要平仓
                    print(f"时间: {datetime.datetime.now().strftime('%H:%M:%S')} | 本金: {current_balance:.2f} | 余额: {equity:.2f} | 持仓盈亏: {position_profit:.2f} | 盈亏比例: {profit_percentage:.2f}% | 状态: {status}")
                    
                    # 如果是亏损且盈亏比例大于等于4.5%，执行平仓
                    if position_profit < 0 and profit_percentage >= 4.5:
                        print("触发风控规则，正在关闭所有持仓...")
                        
                        # 导入并执行关闭所有持仓的函数
                        try:
                            # 由于关闭持仓脚本是独立的，我们需要直接执行它
                            import importlib.util
                            spec = importlib.util.spec_from_file_location("close_positions", os.path.join(yijian_dir, "关闭所有交易持仓.py"))
                            close_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(close_module)
                            
                            # 调用close_all_positions函数
                            if hasattr(close_module, 'close_all_positions'):
                                close_module.close_all_positions()
                            else:
                                print("警告：无法找到close_all_positions函数")
                        except Exception as e:
                            print(f"执行平仓脚本时出错: {e}")
                        
                        # 创建一个标记文件，通知其他交易脚本暂停运行
                        try:
                            with open("暂停交易.flag", "w", encoding="utf-8") as f:
                                f.write(f"{datetime.datetime.now()}\n")
                            print("已创建暂停交易标记文件，通知其他交易脚本暂停")
                            
                            # 等待一段时间后删除标记文件，允许其他交易脚本恢复运行
                            import time as time_module
                            time_module.sleep(10)  # 等待10秒，确保其他脚本收到信号
                            
                            import os
                            if os.path.exists("暂停交易.flag"):
                                os.remove("暂停交易.flag")
                                print("已删除暂停交易标记文件，其他交易脚本可恢复运行")
                        except Exception as e:
                            print(f"创建/删除暂停标记文件时出错: {e}")
                else:
                    print(f"时间: {datetime.datetime.now().strftime('%H:%M:%S')} | 本金为0，无法计算百分比")
            else:
                print(f"时间: {datetime.datetime.now().strftime('%H:%M:%S')} | 获取账户信息失败")
            
            # 等待指定时间后再次检查
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n监控已停止（用户中断）")


if __name__ == "__main__":
    import time
    
    # 运行一次检查（保持原有功能）
    account_stats = get_mt5_account_stats()
    if account_stats:
        print("=== MT5账户财务数据（修正版） ===")
        for key, value in account_stats.items():
            print(f"{key}: {value:.2f}")
        
        print(f"如果亏损（浮动）大于等于账户的4.5%就平仓")
        
        # 检查是否需要平仓
        should_close = check_and_close_positions_by_profit_threshold(4.5)
        if should_close:
            print("根据风控规则，建议执行平仓操作")
        else:
            print("根据风控规则，无需执行平仓操作")
        
        print("\n启动每10秒一次的持续监控...")
        continuous_monitoring(10)