import MetaTrader5 as mt5
import logging
from datetime import datetime
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balance_monitor.log', encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

def check_balance_and_close_positions():
    """
    检查账户余额并根据条件平仓
    条件1: 如果当前余额 > 110020，则平掉所有仓位
    条件2: 如果当前余额 < 初始余额 并且 亏损 >= 450则平掉所有仓位
    """
    
    # 初始化MT5连接
    if not mt5.initialize():
        logger.error("MT5初始化失败")
        return False
    
    try:
        # 获取账户信息
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("无法获取账户信息")
            return False
        
        # 获取当前余额和权益
        current_balance = account_info.balance
        equity = account_info.equity
        currency = account_info.currency
        
        logger.info(f"账户查询 - 余额: {current_balance:.2f}{currency}, 权益: {equity:.2f}{currency}")
        
        # 获取所有持仓
        positions = mt5.positions_total()
        logger.info(f"当前持仓数量: {positions}")
        
        if positions > 0:
            # 获取持仓详情
            open_positions = mt5.positions_get()
            if open_positions:
                total_profit = sum(pos.profit for pos in open_positions)
                logger.info(f"持仓总盈亏: {total_profit:.2f}{currency}")
        
                # 获取程序启动时的初始余额作为基准
        global initial_balance
        if 'initial_balance' not in globals():
            initial_balance = current_balance
            logger.info(f"设置初始余额基准: {initial_balance:.2f}{currency}")
        
        base_balance = initial_balance
        
        # 检查条件1: 如果当前余额 > 110020，则平掉所有仓位
        if current_balance > 110020.0:
            logger.info(f"当前余额 {current_balance:.2f} 超过阈值 110020.0，执行平仓操作")
            close_all_positions()
            return True
        
        # 检查条件2: 如果当前余额 < 基准余额 并且 差值 >= 450，则平掉所有仓位
        balance_diff = base_balance - current_balance
        if current_balance < base_balance and abs(balance_diff) >= 450.0:
            logger.info(f"当前余额 {current_balance:.2f} 低于基准余额 {base_balance:.2f}，且差值 {abs(balance_diff):.2f} >= 450，执行平仓操作")
            close_all_positions()
            return True
        
        logger.info("余额检查完成，未达到平仓条件")
        return False
        
    except Exception as e:
        logger.error(f"检查余额和执行平仓操作时发生异常: {str(e)}")
        return False
    finally:
        # 关闭MT5连接
        mt5.shutdown()

def close_all_positions():
    """
    平掉所有持仓
    """
    try:
        # 初始化MT5连接
        if not mt5.initialize():
            logger.error("MT5初始化失败")
            return False
        
        # 获取所有持仓
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            logger.info("没有持仓需要平仓")
            return True
        
        logger.info(f"发现 {len(positions)} 个持仓，开始平仓...")
        
        closed_count = 0
        for position in positions:
            # 创建平仓请求
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": position.magic,
                "comment": "风控自动平仓",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # 发送平仓请求
            result = mt5.order_send(close_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"平仓成功，订单号: {result.order}, 交易品种: {position.symbol}, 手数: {position.volume}")
                closed_count += 1
            else:
                logger.error(f"平仓失败，错误码: {result.retcode}, 交易品种: {position.symbol}")
        
        logger.info(f"平仓操作完成，共平掉 {closed_count} 个持仓")
        return True
        
    except Exception as e:
        logger.error(f"平仓操作异常: {str(e)}")
        return False
    finally:
        # 关闭MT5连接
        mt5.shutdown()

def check_balance_with_custom_thresholds(today_balance, remaining_balance, 
                                       profit_threshold=110020.0, 
                                       loss_threshold=450.0):
    """
    使用自定义阈值检查余额并决定是否平仓
    
    参数:
        today_balance (float): 今日余额
        remaining_balance (float): 剩余余额
        profit_threshold (float): 盈利平仓阈值，默认110020
        loss_threshold (float): 亏损平仓阈值，默认450
    
    返回:
        bool: 是否执行了平仓操作
    """
    logger.info(f"今日余额: {today_balance:.2f}, 剩余余额: {remaining_balance:.2f}")
    
    # 条件1: 如果今日余额 > 盈利阈值，则平掉所有仓位
    if today_balance > profit_threshold:
        logger.info(f"今日余额 {today_balance:.2f} 超过盈利阈值 {profit_threshold:.2f}，需要平仓")
        return True
    
    # 条件2: 如果今日余额 < 剩余余额 且 差值 >= 亏损阈值，则平掉所有仓位
    balance_diff = remaining_balance - today_balance
    if today_balance < remaining_balance and balance_diff >= loss_threshold:
        logger.info(f"今日余额 {today_balance:.2f} 低于剩余余额 {remaining_balance:.2f}，且亏损 {balance_diff:.2f} >= {loss_threshold:.2f}，需要平仓")
        return True
    
    logger.info("余额检查完成，未达到平仓条件")
    return False

def monitor_account_and_manage_risk():
    """
    监控账户并进行风险管理
    """
    logger.info("开始账户监控和风险管理...")
    
    # 获取当前账户信息
    if not mt5.initialize():
        logger.error("MT5初始化失败")
        return
    
    try:
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("无法获取账户信息")
            return
        
        current_balance = account_info.balance
        equity = account_info.equity
        currency = account_info.currency
        
        # 获取程序启动时的初始余额作为基准
        global initial_balance
        if 'initial_balance' not in globals():
            initial_balance = current_balance
            logger.info(f"设置初始余额基准: {initial_balance:.2f}{currency}")
        
        today_balance = current_balance
        remaining_balance = initial_balance  # 或者是其他基准值
        
        logger.info(f"账户状态 - 余额: {today_balance:.2f}{currency}, 权益: {equity:.2f}{currency}, 初始资金: {remaining_balance:.2f}{currency}")
        
        # 检查是否需要平仓
        should_close = check_balance_with_custom_thresholds(
            today_balance, 
            remaining_balance
        )
        
        if should_close:
            logger.info("触发风控条件，开始平仓...")
            close_all_positions()
        else:
            logger.info("账户状态正常，无需平仓")
            
    finally:
        mt5.shutdown()

def continuous_monitoring():
    """
    持续监控账户，每分钟检查一次
    以程序启动时的余额作为基准进行风控判断
    """
    logger.info("启动持续监控模式，每分钟检查一次账户余额...")
    
    try:
        while True:
            logger.info("执行定期账户余额检查...")
            check_balance_and_close_positions()
            
            # 等待60秒后再次检查
            logger.info("等待60秒后进行下次检查...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("监控已停止（用户中断）")
    except Exception as e:
        logger.error(f"监控过程中发生异常: {str(e)}")

if __name__ == "__main__":
    # 运行一次检查
    # check_balance_and_close_positions()
    
    # 或者使用持续监控模式
    continuous_monitoring()