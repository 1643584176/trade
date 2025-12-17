import MetaTrader5 as mt5
import logging
from datetime import datetime, date, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_account_info():
    """
    查询账户信息，包括今日盈亏、当前余额、昨日余额和正在进行的交易
    """
    try:
        # 初始化MT5连接
        if not mt5.initialize():
            logger.error("MT5初始化失败")
            return None
        
        # 获取账户信息
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("无法获取账户信息")
            mt5.shutdown()
            return None
        
        # 当前余额
        current_balance = account_info.balance
        current_equity = account_info.equity
        
        # 获取今日日期和昨日日期
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        # 获取今天的时间范围
        today_start = datetime(today.year, today.month, today.day)
        today_end = datetime.now()
        
        # 获取昨天的时间范围
        yesterday_start = datetime(yesterday.year, yesterday.month, yesterday.day)
        yesterday_end = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)
        
        # 获取今日历史交易记录
        today_deals = mt5.history_deals_get(today_start, today_end)
        today_profit = 0
        if today_deals is not None:
            for deal in today_deals:
                # 确认是今天的交易
                deal_time = datetime.fromtimestamp(deal.time)
                if deal_time.date() == today:
                    today_profit += deal.profit
        
        # 获取昨日历史交易记录
        yesterday_deals = mt5.history_deals_get(yesterday_start, yesterday_end)
        yesterday_profit = 0
        if yesterday_deals is not None:
            for deal in yesterday_deals:
                # 确认是昨天的交易
                deal_time = datetime.fromtimestamp(deal.time)
                if deal_time.date() == yesterday:
                    yesterday_profit += deal.profit
        
        # 昨日余额 = 当前余额 - 今日盈亏
        yesterday_balance = current_balance - today_profit
        
        # 获取当前持仓信息
        positions = mt5.positions_get()
        floating_profit = 0
        positions_count = 0
        max_floating_profit = 0  # 最大浮动盈利
        max_floating_loss = 0    # 最大浮动亏损
        if positions is not None:
            positions_count = len(positions)
            for position in positions:
                floating_profit += position.profit
                # 更新最大浮动盈亏
                if position.profit > max_floating_profit:
                    max_floating_profit = position.profit
                if position.profit < max_floating_loss:
                    max_floating_loss = position.profit
        
        # 准备返回数据
        result = {
            "current_balance": current_balance,           # 当前余额
            "current_equity": current_equity,             # 当前净值
            "yesterday_balance": yesterday_balance,       # 昨日余额
            "today_profit": today_profit,                 # 今日已实现盈亏
            "floating_profit": floating_profit,           # 持仓浮动盈亏
            "total_today_profit": today_profit + floating_profit,  # 今日总盈亏
            "positions_count": positions_count,            # 持仓数量
            "max_floating_profit": max_floating_profit,   # 最大浮动盈利
            "max_floating_loss": max_floating_loss        # 最大浮动亏损
        }
        
        mt5.shutdown()
        return result
        
    except Exception as e:
        logger.error(f"查询账户信息时出错: {str(e)}")
        try:
            mt5.shutdown()
        except:
            pass
        return None

def main():
    """
    主函数
    """
    logger.info("开始查询账户信息...")
    
    account_info = get_account_info()
    
    if account_info is None:
        logger.error("无法获取账户信息")
        return
    
    # 打印账户信息
    print("=" * 50)
    print("账户信息查询结果")
    print("=" * 50)
    print(f"当前余额: {account_info['current_balance']:.2f} USD")
    print(f"当前净值: {account_info['current_equity']:.2f} USD")
    print(f"昨日余额: {account_info['yesterday_balance']:.2f} USD")
    print(f"今日已实现盈亏: {account_info['today_profit']:.2f} USD")
    print(f"持仓浮动盈亏: {account_info['floating_profit']:.2f} USD")
    print(f"今日总盈亏: {account_info['total_today_profit']:.2f} USD")
    print(f"最大浮动盈利: {account_info['max_floating_profit']:.2f} USD")
    print(f"最大浮动亏损: {account_info['max_floating_loss']:.2f} USD")
    print(f"持仓数量: {account_info['positions_count']} 个")
    
    # 检查是否超过FTMO限制
    if account_info['total_today_profit'] < -4500:
        print(f"\n警告：今日总亏损 {abs(account_info['total_today_profit']):.2f} USD 已超过FTMO最大允许亏损 4500 USD")
    else:
        print(f"\n今日亏损在限制范围内，还可以继续交易")

if __name__ == "__main__":
    main()