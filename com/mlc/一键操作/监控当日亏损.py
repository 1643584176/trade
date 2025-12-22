import MetaTrader5 as mt5
import logging
from datetime import datetime, date, timedelta
import time
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyLossMonitor:
    """
    监控当日亏损的类，当日亏损超过设定阈值时发出警报
    """
    
    def __init__(self, max_loss_threshold=4000):
        """
        初始化监控器
        
        Args:
            max_loss_threshold (float): 最大允许亏损阈值，默认4000美元
        """
        self.max_loss_threshold = max_loss_threshold
        self.initial_balance = None
        self.min_balance = None
        self.today = date.today()
        self.stop_flag_file = "stop_trading.flag"  # 停止交易的标志文件
        
    def get_account_info(self):
        """
        获取账户信息
        
        Returns:
            dict: 包含账户信息的字典，如果失败返回None
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
            
            # 当前余额和净值
            current_balance = account_info.balance
            current_equity = account_info.equity
            
            # 获取今日日期和昨日日期
            today = date.today()
            
            # 获取今天的时间范围
            today_start = datetime(today.year, today.month, today.day)
            today_end = datetime.now()
            
            # 获取今日历史交易记录
            today_deals = mt5.history_deals_get(today_start, today_end)
            today_profit = 0
            if today_deals is not None:
                for deal in today_deals:
                    # 确认是今天的交易
                    deal_time = datetime.fromtimestamp(deal.time)
                    if deal_time.date() == today:
                        today_profit += deal.profit
            
            # 今日初始余额 = 当前余额 - 今日已实现盈亏
            today_initial_balance = current_balance - today_profit
            
            # 获取当前持仓信息
            positions = mt5.positions_get()
            floating_profit = 0
            if positions is not None:
                for position in positions:
                    floating_profit += position.profit
            
            mt5.shutdown()
            
            return {
                "current_balance": current_balance,           # 当前余额
                "current_equity": current_equity,             # 当前净值
                "today_initial_balance": today_initial_balance, # 今日初始余额
                "today_profit": today_profit,                 # 今日已实现盈亏
                "floating_profit": floating_profit,           # 持仓浮动盈亏
                "total_today_profit": today_profit + floating_profit  # 今日总盈亏
            }
            
        except Exception as e:
            logger.error(f"查询账户信息时出错: {str(e)}")
            try:
                mt5.shutdown()
            except:
                pass
            return None
    
    def update_balance_tracking(self):
        """
        更新余额跟踪信息
        
        Returns:
            bool: 更新成功返回True，否则返回False
        """
        account_info = self.get_account_info()
        if account_info is None:
            return False
            
        current_balance = account_info["current_balance"]
        today_initial_balance = account_info["today_initial_balance"]
        
        # 如果是第一次调用或者日期变更，重置初始余额
        if self.initial_balance is None or date.today() != self.today:
            self.initial_balance = today_initial_balance
            self.min_balance = current_balance
            self.today = date.today()
        else:
            # 更新最小余额
            if current_balance < self.min_balance:
                self.min_balance = current_balance
                
        return True
    
    def get_current_loss(self):
        """
        获取当前亏损值
        
        Returns:
            float: 当前亏损值，如果未初始化返回None
        """
        if self.initial_balance is None or self.min_balance is None:
            return None
            
        return self.initial_balance - self.min_balance
    
    def is_loss_exceeded(self):
        """
        检查是否超过最大亏损阈值
        
        Returns:
            tuple: (是否超过阈值(bool), 当前亏损值(float))
        """
        self.update_balance_tracking()
        current_loss = self.get_current_loss()
        
        if current_loss is None:
            return False, 0
            
        is_exceeded = current_loss > self.max_loss_threshold
        return is_exceeded, current_loss
    
    def close_all_positions(self):
        """
        强制平仓所有持仓
        """
        try:
            # 初始化MT5连接
            if not mt5.initialize():
                logger.error("MT5初始化失败，无法执行强制平仓")
                return False
            
            # 获取所有当前持仓
            positions = mt5.positions_get()
            if not positions:
                print("当前无任何持仓")
                mt5.shutdown()
                return True

            print(f"发现 {len(positions)} 个持仓，开始强制平仓...")

            # 遍历每个持仓，逐一平仓
            for position in positions:
                symbol = position.symbol
                volume = position.volume
                position_type = position.type
                position_ticket = position.ticket

                # 构建平仓请求
                close_type = mt5.ORDER_TYPE_SELL if position_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                current_price = mt5.symbol_info_tick(symbol).bid if position_type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": close_type,
                    "position": position_ticket,
                    "price": current_price,
                    "deviation": 3,
                    "comment": "强制平仓",
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }

                # 发送平仓请求
                result = mt5.order_send(close_request)

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ 强制平仓成功！品种：{symbol}，订单号：{position_ticket}，手数：{volume}，平仓价格：{current_price:.5f}")
                else:
                    print(f"❌ 强制平仓失败！品种：{symbol}，订单号：{position_ticket}，错误代码：{result.retcode}，原因：{result.comment}")

            mt5.shutdown()
            return True
            
        except Exception as e:
            logger.error(f"强制平仓时出错: {str(e)}")
            try:
                mt5.shutdown()
            except:
                pass
            return False
    
    def check_and_alert(self):
        """
        检查并发出警报
        
        Returns:
            bool: 是否超过阈值
        """
        account_info = self.get_account_info()
        if account_info is None:
            logger.error("无法获取账户信息")
            return False
            
        is_exceeded, current_loss = self.is_loss_exceeded()
        
        # 在一行显示所有信息
        print(f"今日初始余额: {account_info['today_initial_balance']:.2f} USD, "
              f"当前余额: {account_info['current_balance']:.2f} USD, "
              f"当前净值: {account_info['current_equity']:.2f} USD, "
              f"已实现盈亏: {account_info['today_profit']:.2f} USD, "
              f"持仓盈亏: {account_info['floating_profit']:.2f} USD, "
              f"今日总盈亏: {account_info['total_today_profit']:.2f} USD, "
              f"当前回撤: {current_loss:.2f} USD, "
              f"状态: {'⚠️ 超限' if is_exceeded else '✅ 正常'}")
        
        # 检查是否需要强制平仓（当前余额小于今日初始余额且亏损超过4000）
        today_initial_balance = account_info['today_initial_balance']
        current_balance = account_info['current_balance']
        current_equity = account_info['current_equity']
        
        # 如果当前净值小于今日初始余额且亏损超过4000，则强制平仓
        # 或者如果当前余额小于今日初始余额且亏损超过4000美元，则强制平仓
        loss_amount = today_initial_balance - current_equity
        balance_loss_amount = today_initial_balance - current_balance
        
        if (current_equity < today_initial_balance and loss_amount > 4000) or \
           (current_balance < today_initial_balance and balance_loss_amount > 4000):
            print(f"🚨 触发强制平仓条件：当前净值 {current_equity:.2f} USD < 今日初始余额 {today_initial_balance:.2f} USD，且亏损 {loss_amount:.2f} USD > 4000 USD")
            self.close_all_positions()
            
            # 创建停止交易的标志文件，通知其他交易程序停止运行
            try:
                with open(self.stop_flag_file, "w") as f:
                    f.write(f"Trading stopped at {datetime.now()}\n")
                    f.write(f"Loss amount: {loss_amount:.2f} USD\n")
                print("🛑 已创建停止交易标志文件，通知所有交易程序停止运行")
            except Exception as e:
                logger.error(f"创建停止交易标志文件失败: {str(e)}")
        
        # 如果当前余额大于110020，则强制平仓
        elif current_balance > 110020:
            print(f"🚨 触发盈利平仓条件：当前余额 {current_balance:.2f} USD > 110020 USD")
            self.close_all_positions()
            
            # 创建停止交易的标志文件，通知其他交易程序停止运行
            try:
                with open(self.stop_flag_file, "w") as f:
                    f.write(f"Trading stopped at {datetime.now()}\n")
                    f.write(f"Profit target reached: {current_balance:.2f} USD\n")
                print("🛑 已创建停止交易标志文件，通知所有交易程序停止运行")
            except Exception as e:
                logger.error(f"创建停止交易标志文件失败: {str(e)}")
        
        if is_exceeded:
            logger.warning(f"⚠️  当日亏损已超过阈值！当前回撤: {current_loss:.2f} USD")
        else:
            logger.info(f"✅ 当前回撤在限制范围内。当前回撤: {current_loss:.2f} USD")
            
        return is_exceeded

def main():
    """
    主函数 - 示例用法
    """
    # 创建监控器实例，设置最大亏损阈值为4000美元
    monitor = DailyLossMonitor(max_loss_threshold=4000)
    
    print("开始监控账户亏损情况，每10秒检查一次...")
    print("格式: 今日初始余额, 当前余额, 持仓盈亏, 当前回撤, 状态")
    
    try:
        while True:
            # 检查并发出警报
            is_exceeded = monitor.check_and_alert()
            
            if is_exceeded:
                print("🚨 建议停止交易以控制风险！")
            
            # 等待10秒后再次检查
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
        
        # 清理停止交易的标志文件
        stop_flag_file = "stop_trading.flag"
        if os.path.exists(stop_flag_file):
            try:
                os.remove(stop_flag_file)
                print("🧹 已清理停止交易标志文件")
            except Exception as e:
                logger.error(f"清理停止交易标志文件失败: {str(e)}")

if __name__ == "__main__":
    main()