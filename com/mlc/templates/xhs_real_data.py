import pandas as pd
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 遵循项目规范：使用mt5.init()进行初始化连接，不使用账户密码登录
# 参考项目内存中的MT5连接方式规范

class XiaohongshuRealDataGenerator:
    """
    小红书内容生成器（基于真实MT5交易数据）
    """
    
    def __init__(self):
        """
        初始化内容生成器
        """
        self.trade_history = None
    
    def connect_to_mt5(self):
        """
        连接到MT5平台
        
        返回:
            bool: 连接是否成功
        """
        try:
            # 初始化MT5连接
            if not mt5.initialize():
                logger.error(f"MT5初始化失败: {mt5.last_error()}")
                return False
                
            logger.info("MT5连接成功")
            return True
        except Exception as e:
            logger.error(f"连接MT5异常: {str(e)}")
            return False
    
    def fetch_trade_history(self, days_back=7):
        """
        获取指定天数内的交易历史
        
        参数:
            days_back (int): 查询最近多少天的交易记录
            
        返回:
            list: 交易记录列表
        """
        try:
            # 计算日期范围
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # 获取交易历史
            deals = mt5.history_deals_get(from_date, to_date)
            if deals is None:
                logger.error(f"获取交易历史失败: {mt5.last_error()}")
                return []
            
            # 转换为列表格式
            trade_list = []
            for deal in deals:
                trade_list.append({
                    'ticket': deal.ticket,
                    'timestamp': datetime.fromtimestamp(deal.time),
                    'symbol': deal.symbol,
                    'type': deal.type,
                    'volume': deal.volume,
                    'price': deal.price,
                    'commission': deal.commission,
                    'swap': deal.swap,
                    'profit': deal.profit,
                    'comment': deal.comment
                })
            
            logger.info(f"成功获取{len(trade_list)}条交易记录")
            return trade_list
        except Exception as e:
            logger.error(f"获取交易历史异常: {str(e)}")
            return []
    
    def calculate_daily_stats(self, date=None):
        """
        计算指定日期的交易统计数据
        
        参数:
            date (datetime): 指定日期，默认为今天
            
        返回:
            dict: 统计数据
        """
        try:
            if self.trade_history is None:
                logger.error("没有可用的交易历史数据")
                return {}
            
            if date is None:
                date = datetime.now().date()
            elif isinstance(date, datetime):
                date = date.date()
            
            # 筛选指定日期的交易
            daily_trades = [trade for trade in self.trade_history 
                          if trade['timestamp'].date() == date]
            
            if not daily_trades:
                logger.warning(f"{date} 没有交易记录")
                return {}
            
            # 计算统计数据
            total_profit = sum(trade['profit'] for trade in daily_trades)
            buy_trades = [t for t in daily_trades if t['type'] in [0, 6]]  # BUY, BUY_BY
            sell_trades = [t for t in daily_trades if t['type'] in [1, 7]]  # SELL, SELL_BY
            
            buy_profit = sum(t['profit'] for t in buy_trades)
            sell_profit = sum(t['profit'] for t in sell_trades)
            
            # 计算胜率
            profitable_trades = [t for t in daily_trades if t['profit'] > 0]
            win_rate = (len(profitable_trades) / len(daily_trades)) * 100 if daily_trades else 0
            
            stats = {
                'date': date,
                'total_trades': len(daily_trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_profit': total_profit,
                'buy_profit': buy_profit,
                'sell_profit': sell_profit,
                'win_rate': win_rate,
                'trades': daily_trades
            }
            
            return stats
        except Exception as e:
            logger.error(f"计算统计数据异常: {str(e)}")
            return {}
    
    def generate_daily_report(self, date=None):
        """
        生成每日交易报告，适合发布到小红书
        
        参数:
            date (datetime): 指定日期，默认为今天
            
        返回:
            str: 格式化的内容
        """
        try:
            # 获取交易数据
            if self.trade_history is None:
                logger.error("没有可用的交易历史数据")
                return ""
            
            # 计算统计数据
            stats = self.calculate_daily_stats(date)
            if not stats:
                logger.warning("无法生成报告：缺少统计数据")
                return ""
            
            # 计算关键指标
            initial_balance = 100000  # 初始资金
            final_balance = initial_balance + stats['total_profit']
            total_return = (stats['total_profit'] / initial_balance) * 100
            win_rate = stats['win_rate']
            total_trades = stats['total_trades']
            
            # 计算交易天数
            trade_dates = set(trade['timestamp'].date() for trade in self.trade_history)
            trade_days = len(trade_dates)
            
            # 生成更具个性化的标题
            if total_return > 0:
                if total_return > 5:
                    title = f"💰爆赚+{total_return:.2f}%！今天AI策略超常发挥🔥"
                elif total_return > 2:
                    title = f"💰稳稳收获{total_return:.2f}%，AI策略给力的一天！"
                else:
                    title = f"💰小幅盈利+{total_return:.2f}%，积少成多"
            else:
                if total_return < -3:
                    title = f"💔今天亏损有点大({total_return:.2f}%)，让我们一起复盘看看哪里出了问题"
                elif total_return < 0:
                    title = f"📊微亏{total_return:.2f}%，交易路上的正常波动"
                else:
                    title = f"📊基本持平({total_return:.2f}%)，静待机会"
            
            # 生成正文
            content = f"""
#AI量化交易 #{'盈利' if total_return > 0 else '复盘'} #金融科技

{title}

📈 今日交易总结：
• 初始资金：${initial_balance:,.2f}
• 结束资金：${final_balance:,.2f}
• 总收益率：{total_return:.2f}%
• 胜率：{win_rate:.2f}%
• 总交易次数：{total_trades}
• 交易第{trade_days}天

📌 策略更新说明：
最近我在优化AI模型的特征工程部分，增加了对经济数据发布时间的敏感度判断。
每次更新都会在观摩账户中体现，感谢大家的关注和支持！

📌 XAUUSD（黄金/美元）交易说明：
• 黄金是避险资产，趋势明显且持续性强
• 适合中长线AI量化交易策略
• 受全球经济形势和地缘政治影响较大
• 波动性适中，风险收益比较佳

📌 FTMO挑战账户信息：
账号：1520835905
密码：关注我，私信获取密码
服务器：FTMO-Demo2
更新时间：{datetime.now().strftime('%Y-%m-%d')}

🎯 我正在参与FTMO专业交易员挑战计划（模拟盘阶段），这是迈向专业交易生涯的重要一步！
🔴 实时交易进行中，欢迎随时观摩！

{'如果觉得我的分享对你有帮助，记得点赞关注哦～' if total_return >= 0 else '虽然今天不太理想，但我不会放弃，继续努力优化策略！'}

#量化交易 #AI炒股 #自动化交易 #金融科技 #程序员理财 #外汇交易 #FTMO #黄金交易
            """
            
            logger.info("小红书日报生成成功")
            return content.strip()
            
        except Exception as e:
            logger.error(f"生成日报异常: {str(e)}")
            return ""
    
    def close_connection(self):
        """
        关闭MT5连接
        """
        try:
            mt5.shutdown()
            logger.info("MT5连接已关闭")
        except Exception as e:
            logger.error(f"关闭MT5连接异常: {str(e)}")

def main():
    """
    示例演示如何使用内容生成器
    """
    # 创建生成器实例
    generator = XiaohongshuRealDataGenerator()
    
    try:
        # 连接到MT5
        if not generator.connect_to_mt5():
            logger.error("无法连接到MT5")
            return
        
        # 获取最近7天的交易历史
        generator.trade_history = generator.fetch_trade_history(7)
        
        if not generator.trade_history:
            logger.warning("没有获取到交易历史数据")
            return
        
        # 生成日报
        daily_content = generator.generate_daily_report()
        if daily_content:
            print("=== 小红书日报内容 ===")
            print(daily_content)
        else:
            logger.warning("未能生成日报内容")
    
    finally:
        # 关闭连接
        generator.close_connection()

if __name__ == "__main__":
    main()