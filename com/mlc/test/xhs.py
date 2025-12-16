import pandas as pd
import logging
from datetime import datetime
import json
from datetime import timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XiaohongshuContentGenerator:
    """
    小红书内容生成器，用于自动生成适合在小红书分享的交易相关内容
    """
    
    def __init__(self, backtest_results):
        """
        初始化内容生成器
        
        参数:
            backtest_results (dict): 回测结果
        """
        self.backtest_results = backtest_results
    
    def generate_daily_report(self):
        """
        生成每日交易报告，适合发布到小红书
        
        返回:
            str: 格式化的内容
        """
        try:
            # 计算关键指标
            initial_balance = self.backtest_results['initial_balance']
            final_balance = self.backtest_results['final_balance']
            total_return = self.backtest_results['total_return_pct']
            win_rate = self.backtest_results['win_rate']
            total_trades = self.backtest_results['total_trades']
            
            # 计算交易天数
            trade_days = self._calculate_trade_days()
            
            # 生成标题
            if total_return > 0:
                title = f"💰今日AI自动交易收益+{total_return:.2f}%🔥"
            else:
                title = f"📊今日AI自动交易亏损{total_return:.2f}%，复盘中..."
            
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

✨ 本系统特色：
✅ 全自动AI交易，无需盯盘
✅ 自主进化学习，持续优化
✅ 多因子特征工程，精准预测

📌 策略更新说明：
我持续对AI交易策略进行优化和更新，不断提升系统的稳定性和盈利能力。
每次更新都会在观摩账户中体现，欢迎大家持续关注我的交易表现！

📌 XAUUSD（黄金/美元）交易说明：
• 黄金是避险资产，趋势明显且持续性强
• 适合中长线AI量化交易策略
• 受全球经济形势和地缘政治影响较大
• 波动性适中，风险收益比较佳

📌 FTMO挑战账户信息：
账号：1520835905
密码：关注我，私信获取密码
服务器：FTMO-Demo2
更新时间：2025-12-16

🎯 我正在参与FTMO专业交易员挑战计划（模拟盘阶段），这是迈向专业交易生涯的重要一步！
🔴 实时交易进行中，欢迎随时观摩！

欢迎关注我的模拟交易表现，见证AI自动交易的魅力！

#量化交易 #AI炒股 #自动化交易 #金融科技 #程序员理财 #外汇交易 #FTMO #黄金交易
            """
            
            logger.info("小红书日报生成成功")
            return content.strip()
            
        except Exception as e:
            logger.error(f"生成日报异常: {str(e)}")
            return ""
    
    def _calculate_trade_days(self):
        """
        根据交易历史计算这是交易的第几天
        
        返回:
            int: 交易天数
        """
        try:
            # 从交易历史计算实际交易天数
            if 'trade_history' in self.backtest_results and self.backtest_results['trade_history']:
                trade_history = self.backtest_results['trade_history']
                
                # 收集所有交易发生的日期（去重）
                trade_dates = set()
                for trade in trade_history:
                    timestamp = trade['timestamp']
                    if isinstance(timestamp, str):
                        # 处理字符串格式的时间戳
                        if 'T' in timestamp:
                            timestamp = datetime.fromisoformat(timestamp)
                        else:
                            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    elif hasattr(timestamp, 'to_pydatetime'):
                        # 处理 pandas Timestamp 类型
                        timestamp = timestamp.to_pydatetime()
                    # 只统计开仓交易
                    if trade.get('direction') in ['buy', 'sell']:
                        trade_dates.add(timestamp.date())
                
                # 返回交易天数
                return len(trade_dates) if len(trade_dates) > 0 else 1
            
            # 如果没有交易历史，默认返回1
            return 1
        except Exception as e:
            logger.error(f"计算交易天数异常: {str(e)}")
            return 1

def main():
    """
    示例演示如何使用内容生成器
    """
    # 模拟回测结果，包含真实的交易历史
    sample_results = {
        'initial_balance': 100000,
        'final_balance': 102350,
        'total_return_pct': 2.35,
        'total_trades': 6,
        'profitable_trades': 5,
        'win_rate': 83.33,
        'buy_trades': 3,
        'sell_trades': 3,
        'buy_win_rate': 100.0,
        'sell_win_rate': 66.67,
        'max_balance': 102500,
        'min_balance': 99800,
        'trade_history': [
            {'timestamp': '2025-12-15 10:00:00', 'direction': 'buy'},
            {'timestamp': '2025-12-15 11:00:00', 'direction': 'close'},
            {'timestamp': '2025-12-15 14:00:00', 'direction': 'sell'},
            {'timestamp': '2025-12-15 15:00:00', 'direction': 'close'},
            {'timestamp': '2025-12-16 09:00:00', 'direction': 'buy'},
            {'timestamp': '2025-12-16 10:00:00', 'direction': 'close'},
            {'timestamp': '2025-12-16 11:00:00', 'direction': 'sell'},
            {'timestamp': '2025-12-16 12:00:00', 'direction': 'close'},
            {'timestamp': '2025-12-16 14:00:00', 'direction': 'buy'},
            {'timestamp': '2025-12-16 15:00:00', 'direction': 'close'},
            {'timestamp': '2025-12-16 16:00:00', 'direction': 'sell'},
            {'timestamp': '2025-12-16 17:00:00', 'direction': 'close'}
        ],
        'trade_details': []
    }
    
    generator = XiaohongshuContentGenerator(sample_results)
    
    # 生成日报
    daily_content = generator.generate_daily_report()
    print("=== 日报内容 ===")
    print(daily_content)

if __name__ == "__main__":
    main()