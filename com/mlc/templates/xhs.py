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
            
            # 增加个性化内容
            market_comment = self._generate_market_comment(total_return)
            reflection = self._generate_reflection(total_return, win_rate)
            
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

{market_comment}

{reflection}

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
更新时间：2025-12-16

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
    
    def _generate_market_comment(self, total_return):
        """
        根据收益率生成市场评论
        """
        if total_return > 0:
            return "📈 市场点评：今天黄金价格走势相对稳定，给了AI策略很好的发挥空间。系统成功捕捉到了几次明显的趋势机会，整体表现符合预期。"
        else:
            return "📉 市场点评：今天市场波动较大，特别是在下午时段出现了几次快速反转，这对策略的稳定性提出了更高要求。"
    
    def _generate_reflection(self, total_return, win_rate):
        """
        根据交易结果生成反思内容
        """
        if total_return > 0:
            if win_rate >= 80:
                return "💡 今日反思：高胜率表明策略在当前市场环境下适应性良好，继续保持现有参数配置。同时也在思考是否可以适当增加仓位来提升收益。"
            else:
                return "💡 今日反思：虽然总体收益为正，但胜率有待提高。下一步需要优化入场时机判断，减少无效交易。"
        else:
            if total_return > -2:
                return "💡 今日反思：小幅度亏损是交易的一部分，重要的是找到问题所在。经过分析，主要问题出现在对突发消息的应对不足。"
            else:
                return "💡 今日反思：较大的回撤提醒我们需要审视策略的有效性。准备对模型参数进行重新校准，并增加对异常波动的检测机制。"
    
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