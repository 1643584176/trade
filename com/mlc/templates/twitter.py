import pandas as pd
import logging
from datetime import datetime
import json
from datetime import timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwitterContentGenerator:
    """
    Twitter/X内容生成器，用于自动生成适合在Twitter分享的短内容
    """
    
    def __init__(self, backtest_results):
        """
        初始化内容生成器
        
        参数:
            backtest_results (dict): 回测结果
        """
        self.backtest_results = backtest_results
    
    def generate_tweet(self):
        """
        生成推文内容，适合发布到Twitter
        
        返回:
            list: 多条推文内容列表（用于推文串）
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
            
            tweets = []
            
            # 第一条推文 - 主要结果
            if total_return > 0:
                tweet1 = f"💰 ${total_return:.2f}% profit today with my AI trading bot! \n\nAccount balance: ${final_balance:,.0f} (was ${initial_balance:,.0f})\nWin rate: {win_rate:.1f}%\nTrades executed: {total_trades}\nTrading day: #{trade_days}"
            else:
                tweet1 = f"📊 ${total_return:.2f}% loss today, but every loss is a lesson. \n\nAccount balance: ${final_balance:,.0f} (was ${initial_balance:,.0f})\nWin rate: {win_rate:.1f}%\nTrades executed: {total_trades}\nTrading day: #{trade_days}"
            
            tweets.append(tweet1)
            
            # 第二条推文 - 策略详情
            if total_return > 0:
                tweet2 = f"✅ Today's winning strategy:\n- XAUUSD (Gold/USD) automated trading\n- Fixed position sizing (1 lot)\n- Hard stop-loss rules ($600 max loss)\n- Fully autonomous AI decisions\n\n#AlgoTrading #QuantFinance #AIInvesting"
            else:
                tweet2 = f"⚠️ Today's challenge:\n- Market volatility exceeded predictions\n- Stop-loss triggered on 1 position\n- Model adjustment needed for black swan events\n\nStill committed to improving! #AlgoTrading #QuantFinance"
            
            tweets.append(tweet2)
            
            # 第三条推文 - 项目状态
            tweet3 = f"🚀 My journey to become a professional algo trader:\n\n✅ Completed FTMO Challenge Phase 1\n⏳ In progress: FTMO Challenge Phase 2\n🎯 Goal: Consistent profitability\n\nFollow for daily updates!\n\n#TradingChallenge #RetailTrader #Fintech"
            
            tweets.append(tweet3)
            
            logger.info("Twitter推文生成成功")
            return tweets
            
        except Exception as e:
            logger.error(f"生成Twitter推文异常: {str(e)}")
            return [""]
    
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
    
    generator = TwitterContentGenerator(sample_results)
    
    # 生成推文
    tweets = generator.generate_tweet()
    print("=== Twitter推文内容 ===")
    for i, tweet in enumerate(tweets, 1):
        print(f"推文 {i}:")
        print(tweet)
        print("-" * 50)

if __name__ == "__main__":
    main()