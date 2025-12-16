import pandas as pd
import logging
from datetime import datetime
import json
from datetime import timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZhihuContentGenerator:
    """
    知乎内容生成器，用于自动生成适合在知乎分享的技术分析相关内容
    """
    
    def __init__(self, backtest_results):
        """
        初始化内容生成器
        
        参数:
            backtest_results (dict): 回测结果
        """
        self.backtest_results = backtest_results
    
    def generate_technical_article(self):
        """
        生成技术分析文章，适合发布到知乎
        
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
                title = f"AI量化交易策略实战：单日收益率{total_return:.2f}%的背后逻辑"
            else:
                title = f"AI量化交易策略复盘：今日收益率{total_return:.2f}%的教训与思考"
            
            # 生成正文
            content = f"""
{title}

大家好，我是专注于AI量化交易的技术人员。今天想和大家分享一下我基于机器学习算法构建的XAUUSD（黄金/美元）交易策略在今日的表现和背后的技术逻辑。

## 策略表现概览

- 初始资金：${initial_balance:,.2f}
- 结束资金：${final_balance:,.2f}
- 总收益率：{total_return:.2f}%
- 胜率：{win_rate:.2f}%
- 总交易次数：{total_trades}
- 累计交易天数：第{trade_days}天

## 技术架构解析

我的AI交易系统主要由以下几个核心模块构成：

1. **数据预处理模块**：清洗和标准化来自MT5平台的原始行情数据
2. **特征工程模块**：提取价格、成交量、技术指标等多维度特征
3. **模型预测模块**：采用集成学习算法预测未来价格走势
4. **风险管理模块**：严格执行仓位控制和止损机制
5. **执行模块**：自动下单和平仓，实现全自动化交易

## 今日策略亮点

{'今日策略表现出色，主要原因在于模型成功捕捉到了亚洲盘时段的黄金价格波动特征，并在欧洲盘开盘前及时建立了多头仓位。' if total_return > 0 else '今天的市场波动超出预期，尽管模型在早盘做出了正确判断，但在下午的剧烈震荡中触发了止损机制。'}

## 风险控制机制

为了确保策略稳健运行，我设置了以下风控措施：
- 单笔最大损失限制：1手600美金，5手300美金
- 单一品种最大持仓限制：1手
- 动态仓位调整：根据市场波动性自动调整交易手数

## 未来优化方向

{'基于今天的优异表现，我计划进一步优化特征工程部分，特别是加强对宏观经济数据发布期间的市场反应建模。' if total_return > 0 else '今天的回撤提醒我们需要更好地处理异常波动情况，下一步将加强模型对黑天鹅事件的应对能力。'}

## 结语

量化交易是一个不断迭代优化的过程，每一次交易都是对策略有效性的检验。我会持续分享我的实践经验和研究成果，希望能与更多对量化交易感兴趣的朋友交流探讨。

{'如果你对AI量化交易感兴趣，欢迎关注我的专栏，我会定期分享更多技术干货！' if total_return >= 0 else '量化之路充满挑战，但正是这些挑战让我们不断进步。期待明天有更好的表现！'}

---
注：以上为AI自动化交易系统模拟盘交易结果，非实盘交易，请勿盲目跟随。
"""
            
            logger.info("知乎技术文章生成成功")
            return content.strip()
            
        except Exception as e:
            logger.error(f"生成知乎文章异常: {str(e)}")
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
    
    generator = ZhihuContentGenerator(sample_results)
    
    # 生成日报
    article_content = generator.generate_technical_article()
    print("=== 知乎文章内容 ===")
    print(article_content)

if __name__ == "__main__":
    main()