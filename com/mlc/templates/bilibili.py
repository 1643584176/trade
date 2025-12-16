import pandas as pd
import logging
from datetime import datetime
import json
from datetime import timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BilibiliContentGenerator:
    """
    B站内容生成器，用于自动生成适合在B站分享的视频脚本相关内容
    """
    
    def __init__(self, backtest_results):
        """
        初始化内容生成器
        
        参数:
            backtest_results (dict): 回测结果
        """
        self.backtest_results = backtest_results
    
    def generate_video_script(self):
        """
        生成视频脚本文案，适合发布到B站
        
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
                title = f"【AI量化】单日收益{total_return:.2f}%！我的自动化交易系统又赚钱啦！"
            else:
                title = f"【AI量化复盘】今天亏损{abs(total_return):.2f}%，来看看问题出在哪了"
            
            # 生成视频脚本
            script = f"""
{title}

哈喽大家好，我是你们的AI量化交易程序员！

今天又来给大家汇报一下我的全自动AI交易系统的战绩啦～

## 开场白

{'今天是个好日子！我的AI交易系统又帮我省心地赚了一波！来看看具体表现吧～' if total_return > 0 else '虽然今天亏了一些钱，但我觉得这个复盘过程更有价值，一起来看看是哪里出了问题。'}

## 数据展示环节

画面切换到交易图表和数据面板：

- 初始资金：${initial_balance:,.2f}
- 结束资金：${final_balance:,.2f}
- 今日收益率：{total_return:.2f}%
- 交易胜率：{win_rate:.2f}%
- 今日交易次数：{total_trades}
- 累计交易天数：第{trade_days}天

## 策略解析时间

{'今天的市场走势相对平稳，我的AI系统成功抓住了两次明显的趋势机会，一次是早上10点的多单，另一次是下午2点的空单，都获得了不错的收益。' if total_return > 0 else '今天的市场波动剧烈，我的系统虽然方向判断正确，但在大幅震荡中触发了止损，这也是我们需要接受的风险。'}

画面显示具体的交易记录和K线图：

1. 第一笔交易：买入信号出现
   时间：上午10:00
   价格：XXX
   方向：做多
   
2. 第二笔交易：卖出信号出现
   时间：下午14:00
   价格：XXX
   方向：做空

## 技术要点讲解

接下来给大家简单讲讲我的AI系统是如何工作的：

首先，它会收集大量的市场数据，包括：
- 历史价格数据
- 技术指标
- 市场情绪指标

然后通过机器学习模型进行分析和预测，最后自动执行交易决策。

整个过程不需要我手动干预，真正实现了躺赚～

## 风控机制介绍

当然，为了保护资金安全，我也设置了严格的风控机制：
- 单笔最大损失限制：1手600美金，5手300美金
- 最大持仓限制：同一时间只能持有一个品种的仓位
- 自动止损：一旦达到预设亏损额度立即平仓

## 互动环节

{'今天的战绩还算不错，大家有什么想了解的可以留言告诉我哦～' if total_return >= 0 else '虽然今天不太理想，但失败也是成功之母嘛，大家觉得我的策略还有哪些可以改进的地方？欢迎在评论区讨论！'}

## 结尾彩蛋

最后提醒大家一句：
这个是我的AI自动化交易系统模拟盘交易结果，不是实盘交易，大家不要盲目跟随哈！

如果觉得这个视频对你有帮助的话，记得一键三连支持一下，我们下期再见！

#AI量化交易 #程序员理财 #自动化交易 #外汇交易 #黄金交易 #量化投资
"""
            
            logger.info("B站视频脚本生成成功")
            return script.strip()
            
        except Exception as e:
            logger.error(f"生成B站视频脚本异常: {str(e)}")
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
    
    generator = BilibiliContentGenerator(sample_results)
    
    # 生成视频脚本
    video_script = generator.generate_video_script()
    print("=== B站视频脚本内容 ===")
    print(video_script)

if __name__ == "__main__":
    main()