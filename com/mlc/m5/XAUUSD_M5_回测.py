import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sys
import os
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

# 添加公共模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))

from m5_feature_engineering import M5FeatureEngineer

# 配置参数
class BacktestConfig:
    SYMBOL = "XAUUSD"
    M5_TIMEFRAME = mt5.TIMEFRAME_M5
    M1_TIMEFRAME = mt5.TIMEFRAME_M1
    MODEL_PATH = "xauusd_m5_model.json"  # 模型文件路径
    SCALER_PATH = "m5_scaler.pkl"  # 标准化器路径
    HISTORY_M5_BARS = 120  # 模型输入序列长度
    PREDICT_FUTURE_BARS = 3  # 模型预测长度

    # 交易规则
    LOT_SIZE = 0.2  # 固定手数
    STOP_LOSS_PIPS = 600  # 止损600点位 = 120美金
    TAKE_PROFIT_PIPS = 1000  # 止盈1000点位 = 200美金

    # FTMO规则
    FTMO_MAX_DRAWDOWN = 0.045  # 最大回撤4.5%
    FTMO_PROFIT_TARGET = 0.10  # 盈利目标10%
    FTMO_MIN_BALANCE = 99020  # 最低余额

    # 交易时段（UTC）
    PEAK_HOUR_START = 13
    PEAK_HOUR_END = 20
    UTC_TZ = timezone.utc

class BacktestEngine:
    def __init__(self):
        # 加载模型
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(BacktestConfig.MODEL_PATH)
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.model = None

        # 初始化回测参数
        self.initial_balance = 100000.0  # 10万美元初始资金
        self.current_balance = self.initial_balance
        self.current_equity = self.initial_balance
        self.trade_history = []
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance

        # 特征工程实例
        self.feature_engineer = M5FeatureEngineer()

    def get_backtest_data(self, days: int = 30):
        """获取回测用的MT5历史数据"""
        if not mt5.initialize():
            raise Exception(f"MT5初始化失败：{mt5.last_error()}")

        # 获取市场时间
        tick = mt5.symbol_info_tick(BacktestConfig.SYMBOL)
        if tick is not None:
            current_utc = datetime.fromtimestamp(tick.time, tz=BacktestConfig.UTC_TZ)
        else:
            current_utc = datetime.now(BacktestConfig.UTC_TZ)

        # 计算数据范围
        total_minutes = days * 24 * 60
        start_time = current_utc - timedelta(minutes=total_minutes)

        # 获取M5数据
        m5_raw = mt5.copy_rates_range(
            BacktestConfig.SYMBOL,
            BacktestConfig.M5_TIMEFRAME,
            start_time.timestamp(),
            current_utc.timestamp()
        )
        if m5_raw is None or len(m5_raw) == 0:
            raise Exception(f"获取M5数据失败：{mt5.last_error()}")

        # 构建DataFrame
        df = pd.DataFrame(m5_raw)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)

        # 添加特征
        df = self.feature_engineer.add_core_features(df)
        df = self.feature_engineer.add_enhanced_features(df)

        # 清理空值
        df = df.dropna()
        print(f"成功获取 {len(df)} 根M5 K线数据")

        return df

    def calculate_signals(self, df):
        """计算交易信号"""
        # 定义特征列
        feature_columns = [
            # M5周期特征（主要决策）
            'open', 'high', 'low', 'close', 'tick_volume',
            'price_position', 'atr_14', 'volatility_pct', 'hl_ratio',
            'm15_trend', 'm30_support', 'm30_resistance',
            'spread_change', 'volatility_change', 'tick_density',
            'hour_of_day', 'is_peak_hour',
            # K线形态特征
            'hammer', 'shooting_star', 'engulfing',
            # 技术指标
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bollinger_upper', 'bollinger_lower', 'bollinger_position',
            'ma5', 'ma10', 'ma20', 'ma5_direction', 'ma10_direction', 'ma20_direction',
            # 一致性特征
            'ma_direction_consistency', 'rsi_price_consistency',
            # 跨周期特征
            'rsi_divergence', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long',
            'trend_consistency',
            # 信号特征
            'rsi_signal_strength', 'macd_signal_strength', 'short_long_signal_consistency',
            # 风险特征
            'volatility_regime', 'vol_cluster'
        ]
        
        # 检查所有特征列是否存在
        available_features = []
        for col in feature_columns:
            if col in df.columns:
                available_features.append(col)

        signals = []
        probabilities = []
        
        for i in range(BacktestConfig.HISTORY_M5_BARS, len(df)):
            # 获取特征数据
            row_data = df.iloc[i][available_features].values.reshape(1, -1)
            
            try:
                # 预测概率
                pred_proba = self.model.predict_proba(row_data)[0]
                # 对于多分类，取上涨概率（假设类别0是下跌，1是平，2是上涨）
                # 这里需要根据实际模型的类别顺序调整
                if len(pred_proba) == 3:  # 假设有三个类别：-1, 0, 1
                    # 通常XGBoost会按数值顺序排列，即[-1, 0, 1] -> [0, 1, 2]
                    up_prob = pred_proba[2]  # 上涨类别
                    down_prob = pred_proba[0]  # 下跌类别
                else:
                    # 如果是二分类，直接使用
                    up_prob = pred_proba[1]
                    down_prob = pred_proba[0]
                
                # 生成信号
                if up_prob > 0.7:
                    signal = "BUY"
                elif down_prob > 0.7:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                signals.append(signal)
                probabilities.append(max(up_prob, down_prob))  # 使用最大概率
                
            except Exception as e:
                signals.append("HOLD")
                probabilities.append(0.0)
        
        # 将信号添加到DataFrame
        df_signals = df.iloc[BacktestConfig.HISTORY_M5_BARS:].copy()
        df_signals['signal'] = signals
        df_signals['probability'] = probabilities
        
        return df_signals

    def execute_backtest(self, df_with_signals):
        """执行回测"""
        print("开始执行回测...")
        
        position = None
        entry_price = 0
        entry_time = None
        
        for idx, row in df_with_signals.iterrows():
            current_time = idx
            current_price = row['close']
            signal = row['signal']
            prob = row['probability']
            
            # 检查现有持仓
            if position is not None:
                # 计算当前盈亏
                if position == "BUY":
                    current_pnl = (current_price - entry_price) * 100 * BacktestConfig.LOT_SIZE
                else:  # SELL
                    current_pnl = (entry_price - current_price) * 100 * BacktestConfig.LOT_SIZE
                
                # 止损止盈检查
                should_close = False
                close_reason = ""
                
                # 止损条件 (120美金)
                if abs(current_pnl) >= 120 and current_pnl < 0:
                    should_close = True
                    close_reason = "STOP_LOSS"
                
                # 止盈条件 (200美金)
                elif current_pnl >= 200:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
                
                # 观望信号且盈利超过90美金
                elif signal == "HOLD" and current_pnl > 90:
                    should_close = True
                    close_reason = "PROFIT_TAKE_SIGNAL"
                
                # 每日收盘前平仓
                if current_time.hour >= 20 and position is not None:
                    should_close = True
                    close_reason = "DAY_END"
                
                if should_close:
                    # 平仓
                    self.current_balance += current_pnl
                    self.current_equity = self.current_balance
                    
                    # 记录交易
                    self.trade_history.append({
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "direction": position,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl": current_pnl,
                        "balance_after": self.current_balance,
                        "close_reason": close_reason
                    })
                    
                    print(f"平仓: {position} | 入场: {entry_price:.5f} | 出场: {current_price:.5f} | 盈亏: {current_pnl:.2f} | 原因: {close_reason}")
                    
                    position = None
            
            # 开仓条件
            if position is None and signal in ["BUY", "SELL"]:
                # 检查是否是信号反向（如果之前有持仓方向）
                position = signal
                entry_price = current_price
                entry_time = current_time
                
                print(f"开仓: {signal} | 价格: {current_price:.5f} | 概率: {prob:.3f}")
        
        # 回测结束，如果有持仓则平仓
        if position is not None:
            final_pnl = 0
            if position == "BUY":
                final_pnl = (current_price - entry_price) * 100 * BacktestConfig.LOT_SIZE
            else:
                final_pnl = (entry_price - current_price) * 100 * BacktestConfig.LOT_SIZE
            
            self.current_balance += final_pnl
            self.current_equity = self.current_balance
            
            self.trade_history.append({
                "entry_time": entry_time,
                "exit_time": df_with_signals.index[-1],
                "direction": position,
                "entry_price": entry_price,
                "exit_price": current_price,
                "pnl": final_pnl,
                "balance_after": self.current_balance,
                "close_reason": "BACKTEST_END"
            })
            
            print(f"回测结束强制平仓: {position} | 盈亏: {final_pnl:.2f}")

    def generate_report(self):
        """生成回测报告"""
        print("\n========== 回测报告 ==========")
        print(f"初始账户余额: ¥{self.initial_balance:,.2f}")
        print(f"最终账户余额: ¥{self.current_balance:,.2f}")
        print(f"总盈亏: ¥{self.current_balance - self.initial_balance:,.2f}")
        print(f"总收益率: {(self.current_balance / self.initial_balance - 1) * 100:.2f}%")
        print(f"最大回撤: {self.max_drawdown * 100:.2f}%")
        print(f"总交易次数: {len(self.trade_history)}")

        if len(self.trade_history) == 0:
            return

        # 交易统计
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] < 0]
        breakeven_trades = [t for t in self.trade_history if t['pnl'] == 0]

        total_win = len(winning_trades)
        total_loss = len(losing_trades)
        total_breakeven = len(breakeven_trades)

        win_amount = sum(t['pnl'] for t in winning_trades)
        loss_amount = sum(t['pnl'] for t in losing_trades)

        print(f"\n交易统计:")
        print(f"  盈利交易: {total_win} 笔 (胜率: {total_win / len(self.trade_history) * 100:.2f}%)")
        print(f"  亏损交易: {total_loss} 笔")
        print(f"  平手交易: {total_breakeven} 笔")
        print(f"  总盈利金额: ¥{win_amount:.2f}")
        print(f"  总亏损金额: ¥{loss_amount:.2f}")
        print(f"  平均盈利: ¥{win_amount / total_win:.2f}" if total_win > 0 else "  平均盈利: ¥0.00")
        print(f"  平均亏损: ¥{loss_amount / total_loss:.2f}" if total_loss > 0 else "  平均亏损: ¥0.00")

        # 盈亏比
        if total_loss > 0 and total_win > 0:
            avg_win = win_amount / total_win
            avg_loss = abs(loss_amount / total_loss)
            profit_factor = avg_win / avg_loss
            print(f"  盈亏比: {profit_factor:.2f}:1")

        # FTMO合规性
        print(f"\nFTMO合规性校验:")
        profit_target_met = (self.current_balance / self.initial_balance - 1) >= BacktestConfig.FTMO_PROFIT_TARGET
        drawdown_compliant = self.max_drawdown <= BacktestConfig.FTMO_MAX_DRAWDOWN
        min_balance_met = self.current_balance >= BacktestConfig.FTMO_MIN_BALANCE

        print(f"  盈利目标达标(≥10%): {'是' if profit_target_met else '否'}")
        print(f"  最大回撤合规(≤4.5%): {'是' if drawdown_compliant else '否'}")
        print(f"  最低余额合规(≥99020): {'是' if min_balance_met else '否'}")

        # 前5笔交易详情
        print(f"\n前5笔交易详情:")
        for i, trade in enumerate(self.trade_history[:5]):
            print(f"\n  交易 {i + 1}:")
            print(f"    方向: {trade['direction']} | 入场时间: {trade['entry_time']}")
            print(f"    入场价: {trade['entry_price']:.5f} | 出场价: {trade['exit_price']:.5f}")
            print(f"    盈亏: ¥{trade['pnl']:.2f} | 原因: {trade['close_reason']}")

    def run_backtest(self, days: int = 7):
        """运行完整回测"""
        print(f"\n========== 开始回测最近 {days} 天数据 ==========")
        try:
            df = self.get_backtest_data(days=days)
            df_with_signals = self.calculate_signals(df)
            self.execute_backtest(df_with_signals)
            self.generate_report()
        except Exception as e:
            print(f"回测执行失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("开始XAUUSD M5周期回测")
    try:
        engine = BacktestEngine()
        engine.run_backtest(days=7)  # 回测最近7天数据
    except Exception as e:
        print(f"回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭MT5连接
        mt5.shutdown()
        print("\n回测完成，MT5连接已关闭")

if __name__ == "__main__":
    main()