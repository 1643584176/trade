import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sys
import os
import time
from datetime import datetime, timedelta, timezone
import logging
from threading import Thread, Event
import warnings
warnings.filterwarnings('ignore')

# 添加公共模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))

from m5_feature_engineering import M5FeatureEngineer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xauusd_m5_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class M5RealTimeTrader:
    def __init__(self, model_path="xauusd_m5_model.json", lot_size=0.2):
        """
        初始化M5实时交易器
        
        参数:
            model_path (str): 模型路径
            lot_size (float): 交易手数
        """
        # 配置参数
        self.SYMBOL = "XAUUSD"
        self.M5_TIMEFRAME = mt5.TIMEFRAME_M5
        self.MODEL_PATH = model_path
        self.LOT_SIZE = lot_size
        self.STOP_LOSS_PIPS = 600  # 止损600点位 = 120美金
        self.TAKE_PROFIT_PIPS = 1000  # 止盈1000点位 = 200美金
        self.MAGIC_NUMBER = 10000005  # M5周期魔法数字
        self.HISTORY_M5_BARS = 120  # 用于预测的K线数量
        
        # FTMO规则
        self.FTMO_MAX_DRAWDOWN = 0.045  # 最大回撤4.5%
        self.FTMO_PROFIT_TARGET = 0.10  # 盈利目标10%
        self.FTMO_MIN_BALANCE = 99020  # 最低余额
        
        # 交易状态
        self.current_position = None
        self.is_running = False
        self.stop_event = Event()
        
        # 特征工程实例
        self.feature_engineer = M5FeatureEngineer()
        
        # 初始化MT5连接
        if not mt5.initialize():
            raise Exception(f"MT5初始化失败：{mt5.last_error()}")
        
        # 确保交易品种被选中
        if not mt5.symbol_select(self.SYMBOL, True):
            raise Exception(f"无法选择交易品种 {self.SYMBOL}")
        
        # 加载模型
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.MODEL_PATH)
            logger.info(f"模型已从 {self.MODEL_PATH} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise e
        
        # 检查现有持仓
        self.check_existing_positions()
        
        logger.info(f"MT5连接成功")
        logger.info(f"开始基于M5周期的实时交易 {self.SYMBOL}，手数: {self.LOT_SIZE}")
        logger.info(f"策略: 信号反向平仓、观望信号且盈利超90美金平仓、每日收盘前平仓")

    def check_existing_positions(self):
        """检查现有持仓"""
        try:
            positions = mt5.positions_get(symbol=self.SYMBOL)
            if positions:
                pos = positions[0]  # 假设只处理一个持仓
                direction = "做多" if pos.type == mt5.POSITION_TYPE_BUY else "做空"
                self.current_position = {
                    'ticket': pos.ticket,
                    'type': pos.type,
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'time': pos.time,
                    'direction': direction
                }
                logger.info(f"检测到现有持仓: {direction}, 手数: {pos.volume}")
            else:
                logger.info("未检测到现有持仓")
        except Exception as e:
            logger.error(f"检查现有持仓失败: {e}")

    def get_current_market_data(self, bars_count: int = 200):
        """获取当前市场数据"""
        try:
            # 获取当前时间
            current_tick = mt5.symbol_info_tick(self.SYMBOL)
            if current_tick is None:
                logger.error("无法获取当前市场数据")
                return None
            
            # 计算开始时间（获取最近的K线数据）
            current_time = datetime.fromtimestamp(current_tick.time)
            start_time = current_time - timedelta(minutes=5*bars_count)
            
            # 获取M5历史数据
            rates = mt5.copy_rates_range(
                self.SYMBOL,
                self.M5_TIMEFRAME,
                int(start_time.timestamp()),
                int(current_time.timestamp())
            )
            
            if rates is None or len(rates) == 0:
                logger.error(f"获取M5历史数据失败: {mt5.last_error()}")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            
            # 添加特征
            df = self.feature_engineer.add_core_features(df)
            df = self.feature_engineer.add_enhanced_features(df)
            
            # 清理数据
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return None

    def calculate_signal(self, df):
        """计算交易信号"""
        try:
            if len(df) < self.HISTORY_M5_BARS:
                logger.warning(f"数据不足，需要{self.HISTORY_M5_BARS}根K线，当前{len(df)}根")
                return "HOLD", 0.0
            
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
            
            # 获取最新的特征数据
            latest_data = df.iloc[-1][available_features].values.reshape(1, -1)
            
            # 预测概率
            pred_proba = self.model.predict_proba(latest_data)[0]
            
            # 获取上涨和下跌概率
            # 假设类别顺序为[-1, 0, 1] -> [下跌, 观望, 上涨]
            if len(pred_proba) == 3:
                down_prob = pred_proba[0]  # 下跌概率
                hold_prob = pred_proba[1]  # 观望概率
                up_prob = pred_proba[2]    # 上涨概率
            else:
                # 如果是二分类，需要根据实际情况调整
                down_prob = pred_proba[0]
                up_prob = pred_proba[1]
                hold_prob = 1 - up_prob - down_prob  # 中间概率
            
            # 生成信号
            if up_prob > 0.7:
                signal = "BUY"
                confidence = up_prob
                reason = f"上涨概率 {up_prob:.4f} 超过阈值0.7，市场可能上涨"
            elif down_prob > 0.7:
                signal = "SELL"
                confidence = down_prob
                reason = f"下跌概率 {down_prob:.4f} 超过阈值0.7，市场可能下跌"
            else:
                signal = "HOLD"
                confidence = max(up_prob, down_prob)
                reason = f"无明确方向，上涨概率 {up_prob:.4f}，下跌概率 {down_prob:.4f}，均未超过阈值0.7"
            
            # 返回信号和最高概率
            logger.info(f"预测概率 - 上涨: {up_prob:.4f}, 下跌: {down_prob:.4f}, 观望: {hold_prob:.4f}")
            logger.info(f"交易信号: {signal} (置信度: {confidence:.4f})")
            logger.info(f"决策依据: {reason}")
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"计算信号失败: {e}")
            return "HOLD", 0.0

    def place_order(self, signal):
        """下单"""
        try:
            # 获取当前价格
            tick = mt5.symbol_info_tick(self.SYMBOL)
            if tick is None:
                logger.error("无法获取当前价格")
                return False
            
            # 确定订单类型
            if signal == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = price - 6  # 6美金止损，对应120美金(600点位)对0.2手的计算
                tp = price + 10  # 10美金止盈，对应200美金(1000点位)
            elif signal == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = price + 6  # 6美金止损
                tp = price - 10  # 10美金止盈
            else:
                logger.warning("无效的交易信号")
                return False
            
            # 准备订单请求
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.SYMBOL,
                "volume": self.LOT_SIZE,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": self.MAGIC_NUMBER,
                "comment": f"M5信号交易_{signal}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # 执行订单
            result = mt5.order_send(request)
            if result is None:
                logger.error("订单发送失败")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"订单执行失败: {result.retcode} - {result.comment}")
                return False
            
            # 更新持仓信息
            self.current_position = {
                'ticket': result.order,
                'type': order_type,
                'volume': self.LOT_SIZE,
                'price_open': price,
                'time': datetime.now(),
                'direction': "做多" if order_type == mt5.ORDER_TYPE_BUY else "做空"
            }
            
            logger.info(f"开仓成功: {signal} | 手数: {self.LOT_SIZE} | 订单号: {result.order}")
            return True
            
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return False

    def close_position(self, reason=""):
        """平仓"""
        if self.current_position is None:
            logger.info("当前无持仓")
            return True
        
        try:
            # 获取持仓信息
            ticket = self.current_position['ticket']
            pos_type = self.current_position['type']
            
            # 获取当前价格
            tick = mt5.symbol_info_tick(self.SYMBOL)
            if tick is None:
                logger.error("无法获取当前价格")
                return False
            
            # 确定平仓价格和类型
            if pos_type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            # 准备平仓订单请求
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.SYMBOL,
                "volume": self.LOT_SIZE,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": self.MAGIC_NUMBER,
                "comment": f"M5平仓_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # 执行平仓订单
            result = mt5.order_send(request)
            if result is None:
                logger.error("平仓订单发送失败")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"平仓订单执行失败: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"平仓成功: {reason} | 订单号: {ticket}")
            
            # 清除持仓信息
            self.current_position = None
            return True
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return False

    def check_and_close_by_signal(self, current_signal):
        """根据信号检查是否需要平仓"""
        if self.current_position is None:
            return False
        
        current_direction = self.current_position['direction']
        
        # 信号反向时平仓
        if (current_direction == "做多" and current_signal == "SELL") or \
           (current_direction == "做空" and current_signal == "BUY"):
            logger.info(f"平仓: 信号反向出现")
            return self.close_position("信号反向")
        
        # 检查持仓盈利
        try:
            # 获取持仓详情
            positions = mt5.positions_get(symbol=self.SYMBOL)
            if positions and len(positions) > 0:
                pos = positions[0]
                profit = pos.profit
                
                # 观望信号且盈利超过90美金时平仓
                if current_signal == "HOLD" and profit > 90:
                    logger.info(f"平仓: 观望信号且盈利超过90美金 ({profit:.2f}美金)")
                    return self.close_position("观望信号盈利")
        except Exception as e:
            logger.error(f"检查持仓盈利失败: {e}")
        
        return False

    def check_daily_close(self):
        """检查是否需要每日收盘前平仓"""
        if self.current_position is None:
            return False
        
        # 获取当前XAUUSD市场时间
        tick = mt5.symbol_info_tick(self.SYMBOL)
        if tick is None:
            return False
        
        current_time = datetime.fromtimestamp(tick.time)
        
        # 每日20:00 UTC平仓（XAUUSD市场时间）
        if current_time.hour >= 20 and current_time.minute >= 0:
            logger.info("平仓: 每日收盘前平仓")
            return self.close_position("每日收盘")
        
        return False

    def check_risk_management(self):
        """检查风控管理"""
        try:
            # 获取账户信息
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("无法获取账户信息")
                return False
            
            balance = account_info.balance
            equity = account_info.equity
            
            # 计算当日盈亏（简化版，实际需要记录当日初始余额）
            daily_pnl = equity - balance  # 这里是简化处理
            
            # 检查是否超过最大回撤限制
            if daily_pnl < -abs(balance * self.FTMO_MAX_DRAWDOWN):
                logger.warning(f"超过最大回撤限制: 当日亏损 {daily_pnl:.2f}, 限制: {balance * self.FTMO_MAX_DRAWDOWN:.2f}")
                if self.current_position is not None:
                    logger.info("执行风控平仓")
                    return self.close_position("风控平仓")
            
            # 检查账户余额是否低于最低要求
            if balance < self.FTMO_MIN_BALANCE:
                logger.warning(f"账户余额低于最低要求: {balance} < {self.FTMO_MIN_BALANCE}")
                if self.current_position is not None:
                    logger.info("执行风控平仓")
                    return self.close_position("余额不足")
            
            # 检查是否达到盈利目标
            initial_balance = 100000  # 假设初始余额
            if balance >= initial_balance * (1 + self.FTMO_PROFIT_TARGET):
                logger.info(f"达到盈利目标: {balance} >= {initial_balance * (1 + self.FTMO_PROFIT_TARGET)}")
                if self.current_position is not None:
                    logger.info("执行盈利目标平仓")
                    return self.close_position("盈利目标")
            
        except Exception as e:
            logger.error(f"风控检查失败: {e}")
        
        return False

    def incremental_training(self, new_data=None):
        """增量训练模型"""
        try:
            logger.info("开始增量训练...")
            
            # 获取最新的市场数据用于训练
            if new_data is None:
                new_data = self.get_current_market_data(bars_count=500)
                if new_data is None:
                    logger.error("获取新数据失败，跳过增量训练")
                    return False
            
            # 准备特征和目标变量
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
                'ma_direction_consistency', 'rsi_price_consistency'
            ]
            
            available_features = []
            for col in feature_columns:
                if col in new_data.columns:
                    available_features.append(col)
            
            # 创建目标变量：预测未来1根K线的涨跌
            new_data['future_close'] = new_data['close'].shift(-1)
            new_data['price_change_pct'] = (new_data['future_close'] - new_data['close']) / new_data['close']
            new_data['target'] = np.where(new_data['price_change_pct'] > 0.001, 1,  # 涨
                                        np.where(new_data['price_change_pct'] < -0.001, -1, 0))  # 跌和平
            
            # 准备训练数据
            X_new = new_data[available_features].values
            y_new = new_data['target'].values
            
            # 过滤掉NaN值
            mask = ~(np.isnan(X_new).any(axis=1) | np.isnan(y_new))
            X_new = X_new[mask]
            y_new = y_new[mask]
            
            if len(X_new) == 0:
                logger.warning("没有有效的训练数据，跳过增量训练")
                return False
            
            # 获取最新的一部分数据进行增量训练
            n_samples = min(200, len(X_new))  # 最多使用200个样本
            X_recent = X_new[-n_samples:]
            y_recent = y_new[-n_samples:]
            
            # 使用现有模型进行增量训练（warm start）
            # 注意：XGBoost原生不支持增量训练，这里使用更新模型的方法
            # 实际应用中可能需要使用其他支持增量学习的算法
            logger.info(f"使用 {len(X_recent)} 个新样本进行增量训练")
            
            # 评估旧模型性能
            try:
                old_score = self.model.score(X_recent, y_recent)
                logger.info(f"旧模型在新数据上的准确率: {old_score:.4f}")
            except:
                logger.warning("无法评估旧模型性能")
            
            # 由于XGBoost不直接支持增量训练，我们使用部分拟合的方法
            # 这里简化处理，实际应用中可能需要更复杂的策略
            self.model.fit(X_recent, y_recent, 
                          xgb_model=self.MODEL_PATH if os.path.exists(self.MODEL_PATH) else None)
            
            # 评估新模型性能
            try:
                new_score = self.model.score(X_recent, y_recent)
                logger.info(f"新模型在新数据上的准确率: {new_score:.4f}")
                
                # 如果性能提升，则保存新模型
                if new_score > old_score:
                    new_model_path = f"xauusd_m5_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.model.save_model(new_model_path)
                    logger.info(f"新模型已保存到: {new_model_path}")
                    self.MODEL_PATH = new_model_path
            except:
                logger.warning("无法评估新模型性能")
            
            return True
            
        except Exception as e:
            logger.error(f"增量训练失败: {e}")
            return False

    def run_trading_cycle(self):
        """执行单次交易循环"""
        try:
            # 获取市场数据
            df = self.get_current_market_data()
            if df is None:
                logger.error("获取市场数据失败")
                return False
            
            # 计算交易信号
            signal, prob = self.calculate_signal(df)
            
            # 风控检查
            self.check_risk_management()
            
            # 每日收盘前平仓检查
            self.check_daily_close()
            
            # 检查是否需要根据信号平仓
            if self.check_and_close_by_signal(signal):
                logger.info("已根据信号平仓")
            
            # 如果没有持仓且有明确信号，则开仓
            if self.current_position is None and signal in ["BUY", "SELL"]:
                if prob > 0.6:  # 确保信号足够强
                    logger.info(f"开仓: {signal} 信号，置信度 {prob:.3f}")
                    self.place_order(signal)
                else:
                    logger.info(f"信号置信度 {prob:.3f} 不足，暂不交易")
            
            # 检查持仓状态
            if self.current_position is not None:
                # 获取当前持仓盈亏
                try:
                    positions = mt5.positions_get(symbol=self.SYMBOL)
                    if positions and len(positions) > 0:
                        pos = positions[0]
                        profit = pos.profit
                        direction = "做多" if pos.type == mt5.POSITION_TYPE_BUY else "做空"
                        logger.info(f"当前持仓: {direction}, 持仓收益: {profit:.2f}美金")
                        
                        # 记录当前市场状态
                        tick = mt5.symbol_info_tick(self.SYMBOL)
                        if tick:
                            current_price = tick.ask if pos.type == mt5.POSITION_TYPE_BUY else tick.bid
                            logger.info(f"当前价格: {current_price:.5f}, 入场价格: {pos.price_open:.5f}, 点位变化: {abs(current_price - pos.price_open):.5f}")
                    else:
                        logger.info("当前无持仓")
                        self.current_position = None
                except Exception as e:
                    logger.error(f"获取持仓信息失败: {e}")
            else:
                logger.info("当前无持仓")
            
            return True
            
        except Exception as e:
            logger.error(f"交易循环执行失败: {e}")
            return False

    def run_trading_loop(self, interval=300):  # 5分钟间隔，对应M5周期
        """运行交易循环"""
        self.is_running = True
        logger.info("开始M5实时交易循环")
        
        # 记录上次增量训练时间
        last_training_time = datetime.now()
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # 执行单次交易循环
                self.run_trading_cycle()
                
                # 每小时执行一次增量训练
                current_time = datetime.now()
                if (current_time - last_training_time).total_seconds() >= 3600:  # 1小时
                    self.incremental_training()
                    last_training_time = current_time
                
                # 等待到下一个M5周期
                # 使用市场时间而不是系统时间
                current_tick = mt5.symbol_info_tick(self.SYMBOL)
                if current_tick is not None:
                    now_market_time = datetime.fromtimestamp(current_tick.time)
                    minutes = now_market_time.minute
                    seconds = now_market_time.second
                else:
                    # 如果无法获取市场时间，则使用系统时间作为备选
                    now_market_time = datetime.now()
                    minutes = now_market_time.minute
                    seconds = now_market_time.second
                
                # 计算到下一个M5周期的时间
                next_minute = ((minutes // 5) + 1) * 5
                if next_minute >= 60:
                    next_minute = 0
                
                # 计算等待时间（秒）
                if next_minute > minutes:
                    wait_seconds = (next_minute - minutes) * 60 - seconds
                elif next_minute < minutes:
                    wait_seconds = (next_minute + 60 - minutes) * 60 - seconds
                else:
                    wait_seconds = 300 - seconds  # 5分钟 = 300秒
                
                logger.info(f"等待 {wait_seconds} 秒到下一个M5周期")
                
                # 在等待期间，定期检查是否需要停止
                wait_start = time.time()
                while time.time() - wait_start < wait_seconds and not self.stop_event.is_set():
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"交易循环异常: {e}")
                time.sleep(10)  # 异常后暂停10秒再继续
        
        logger.info("M5实时交易循环结束")

    def stop_trading(self):
        """停止交易"""
        logger.info("正在停止交易...")
        self.is_running = False
        self.stop_event.set()
        
        # 如果有持仓，执行平仓
        if self.current_position is not None:
            logger.info("检测到持仓，执行平仓")
            self.close_position("停止交易")
        
        # 关闭MT5连接
        mt5.shutdown()
        logger.info("MT5连接已关闭")

def main():
    """主函数"""
    trader = None
    try:
        # 创建交易实例
        trader = M5RealTimeTrader(
            model_path="xauusd_m5_model.json",  # 使用训练好的模型
            lot_size=0.2  # 固定手数
        )
        
        # 运行交易循环
        trader.run_trading_loop()
        
    except Exception as e:
        logger.error(f"交易程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if trader:
            trader.stop_trading()

if __name__ == "__main__":
    main()