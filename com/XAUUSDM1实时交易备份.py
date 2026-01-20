"""
1.缺少ftmo每日交易限制
2.容易由赢转亏
"""
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import time
import os
import json
from datetime import datetime, timedelta
import glob
from threading import Thread, Event
import warnings
import subprocess
import sys
import importlib.util


# 时区处理
from datetime import datetime, timezone
import pytz


warnings.filterwarnings('ignore')

# 设置UTC+2时区
UTC_PLUS_2 = pytz.timezone('Etc/GMT-2')  # 注意：GMT-2 实际上是UTC+2

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class XAUUSDM1RealTimeTrader:
    """XAUUSD M1 实时交易系统"""
    
    def __init__(self):
        # 初始化MT5
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return None
            
        # 检查交易品种
        self.symbol = "XAUUSD"
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"❌ 品种 {self.symbol} 不可用")
            mt5.shutdown()
            return None

        if not symbol_info.visible:
            print(f"✅ 启用品种 {self.symbol}...")
            if not mt5.symbol_select(self.symbol, True):
                print(f"❌ 启用品种失败")
                mt5.shutdown()
                return None

        # 获取真实账户信息
        account_info = mt5.account_info()
        if account_info is None:
            print(f"❌ 无法获取账户信息")
            mt5.shutdown()
            return None

        # 交易参数
        self.fixed_lot_size = 0.2  # 固定手数0.2手
        self.initial_balance = account_info.balance  # 使用真实账户余额
        self.current_balance = self.initial_balance
        self.active_positions = []  # 活跃持仓
        self.trade_history = []  # 交易历史
        self.magic_number = 234000  # 魔法数字
        
        # 控制变量
        self.running = False
        self.trade_thread = None
        self.stop_event = Event()
        
        # 信号文件路径
        self.signals_dir = "."  # 当前目录
        self.last_signal_time = None
        self.current_signal = None
        
        # 启动时检查是否有持仓
        self.check_existing_positions()
        
        print("=" * 80)
        print("💰 XAUUSD M1 实时交易系统")
        print(f"📊 初始账户资金: ${self.initial_balance:,}")
        print(f"📈 固定手数: {self.fixed_lot_size}手")
        print(f"🔮 魔法数字: {self.magic_number}")
        print("🔄 系统将在每30秒检查一次交易信号")
        print("🔄 请确保m1_data_analyzer_and_trainer.py已生成交易信号")
        print("=" * 80)
    
    
    def get_latest_m1_time(self):
        """获取最新的M1数据时间（UTC+2时区）"""
        print(f"\n📡 获取XAUUSD最新M1数据时间...")
        
        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return datetime.now(UTC_PLUS_2)  # 如果连接失败，返回UTC+2时区的当前时间
        
        # 检查交易品种
        symbol = "XAUUSD"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ 品种 {symbol} 不可用")
            mt5.shutdown()
            return datetime.now(UTC_PLUS_2)

        if not symbol_info.visible:
            print(f"✅ 启用品种 {symbol}...")
            if not mt5.symbol_select(symbol, True):
                print(f"❌ 启用品种失败")
                mt5.shutdown()
                return datetime.now(UTC_PLUS_2)

        # 获取最近的M1数据
        # 只需要获取最新的1根K线
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        
        if rates is None or len(rates) == 0:
            print(f"❌ 未获取到最新的M1数据")
            mt5.shutdown()
            return datetime.now(UTC_PLUS_2)
        
        # 获取最新K线的时间
        latest_time = pd.to_datetime(rates[0]['time'], unit='s', utc=True)
        # 转换为UTC+2时区
        latest_time = latest_time.tz_convert(UTC_PLUS_2)
        
        print(f"✅ 最新M1数据时间（UTC+2）: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 断开MT5连接
        mt5.shutdown()
        
        return latest_time
    
    
    
    
    def extract_current_features(self):
        """从当前市场数据中提取特征用于AI预测"""
        print("🔍 提取当前市场特征...")
        
        # 获取最新的M1数据
        data = self.get_latest_m1_data(count=250)  # 获取更多数据用于特征计算
        if data is None or len(data) < 50:  # 确保有足够的数据
            print("❌ 数据不足，无法提取特征")
            return None
        
        # 使用最后一条数据点作为当前时刻
        current_data = data.iloc[-1]
        
        # 计算各种技术指标
        close_prices = data['close']
        
        # 计算RSI
        rsi_values = self.calculate_rsi_simple(close_prices.values)
        current_rsi = rsi_values.iloc[-1] if rsi_values is not None and not pd.isna(rsi_values.iloc[-1]) else 0
        
        # 计算布林带
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close_prices)
        current_bb_upper = bb_upper.iloc[-1] if bb_upper is not None and not pd.isna(bb_upper.iloc[-1]) else 0
        current_bb_middle = bb_middle.iloc[-1] if bb_middle is not None and not pd.isna(bb_middle.iloc[-1]) else 0
        current_bb_lower = bb_lower.iloc[-1] if bb_lower is not None and not pd.isna(bb_lower.iloc[-1]) else 0
        
        # 计算价格相对于布林带的位置
        if current_bb_upper != current_bb_lower and (current_bb_upper - current_bb_lower) != 0:
            bb_position = (current_data['close'] - current_bb_lower) / (current_bb_upper - current_bb_lower)
        else:
            bb_position = 0.5  # 中间位置
        
        # 时间特征 - 使用UTC+2时区
        current_time = current_data['timestamp']
        hour = current_time.hour
        weekday = current_time.weekday()
        
        # 一周七天标识特征
        is_monday = 1 if weekday == 0 else 0
        is_tuesday = 1 if weekday == 1 else 0
        is_wednesday = 1 if weekday == 2 else 0
        is_thursday = 1 if weekday == 3 else 0
        is_friday = 1 if weekday == 4 else 0
        is_saturday = 1 if weekday == 5 else 0
        is_sunday = 1 if weekday == 6 else 0
        
        # 交易时段特征
        session_asia = 1 if 0 <= hour <= 8 else 0
        session_europe = 1 if 9 <= hour <= 17 else 0
        session_us = 1 if 18 <= hour <= 23 else 0
        
        # 价格特征
        price_round = round(current_data['close'] / 10) * 10  # 价格整数位（心理关口）
        
        # 计算近期波动率
        recent_returns = close_prices.pct_change().tail(20).dropna()
        volatility = recent_returns.std() if len(recent_returns) > 1 else 0
        
        # 计算滚动振幅（使用近期价格变化的均值）
        recent_changes = close_prices.pct_change().tail(5).abs().mean() if len(close_prices) >= 5 else 0
        
        # 计算成交量相关特征
        recent_volume = data['volume'].tail(10).mean() if len(data) >= 10 else 0
        current_volume = current_data['volume']
        volume_ma_ratio = current_volume / recent_volume if recent_volume != 0 else 1.0  # 成交量与平均成交量比率
        
        # 计算趋势方向（基于最近的价格变化）
        if len(close_prices) >= 2:
            trend_direction = 1 if close_prices.iloc[-1] > close_prices.iloc[-2] else 0
        else:
            trend_direction = 0
        
        # 计算连续同向趋势的数量
        recent_trend_directions = []
        for i in range(max(0, len(close_prices)-10), len(close_prices)-1):
            if i >= 1:
                recent_trend_directions.append(1 if close_prices.iloc[i] > close_prices.iloc[i-1] else 0)
        consecutive_same_trend = 0
        if recent_trend_directions:
            current_direction = trend_direction
            streak = 1
            for direction in reversed(recent_trend_directions):
                if direction == current_direction:
                    streak += 1
                else:
                    break
            consecutive_same_trend = streak
        
        # 趋势持续时间的移动平均
        trend_duration_ma = close_prices.tail(5).mean() if len(close_prices) >= 5 else current_data['close']
        
        # 趋势强度（幅度/持续时间）- 简化计算
        trend_strength = (close_prices.iloc[-1] - close_prices.iloc[-5]) / 5 if len(close_prices) >= 5 else 0
        
        # 价格偏离均线的程度
        price_ma_10 = close_prices.tail(10).mean() if len(close_prices) >= 10 else current_data['close']
        price_std_10 = close_prices.tail(10).std() if len(close_prices) >= 10 else 0
        price_deviation = (current_data['close'] - price_ma_10) / price_std_10 if price_std_10 != 0 else 0
        
        # RSI确认突破（简化）
        rsi_confirmation = 0
        if not pd.isna(current_rsi):
            if bb_position > 0.8 and current_rsi > 70:  # 高位且超买
                rsi_confirmation = 1
            elif bb_position < 0.2 and current_rsi < 30:  # 低位且超卖
                rsi_confirmation = 1
            elif 0.4 <= bb_position <= 0.6:  # 中位区域
                rsi_confirmation = 1
        
        # 支撑阻力位（基于滚动窗口的高低点）
        resistance = close_prices.tail(20).max() if len(close_prices) >= 20 else current_data['close']
        support = close_prices.tail(20).min() if len(close_prices) >= 20 else current_data['close']
        
        # 判断是否突破支撑阻力位
        breaks_resistance = 1 if current_data['close'] > resistance * 0.999 else 0
        breaks_support = 1 if current_data['close'] < support * 1.001 else 0
        
        # 突破有效性判断（简化）
        break_validity = 0
        if breaks_resistance == 1:
            # 向上突破，看是否维持高位
            future_prices = close_prices.tail(5) if len(close_prices) >= 5 else close_prices
            if len(future_prices) > 0 and (future_prices > resistance * 0.999).any():
                break_validity = 1  # 有效突破
            else:
                break_validity = -1  # 假突破
        elif breaks_support == 1:
            # 向下突破，看是否维持低位
            future_prices = close_prices.tail(5) if len(close_prices) >= 5 else close_prices
            if len(future_prices) > 0 and (future_prices < support * 1.001).any():
                break_validity = 1  # 有效突破
            else:
                break_validity = -1  # 假突破
        
        # 高成交量确认突破
        volume_ma = data['volume'].tail(20).mean() if len(data) >= 20 else current_volume
        high_volume_on_breakout = 1 if (breaks_resistance or breaks_support) and current_volume > volume_ma * 1.5 else 0
        
        # 亚盘突破特征（简化）
        euro_breaks_asian_high = 0
        euro_breaks_asian_low = 0
        us_breaks_asian_high = 0
        us_breaks_asian_low = 0
        
        # 计算均线方向一致性
        ma_direction_consistency = (
            (sma_5_direction == sma_10_direction) + 
            (sma_10_direction == sma_20_direction) + 
            (sma_5_direction == sma_20_direction)
        )
        
        # 计算RSI与价格方向一致性
        # 获取前一个价格和RSI值
        prev_price = close_prices.iloc[-2] if len(close_prices) >= 2 else current_data['close']
        
        rsi_price_consistency = 1 if (current_data['close'] > prev_price) == (current_rsi > 50) else 0
        
        # 新高新低特征
        new_high = 1 if current_data['close'] == close_prices.tail(20).max() else 0 if len(close_prices) >= 20 else 0
        new_low = 1 if current_data['close'] == close_prices.tail(20).min() else 0 if len(close_prices) >= 20 else 0
        
        # 构建特征向量 - 确保与训练时的特征列一致
        features = pd.DataFrame([{
            'hour': hour,
            'weekday': weekday,
            'is_monday': is_monday,
            'is_tuesday': is_tuesday,
            'is_wednesday': is_wednesday,
            'is_thursday': is_thursday,
            'is_friday': is_friday,
            'is_saturday': is_saturday,
            'is_sunday': is_sunday,
            'session_asia': session_asia,
            'session_europe': session_europe,
            'session_us': session_us,
            'start_price': current_data['close'],
            'amplitude_is_multiple_of_ten': amplitude_is_multiple_of_ten,
            'sma_5_direction': sma_5_direction,
            'sma_10_direction': sma_10_direction,
            'sma_20_direction': sma_20_direction,
            'rsi_direction': rsi_direction,
            'ma_direction_consistency': ma_direction_consistency,
            'rsi_price_consistency': rsi_price_consistency,
            'amplitude_ratio': volatility,  # 使用波动率作为振幅比率
            'rolling_amplitude': recent_changes,  # 使用近期价格变化均值作为滚动振幅
            'consecutive_same_trend': consecutive_same_trend,
            'trend_duration_ma': trend_duration_ma,
            'trend_strength': trend_strength,
            'price_deviation': price_deviation,
            'rsi': current_rsi,
            'bb_position': bb_position,
            'volatility': volatility,
            'euro_breaks_asian_high': euro_breaks_asian_high,
            'euro_breaks_asian_low': euro_breaks_asian_low,
            'us_breaks_asian_high': us_breaks_asian_high,
            'us_breaks_asian_low': us_breaks_asian_low,
            'new_high': new_high,
            'new_low': new_low,
            'breaks_resistance': breaks_resistance,
            'breaks_support': breaks_support,
            'break_validity': break_validity,
            'high_volume_on_breakout': high_volume_on_breakout,
            'rsi_confirmation': rsi_confirmation,
            'start_volume': current_volume,
            'volume_ma_ratio': volume_ma_ratio  # 添加成交量比率特征
        }])
        
        # 确保特征列的顺序与训练时一致
        if hasattr(self.scaler, 'feature_names_in_'):
            feature_order = self.scaler.feature_names_in_
            # 确保所有必需的特征都在features中
            for col in feature_order:
                if col not in features.columns:
                    features[col] = 0  # 添加缺失的特征列，填充0
            features = features[feature_order]  # 按照训练时的顺序重新排列
        
        # 使用之前加载的scaler进行标准化
        try:
            features_scaled = self.scaler.transform(features)
            print("✅ 特征提取完成")
            return features_scaled
        except Exception as e:
            print(f"❌ 特征标准化失败: {str(e)}")
            # 如果特征不匹配，尝试重新排列特征顺序以匹配训练时的特征
            try:
                # 获取训练时的特征名称
                feature_names = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else None
                if feature_names is not None:
                    # 重新排列特征列以匹配训练时的顺序
                    features = features.reindex(columns=feature_names, fill_value=0)
                    features_scaled = self.scaler.transform(features)
                    print("✅ 特征提取完成（已修复列顺序）")
                    return features_scaled
            except Exception as e2:
                print(f"❌ 特征修复失败: {str(e2)}")
                return None
    
    
    def check_existing_positions(self):
        """启动时检查是否有现有持仓"""
        # 获取所有持仓
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            print(f"⚠️ 检测到 {len(positions)} 个现有持仓，系统将接管管理:")
            for pos in positions:
                # 从MT5获取实际持仓信息
                position_info = {
                    'ticket': pos.ticket,
                    'direction': '做多' if pos.type == mt5.POSITION_TYPE_BUY else '做空',
                    'entry_price': pos.price_open,
                    'volume': pos.volume,
                    'entry_time': datetime.now(UTC_PLUS_2),  # 实际无法获取，使用当前UTC+2时间
                    'take_profit': pos.tp,  # 使用实际持仓的止盈价格
                    'stop_loss': pos.sl,  # 使用实际持仓的止损价格
                    'lot_size': pos.volume,
                    'confidence': 0  # 对于现有持仓，置信度未知，设为0
                }
                
                # 计算预计平仓时间（如果有的话）
                if hasattr(pos, 'expiration') and pos.expiration:
                    position_info['expected_close_time'] = pd.to_datetime(pos.expiration, unit='s', utc=True).tz_convert(UTC_PLUS_2)
                else:
                    # 如果没有到期时间，使用默认30分钟
                    position_info['expected_close_time'] = datetime.now(UTC_PLUS_2) + timedelta(minutes=30)
                
                self.active_positions.append(position_info)
                print(f"   持仓单号: {pos.ticket}, 方向: {position_info['direction']}, 手数: {pos.volume}")
        else:
            print("✅ 未检测到现有持仓，系统正常启动")
    
    def load_latest_signal(self):
        """直接从m1_data_analyzer_and_trainer模块获取最新的交易信号"""
        try:
            # 动态导入m1_data_analyzer_and_trainer模块
            spec = importlib.util.spec_from_file_location("m1_data_analyzer_and_trainer", "./m1_data_analyzer_and_trainer.py")
            m1_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m1_module)
            
            # 直接调用模块中的run函数获取最新信号
            latest_signal = m1_module.run()
            
            if latest_signal is None:
                raise Exception("❌ m1_data_analyzer_and_trainer未返回任何交易信号")
            
            # 检查信号时间是否更新
            signal_time_str = latest_signal.get('开仓时间', '')
            if not signal_time_str:
                raise Exception("❌ 交易信号中缺少开仓时间，m1_data_analyzer_and_trainer生成的信号格式错误")
            
            signal_time = pd.to_datetime(signal_time_str)
            # 将信号时间转换为UTC+2时区
            if signal_time.tz is None:
                # 如果信号时间没有时区信息，则假定为本地时间并转换为UTC+2
                signal_time = signal_time.tz_localize('Etc/GMT-2')
            else:
                # 如果信号时间有时区信息，转换为UTC+2
                signal_time = signal_time.tz_convert(UTC_PLUS_2)
            
            if self.last_signal_time and signal_time.replace(tzinfo=None) <= self.last_signal_time.replace(tzinfo=None):
                # 信号没有更新
                return None
            
            # 检查是否到达开仓时间（允许1分钟误差）
            current_time = datetime.now(UTC_PLUS_2)  # 使用UTC+2时区
            
            # 如果信号时间在过去超过1分钟，则跳过执行
            if signal_time.astimezone(UTC_PLUS_2).replace(tzinfo=None) < current_time.replace(tzinfo=None) - timedelta(minutes=1):
                print(f"⏰ 已超过开仓时间1分钟以上，跳过执行: {signal_time_str}")
                return None
            
            # 如果信号时间在未来超过5分钟，则等待执行
            if signal_time.astimezone(UTC_PLUS_2).replace(tzinfo=None) > current_time.replace(tzinfo=None) + timedelta(minutes=5):
                print(f"⏰ 未到达开仓时间，等待执行: {signal_time_str}")
                return None
            
            # 如果信号时间在未来1-5分钟内，则准备执行
            if signal_time.astimezone(UTC_PLUS_2).replace(tzinfo=None) > current_time.replace(tzinfo=None):
                print(f"⏰ 信号已就绪，等待开仓时间: {signal_time_str}")
                # 可以提前准备，但不执行交易
                return latest_signal
            
            self.last_signal_time = signal_time
            self.current_signal = latest_signal
            
            # 检查信号是否有效
            if latest_signal.get('信号有效性', '') == '✅ 有效':
                print(f"✅ 从m1_data_analyzer_and_trainer.py获取到新的有效交易信号:")
                print(f"   交易方向: {latest_signal.get('交易方向', '')}")
                print(f"   开仓时间: {signal_time_str}")
                print(f"   持仓时长: {latest_signal.get('持仓时长(分钟)', 0)}分钟")
                print(f"   止盈幅度: {latest_signal.get('止盈幅度(美元)', 0)}美元")
                print(f"   止损幅度: {latest_signal.get('止损幅度(美元)', 0)}美元")
                print(f"   置信度: {latest_signal.get('置信度(%)', 0)}%")
                return latest_signal
            else:
                raise Exception(f"❌ 从m1_data_analyzer_and_trainer.py获取到的交易信号无效，跳过执行")
                
        except ImportError:
            raise Exception("❌ 无法导入m1_data_analyzer_and_trainer模块")
        except Exception as e:
            raise Exception(f"❌ 获取交易信号失败: {str(e)}")
    
    def get_current_price(self):
        """获取当前市场价格"""
        # 确保MT5连接正常
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return None, None
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            print(f"❌ 无法获取 {self.symbol} 的当前价格")
            return None, None
        
        return tick.ask, tick.bid
    
    def place_order(self, signal):
        """根据信号下单"""
        direction = signal.get('交易方向', '')
        lot_size = self.fixed_lot_size
        take_profit_usd = float(signal.get('止盈幅度(美元)', 0))
        stop_loss_usd = float(signal.get('止损幅度(美元)', 0))
        confidence = float(signal.get('置信度(%)', 0))
        duration_minutes = int(signal.get('持仓时长(分钟)', 0))
        
        # 获取信号开仓时间
        signal_time_str = signal.get('开仓时间', '')
        if not signal_time_str:
            print("❌ 信号中缺少开仓时间")
            return False
        
        signal_time = pd.to_datetime(signal_time_str)
        # 将信号时间转换为UTC+2时区
        if signal_time.tz is None:
            signal_time = signal_time.tz_localize('Etc/GMT-2')
        else:
            signal_time = signal_time.tz_convert(UTC_PLUS_2)
        
        # 检查当前时间是否接近信号开仓时间（在1分钟内）
        current_time = datetime.now(UTC_PLUS_2)
        time_diff = abs((signal_time.astimezone(UTC_PLUS_2).replace(tzinfo=None) - current_time.replace(tzinfo=None)).total_seconds())
        
        if time_diff > 60:  # 如果时间差超过1分钟，暂不执行
            print(f"⏰ 信号开仓时间未到或已过期，跳过执行: {signal_time_str}")
            return False
        
        # 检查MT5连接状态，如果未连接则初始化
        if mt5.account_info() is None:
            if not mt5.initialize():
                print(f"❌ MT5初始化失败: {mt5.last_error()}")
                return False
        
        # 获取当前价格
        current_ask, current_bid = self.get_current_price()
        if current_ask is None or current_bid is None:
            print("❌ 无法获取当前价格，下单失败")
            return False
        
        # 检查MT5中的实际持仓
        mt5_positions = mt5.positions_get(symbol=self.symbol)
        if mt5_positions and len(mt5_positions) >= 1:  # 检查MT5中是否已有持仓
            existing_position = mt5_positions[0]  # 只处理第一个持仓
            existing_direction = '做多' if existing_position.type == mt5.POSITION_TYPE_BUY else '做空'
            
            # 如果新信号方向与当前持仓方向相同，则不能开仓
            if existing_direction == direction:
                print(f"⚠️ MT5中已有{existing_direction}持仓，无法开立同方向新仓")
                # 同步本地持仓列表
                self.active_positions = []
                for pos in mt5_positions:
                    position_info = {
                        'ticket': pos.ticket,
                        'direction': '做多' if pos.type == mt5.POSITION_TYPE_BUY else '做空',
                        'entry_price': pos.price_open,
                        'volume': pos.volume,
                        'entry_time': datetime.now(UTC_PLUS_2),
                        'take_profit': pos.tp,
                        'stop_loss': pos.sl,
                        'lot_size': pos.volume,
                        'confidence': 0
                    }
                    self.active_positions.append(position_info)
                return False
            else:
                # 如果新信号方向与当前持仓方向相反，则先平掉当前持仓，再开新仓
                print(f"⚠️ 检测到反向信号({direction})，与当前持仓方向({existing_direction})相反，先平掉现有持仓")
                
                # 执行平仓
                close_success = self.close_position_directly(existing_position, "反向开仓", 0)  # 临时利润为0，实际盈亏会在close_position中计算
                if close_success:
                    print(f"✅ 原{existing_direction}持仓已平仓，准备开立{direction}新仓")
                    # 等待一段时间确保平仓完成
                    time.sleep(1)
                else:
                    print(f"❌ 原{existing_direction}持仓平仓失败，取消开仓")
                    return False
        
        # 检查本地持仓限制
        if len(self.active_positions) >= 1:  # 限制同时只能有一个持仓
            # 检查本地持仓方向与新信号方向是否一致
            existing_direction = self.active_positions[0]['direction']
            
            if existing_direction == direction:
                print(f"⚠️ 当前已有{existing_direction}持仓，无法开立同方向新仓")
                return False
            else:
                # 如果新信号方向与当前持仓方向相反，则先平掉当前持仓，再开新仓
                print(f"⚠️ 检测到反向信号({direction})，与当前持仓方向({existing_direction})相反，先平掉现有持仓")
                position_to_close = self.active_positions[0]
                
                # 获取当前价格
                current_ask, current_bid = self.get_current_price()
                if current_ask is None or current_bid is None:
                    print("❌ 无法获取当前价格，平仓失败")
                    return False
                
                # 根据持仓方向确定平仓价格
                if position_to_close['direction'] == '做多':
                    profit = (current_bid - position_to_close['entry_price']) * position_to_close['lot_size'] * 100
                else:  # 做空
                    profit = (position_to_close['entry_price'] - current_ask) * position_to_close['lot_size'] * 100
                
                # 执行平仓
                close_success = self.close_position(position_to_close, "反向开仓", profit)
                if close_success:
                    print(f"✅ 原{existing_direction}持仓已平仓，准备开立{direction}新仓")
                    # 从本地列表移除已平仓的持仓
                    self.active_positions.remove(position_to_close)
                    # 等待一段时间确保平仓完成
                    time.sleep(1)
                else:
                    print(f"❌ 原{existing_direction}持仓平仓失败，取消开仓")
                    return False
        
        # 计算止盈止损价格
        if direction == '做多':
            # 做多：开仓价 + 止盈点数，开仓价 - 止损点数
            entry_price = current_ask
            # 止盈止损应该直接基于信号中的美元幅度
            take_profit = round(entry_price + take_profit_usd, 2)  # 精确到小数点后2位
            stop_loss = round(entry_price - stop_loss_usd, 2)  # 精确到小数点后2位
            
            # 创建买入订单
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": self.magic_number,  # 使用魔法数字
                "comment": f"AI信号做多 {confidence}%置信度 持仓{duration_minutes}分钟",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
        elif direction == '做空':
            # 做空：开仓价 - 止盈点数，开仓价 + 止损点数
            entry_price = current_bid
            take_profit = round(entry_price - take_profit_usd, 2)  # 精确到小数点后2位
            stop_loss = round(entry_price + stop_loss_usd, 2)  # 精确到小数点后2位
            
            # 创建卖出订单
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": self.magic_number,  # 使用魔法数字
                "comment": f"AI信号做空 {confidence}%置信度 持仓{duration_minutes}分钟",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        else:
            print("❌ 未知的交易方向")
            return False
        
        # 确保止盈止损价格合理
        if direction == '做多':
            if stop_loss >= entry_price or entry_price >= take_profit or stop_loss >= take_profit:
                print(f"❌ 止盈止损价格不合理: SL({stop_loss}) < Entry({entry_price}) < TP({take_profit})")
                return False
        else:  # 做空
            if take_profit >= entry_price or entry_price >= stop_loss or take_profit >= stop_loss:
                print(f"❌ 止盈止损价格不合理: TP({take_profit}) < Entry({entry_price}) < SL({stop_loss})")
                return False
        
        # 发送订单
        result = mt5.order_send(request)
        if result is None:
            print(f"❌ 订单发送失败: 无返回结果")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ 订单执行失败: {result.retcode} - {result.comment}")
            return False
        
        # 记录持仓
        position_info = {
            'ticket': result.order,
            'direction': direction,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'lot_size': lot_size,
            'entry_time': datetime.now(UTC_PLUS_2),  # 使用UTC+2时区
            'confidence': confidence,
            'duration_minutes': duration_minutes,  # 预计持仓时长
            'expected_close_time': datetime.now(UTC_PLUS_2) + timedelta(minutes=duration_minutes)  # 预计平仓时间，使用UTC+2时区
        }
        
        self.active_positions.append(position_info)
        
        print(f"✅ {direction}订单已成交:")
        print(f"   手数: {lot_size}")
        print(f"   价格: {entry_price}")
        print(f"   止盈: {take_profit}")
        print(f"   止损: {stop_loss}")
        print(f"   预计持仓: {duration_minutes}分钟")
        print(f"   预计平仓时间: {position_info['expected_close_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   置信度: {confidence}%")
        
        return True
    
    def check_close_conditions(self):
        """检查平仓条件"""
        # 首先同步MT5中的实际持仓状态
        mt5_positions = mt5.positions_get(symbol=self.symbol)
        mt5_position_tickets = set() if mt5_positions is None else {pos.ticket for pos in mt5_positions}
        
        # 检查本地持仓列表中是否有实际已不存在的持仓，从本地列表中移除
        positions_to_remove = []
        for position in self.active_positions:
            if position['ticket'] not in mt5_position_tickets:
                print(f"⚠️ 持仓单号 {position['ticket']} 在MT5中已不存在，从本地列表中移除")
                positions_to_remove.append(position)
        
        for position in positions_to_remove:
            self.active_positions.remove(position)
        
        positions_to_close = []
        
        for position in self.active_positions[:]:  # 使用副本遍历
            # 检查MT5连接状态，如果未连接则初始化
            if mt5.account_info() is None:
                if not mt5.initialize():
                    print(f"❌ MT5初始化失败: {mt5.last_error()}")
                    continue
            
            # 获取当前价格
            current_ask, current_bid = self.get_current_price()
            if current_ask is None or current_bid is None:
                continue
            
            # 检查止盈止损条件
            should_close = False
            close_reason = ""
            
            # 检查持仓是否包含必要的字段
            has_tp_sl = 'take_profit' in position and 'stop_loss' in position
            has_expected_close_time = 'expected_close_time' in position
            has_lot_size = 'lot_size' in position
            
            if position['direction'] == '做多':
                current_price = current_bid  # 平多单用bid价
                if has_tp_sl:
                    if current_price >= position['take_profit']:
                        should_close = True
                        close_reason = "止盈"
                    elif current_price <= position['stop_loss']:
                        should_close = True
                        close_reason = "止损"
                else:
                    # 对于现有持仓，我们通过MT5获取实际的止盈止损
                    pos_info = mt5.positions_get(ticket=position['ticket'])
                    if pos_info:
                        pos_info = pos_info[0]
                        if pos_info.tp != 0 and current_price >= pos_info.tp:
                            should_close = True
                            close_reason = "止盈"
                        elif pos_info.sl != 0 and current_price <= pos_info.sl:
                            should_close = True
                            close_reason = "止损"
            else:  # 做空
                current_price = current_ask  # 平空单用ask价
                if has_tp_sl:
                    if current_price <= position['take_profit']:
                        should_close = True
                        close_reason = "止盈"
                    elif current_price >= position['stop_loss']:
                        should_close = True
                        close_reason = "止损"
                else:
                    # 对于现有持仓，我们通过MT5获取实际的止盈止损
                    pos_info = mt5.positions_get(ticket=position['ticket'])
                    if pos_info:
                        pos_info = pos_info[0]
                        if pos_info.tp != 0 and current_price <= pos_info.tp:
                            should_close = True
                            close_reason = "止盈"
                        elif pos_info.sl != 0 and current_price >= pos_info.sl:
                            should_close = True
                            close_reason = "止损"
            
            # 检查是否达到预计持仓时长（时间到期平仓）
            # 如果剩余时间小于等于0，则执行平仓
            if not should_close and has_expected_close_time:
                time_left = position['expected_close_time'] - datetime.now(UTC_PLUS_2)  # 使用UTC+2时区
                if time_left.total_seconds() <= 0:
                    should_close = True
                    close_reason = "时间到期"
            
            if should_close:
                # 计算盈亏
                lot_size = position.get('lot_size', position.get('volume', 0))
                if position['direction'] == '做多':
                    profit = (current_price - position['entry_price']) * lot_size * 100
                else:
                    profit = (position['entry_price'] - current_price) * lot_size * 100
                
                # 执行平仓
                if self.close_position(position, close_reason, profit):
                    positions_to_close.append(position)
        
        # 从活跃持仓列表中移除已平仓的持仓
        for position in positions_to_close:
            if position in self.active_positions:
                self.active_positions.remove(position)
    
    def close_position(self, position, reason, profit):
        """平仓操作"""
        # 检查MT5连接状态，如果未连接则初始化
        if mt5.account_info() is None:
            if not mt5.initialize():
                print(f"❌ MT5初始化失败: {mt5.last_error()}")
                return False
        
        # 获取当前价格
        current_ask, current_bid = self.get_current_price()
        if current_ask is None or current_bid is None:
            print("❌ 无法获取当前价格，平仓失败")
            return False
        
        # 通过MT5 API查询当前持仓以获取准确的手数
        positions = mt5.positions_get(ticket=position['ticket'])
        if not positions:
            print(f"❌ 未找到持仓单号 {position['ticket']}，可能已被平仓")
            return True  # 如果持仓不存在，认为已经平仓成功
        
        pos = positions[0]  # 获取持仓信息
        volume = pos.volume  # 使用实际持仓手数
        
        # 准备平仓请求
        if position['direction'] == '做多':
            # 做多平仓用SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,  # 使用实际持仓手数
                "type": mt5.ORDER_TYPE_SELL,
                "position": position['ticket'],
                "magic": self.magic_number,
                "comment": f"AI信号{reason}平仓",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            exit_price = current_bid  # 平多单用bid价
        else:  # 做空
            # 做空平仓用BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,  # 使用实际持仓手数
                "type": mt5.ORDER_TYPE_BUY,
                "position": position['ticket'],
                "magic": self.magic_number,
                "comment": f"AI信号{reason}平仓",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            exit_price = current_ask  # 平空单用ask价
        
        # 发送平仓订单
        result = mt5.order_send(request)
        if result is None:
            print(f"❌ 平仓失败: 无返回结果")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ 平仓失败: {result.retcode} - {result.comment}")
            return False
        
        # 更新账户余额
        self.current_balance += profit
        
        # 记录交易历史
        trade_record = {
            'close_time': datetime.now(UTC_PLUS_2),  # 使用UTC+2时区
            'direction': position['direction'],
            'entry_price': position.get('entry_price'),
            'exit_price': exit_price,  # 使用正确的平仓价格
            'profit': profit,
            'reason': reason,
            'confidence': position.get('confidence', 0)
        }
        self.trade_history.append(trade_record)
        
        print(f"✅ {position['direction']}仓位已{reason}平仓:")
        print(f"   盈亏: ${profit:.2f}")
        print(f"   当前余额: ${self.current_balance:.2f}")
        
        return True

    def close_position_directly(self, mt5_position, reason, profit):
        """直接平仓操作，传入MT5持仓对象"""
        # 检查MT5连接状态，如果未连接则初始化
        if mt5.account_info() is None:
            if not mt5.initialize():
                print(f"❌ MT5初始化失败: {mt5.last_error()}")
                return False
        
        # 获取当前价格
        current_ask, current_bid = self.get_current_price()
        if current_ask is None or current_bid is None:
            print("❌ 无法获取当前价格，平仓失败")
            return False
        
        # 使用传入的MT5持仓对象
        volume = mt5_position.volume  # 使用实际持仓手数
        ticket = mt5_position.ticket
        direction = '做多' if mt5_position.type == mt5.POSITION_TYPE_BUY else '做空'
        
        # 准备平仓请求
        if direction == '做多':
            # 做多平仓用SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,  # 使用实际持仓手数
                "type": mt5.ORDER_TYPE_SELL,
                "position": ticket,
                "magic": self.magic_number,
                "comment": f"AI信号{reason}平仓",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            exit_price = current_bid  # 平多单用bid价
        else:  # 做空
            # 做空平仓用BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,  # 使用实际持仓手数
                "type": mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "magic": self.magic_number,
                "comment": f"AI信号{reason}平仓",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            exit_price = current_ask  # 平空单用ask价
        
        # 发送平仓订单
        result = mt5.order_send(request)
        if result is None:
            print(f"❌ 平仓失败: 无返回结果")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ 平仓失败: {result.retcode} - {result.comment}")
            return False
        
        # 计算盈亏
        if direction == '做多':
            calculated_profit = (current_bid - mt5_position.price_open) * volume * 100
        else:
            calculated_profit = (mt5_position.price_open - current_ask) * volume * 100
        
        # 更新账户余额
        self.current_balance += calculated_profit
        
        # 记录交易历史
        trade_record = {
            'close_time': datetime.now(UTC_PLUS_2),  # 使用UTC+2时区
            'direction': direction,
            'entry_price': mt5_position.price_open,
            'exit_price': exit_price,  # 使用正确的平仓价格
            'profit': calculated_profit,
            'reason': reason,
            'confidence': 0  # 从MT5持仓获取不到置信度
        }
        self.trade_history.append(trade_record)
        
        print(f"✅ {direction}仓位已{reason}平仓:")
        print(f"   盈亏: ${calculated_profit:.2f}")
        print(f"   当前余额: ${self.current_balance:.2f}")
        
        return True
    
    def get_account_info(self):
        """获取账户信息"""
        # 检查MT5连接状态，如果未连接则初始化
        if mt5.account_info() is None:
            if not mt5.initialize():
                print(f"❌ MT5初始化失败: {mt5.last_error()}")
                return None
        
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage
        }
    
    def display_status(self):
        """显示当前状态"""
        # 同步MT5中的实际持仓状态
        mt5_positions = mt5.positions_get(symbol=self.symbol)
        if mt5_positions:
            # 更新本地持仓列表以匹配MT5中的实际持仓
            mt5_position_tickets = {pos.ticket for pos in mt5_positions}
            local_tickets = {pos['ticket'] for pos in self.active_positions}
            
            # 移除本地列表中已不存在的持仓
            self.active_positions = [pos for pos in self.active_positions if pos['ticket'] in mt5_position_tickets]
            
            # 添加MT5中存在但本地列表中没有的持仓
            for pos in mt5_positions:
                if pos.ticket not in local_tickets:
                    # 添加新持仓到本地列表
                    position_info = {
                        'ticket': pos.ticket,
                        'direction': '做多' if pos.type == mt5.POSITION_TYPE_BUY else '做空',
                        'entry_price': pos.price_open,
                        'volume': pos.volume,
                        'entry_time': datetime.now(UTC_PLUS_2),
                        'take_profit': pos.tp,
                        'stop_loss': pos.sl,
                        'lot_size': pos.volume,
                        'confidence': 0,
                        'expected_close_time': datetime.now(UTC_PLUS_2) + timedelta(minutes=30)  # 默认30分钟
                    }
                    self.active_positions.append(position_info)
        
        print(f"\n📋 当前状态:")
        print(f"   账户余额: ${self.current_balance:.2f}")
        print(f"   活跃持仓数: {len(self.active_positions)}")
        
        if self.active_positions:
            print("   持仓详情:")
            for i, pos in enumerate(self.active_positions):
                # 检查MT5连接状态，如果未连接则初始化
                if mt5.account_info() is None:
                    if not mt5.initialize():
                        print(f"     #{i+1} {pos['direction']} 无法获取价格信息")
                        continue
                
                current_ask, current_bid = self.get_current_price()
                if current_ask is not None and current_bid is not None:
                    if pos['direction'] == '做多':
                        current_price = current_bid
                        # 检查是否所有必要的字段都存在
                        if 'entry_price' in pos and 'lot_size' in pos:
                            unrealized_pnl = (current_price - pos['entry_price']) * pos['lot_size'] * 100
                        elif 'entry_price' in pos and 'volume' in pos:  # 处理现有持仓
                            unrealized_pnl = (current_price - pos['entry_price']) * pos['volume'] * 100
                        else:
                            unrealized_pnl = 0
                    else:
                        current_price = current_ask
                        # 检查是否所有必要的字段都存在
                        if 'entry_price' in pos and 'lot_size' in pos:
                            unrealized_pnl = (pos['entry_price'] - current_price) * pos['lot_size'] * 100
                        elif 'entry_price' in pos and 'volume' in pos:  # 处理现有持仓
                            unrealized_pnl = (pos['entry_price'] - current_price) * pos['volume'] * 100
                        else:
                            unrealized_pnl = 0
                    
                    remaining_time = ""
                    if 'expected_close_time' in pos:
                        time_left = pos['expected_close_time'] - datetime.now(UTC_PLUS_2)  # 使用UTC+2时区
                        if time_left.total_seconds() > 0:
                            minutes_left = int(time_left.total_seconds() // 60)
                            remaining_time = f" 剩余{minutes_left}分钟"
                        else:
                            remaining_time = " 即将到期"
                    
                    # 检查置信度字段是否存在
                    confidence = pos.get('confidence', 0)
                    print(f"     #{i+1} {pos['direction']} {unrealized_pnl:+.2f}$ 方向:{pos['direction']}{remaining_time} 置信度:{confidence}%")
                else:
                    print(f"     #{i+1} {pos['direction']} 暂无法计算盈亏")
        else:
            print("   持仓详情: 无")
        
        print(f"   交易历史: {len(self.trade_history)}笔")
        if self.trade_history:
            recent_trades = self.trade_history[-3:]  # 显示最近3笔交易
            for trade in recent_trades:
                print(f"     {trade['close_time'].strftime('%H:%M:%S')} {trade['direction']} {trade['profit']:+.2f}$ ({trade['reason']})")
    
    def run_trading_cycle(self):
        """运行交易循环"""
        while not self.stop_event.is_set():
            try:
                # 检查是否有新信号
                try:
                    new_signal = self.load_latest_signal()
                    if new_signal:
                        # 根据信号下单
                        self.place_order(new_signal)
                except Exception as e:
                    print(f"❌ 从m1_data_analyzer_and_trader.py获取交易信号失败: {str(e)}")
                    # 等待一段时间后重试
                    time.sleep(30)
                    continue
                
                # 检查平仓条件
                self.check_close_conditions()
                
                # 显示状态
                self.display_status()
                
                # 等待30秒
                for _ in range(30):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ 交易循环出错: {str(e)}")
                time.sleep(5)  # 出错后稍等再继续
    
    def start_trading(self):
        """启动实时交易"""
        if self.running:
            print("⚠️ 交易系统已在运行中")
            return
        
        self.running = True
        self.stop_event.clear()
        
        print("🚀 启动实时交易系统...")
        print("💡 提示: 按 Ctrl+C 可停止交易")
        
        # 启动交易线程
        self.trade_thread = Thread(target=self.run_trading_cycle)
        self.trade_thread.daemon = True
        self.trade_thread.start()
        
        try:
            # 主线程等待
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 用户中断，正在停止交易系统...")
            self.stop_trading()
    
    def stop_trading(self):
        """停止实时交易"""
        print("🛑 正在停止交易系统...")
        self.running = False
        self.stop_event.set()
        
        if self.trade_thread and self.trade_thread.is_alive():
            self.trade_thread.join(timeout=5)  # 最多等待5秒
        
        # 关闭MT5连接
        mt5.shutdown()
        print("✅ 交易系统已停止")


def main():
    """主函数"""
    trader = XAUUSDM1RealTimeTrader()
    if trader:
        trader.start_trading()


if __name__ == "__main__":
    main()