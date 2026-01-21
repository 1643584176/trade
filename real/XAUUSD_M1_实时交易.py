"""
XAUUSD M1 实时交易系统
基于m1_data_analyzer_and_trainer.py生成的交易信号进行实时交易
每30秒查询一次交易信号并执行相应操作
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

# 设置UTC+2时区
UTC_PLUS_2 = pytz.timezone('Etc/GMT-2')  # 注意：GMT-2 实际上是UTC+2


warnings.filterwarnings('ignore')

# 🔥 FTMO交易参数
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）
MAX_DAILY_LOSS_PERCENTAGE = 4.5  # 最大日亏损比例4.5%
INITIAL_ACCOUNT_BALANCE = 10000  # FTMO挑战账户初始资金1万美元


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
        # 获取今日0点0分的UTC时间戳
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # 获取今天0点时的余额作为今日初始余额
        # 通过获取历史交易记录来估算今日初始余额
        self.daily_start_balance = self.get_today_initial_balance(today_start)
        self.startup_balance = self.daily_start_balance  # 程序启动时的账户余额（今日初始余额）
        self.current_balance = self.startup_balance
        self.active_positions = []  # 活跃持仓
        self.trade_history = []  # 交易历史
        self.magic_number = 234000  # 魔法数字
        
        # 添加冷却时间属性 - 用于在止盈止损后暂停交易
        self.cooling_period_until = None  # 冷却期截止时间
        
        # FTMO风控参数
        self.daily_loss_limit_percentage = MAX_DAILY_LOSS_PERCENTAGE  # 每日最大亏损百分比
        self.daily_loss_limit_amount = self.daily_start_balance * (MAX_DAILY_LOSS_PERCENTAGE / 100)  # 每日最大亏损金额
        self.max_drawdown_percentage = MAX_DAILY_LOSS_PERCENTAGE  # 最大回撤百分比
        
        # 控制变量
        self.running = False
        self.trade_thread = None
        self.stop_event = Event()
        self.trading_disabled = False  # 交易禁用标志
        
        # 加载AI模型和scaler
        self.load_ai_models()
        
        # 信号文件路径
        self.signals_dir = "."  # 当前目录
        self.last_signal_time = None
        self.current_signal = None
        
        # 启动时检查是否有持仓
        self.check_existing_positions()
        
        print("=" * 80)
        print("💰 XAUUSD M1 实时交易系统")
        print(f"📊 程序启动时账户资金: ${self.startup_balance:.2f}")
        print(f"📊 今日初始账户资金: ${self.daily_start_balance:.2f}")
        print(f"📈 固定手数: {self.fixed_lot_size}手")
        print(f"🔮 魔法数字: {self.magic_number}")
        print(f"💸 每日最大亏损限制: {self.daily_loss_limit_percentage}% (${self.daily_loss_limit_amount:.2f})")
        print("🔄 系统将在每30秒检查一次交易信号")
        print("🔄 请确保m1_data_analyzer_and_trainer.py已生成交易信号")
        
        # # 查询并显示最新一笔交易订单
        # self.get_latest_trade()
        #
        # # 查询并显示今天交易记录和当前持仓
        # self.get_today_deals_and_positions()
        
        # 从当日历史统计获取数据
        self.update_daily_statistics()
        
        print("=" * 80)
    
    def get_today_initial_balance(self, today_start):
        """获取今日初始余额（今天交易开始前的余额）"""        
        # 尝试获取今天0点时的账户余额
        # 通过获取历史订单和交易来推算今日初始余额
        try:
            # 获取今天的交易历史
            from_time = today_start
            to_time = datetime.now()
            
            # 将时间转换为时间戳格式，确保获取历史订单的准确性
            from_timestamp = int(from_time.timestamp())
            to_timestamp = int(to_time.timestamp())
            

            # 获取今天的交易历史
            history_deals = mt5.history_deals_get(from_time, to_time)
            

            
            # 如果上面的方法没有返回结果，尝试使用时间戳
            if history_deals is None or len(history_deals) == 0:
                print("⚠️ 使用日期时间范围查询失败，尝试使用时间戳")
                history_deals = mt5.history_deals_get(from_timestamp, to_timestamp)
                print(f"📊 使用时间戳查询结果: {history_deals}")
            
            # 如果还是没有结果，尝试扩大搜索范围到过去24小时
            if history_deals is None or len(history_deals) == 0:
                print("⚠️ 未找到今天的交易记录，尝试查询过去24小时的交易记录")
                yesterday_start = to_time - timedelta(hours=24)
                history_deals = mt5.history_deals_get(yesterday_start, to_time)

                if history_deals is None or len(history_deals) == 0:
                    print("⚠️ 仍未找到交易记录，可能今天确实没有交易")
                else:
                    # 过滤出今天的交易记录
                    today_deals = []
                    print(f"📅 今天的日期: {today_start.date()}")
                    for deal in history_deals:
                        # 正确处理MT5时间戳，先转为UTC时间，再转为UTC+2
                        deal_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)
                        print(f"   检查交易时间: {deal_time.strftime('%Y-%m-%d %H:%M:%S')}, 今天日期: {today_start.date()}, 交易日期: {deal_time.date()}")
                        if deal_time.date() == today_start.date():
                            today_deals.append(deal)
                            print(f"   ✅ 发现今天的交易: ID {deal.ticket}, 时间 {deal_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    history_deals = today_deals
                    print(f"📊 从过去24小时中过滤出今天的交易记录数量: {len(history_deals)}")
                    if len(history_deals) > 0:
                        print(f"✅ 成功找到 {len(history_deals)} 笔今天的交易记录")
            
            # 如果还是没有找到今天的交易记录，尝试使用更广泛的查询方法
            if history_deals is None or len(history_deals) == 0:
                print("🔍 尝试使用更广泛的查询方法获取今天的交易记录")
                # 尝试获取所有交易记录，然后手动过滤
                # 首先检查历史交易总数
                total_history_deals = mt5.history_deals_total(0, 10000)  # 获取最多10000个交易记录
                print(f"📊 总历史交易数: {total_history_deals}")
                
                if total_history_deals > 0:
                    # 获取最近的所有交易记录
                    all_deals = mt5.history_deals_get(0, min(total_history_deals, 10000))
                    if all_deals:
                        today_deals = []
                        print(f"📅 今天的日期: {today_start.date()}")
                        for deal in all_deals:
                            # 将时间戳转换为UTC+2时区
                            deal_time = pd.to_datetime(deal.time, unit='s', utc=True).tz_convert(UTC_PLUS_2)
                            print(f"   检查交易时间: {deal_time.strftime('%Y-%m-%d %H:%M:%S')}, 今天日期: {today_start.date()}, 交易日期: {deal_time.date()}")
                            if deal_time.date() == today_start.date():
                                today_deals.append(deal)
                                # print(f"   ✅ 发现今天的交易: ID {deal.ticket}, 时间 {deal_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        history_deals = today_deals
                        if len(today_deals) > 0:
                            print(f"✅ 从所有历史记录中找到 {len(today_deals)} 笔今天的交易记录")
                        else:
                            print("⚠️ 从所有历史记录中也未找到今天的交易记录")
                    else:
                        print("⚠️ 获取所有历史记录失败")
            
            # 再尝试一种方法：使用交易者账号查询
            if history_deals is None or len(history_deals) == 0:
                print("🔍 尝试使用交易者账号查询今天的交易记录")
                # 使用当前时间作为起点，向前查找今天的交易
                today_midnight_timestamp = from_timestamp
                end_of_day_timestamp = to_timestamp
                
                # 使用历史订单查询作为替代方法
                try:
                    history_orders = mt5.history_orders_get(today_start, to_time)
                    print(f"📊 历史订单查询结果: {history_orders}")
                    
                    # 如果历史订单中有，再检查相关的交易
                    if history_orders:
                        today_order_deals = []
                        print(f"📅 今天的日期: {today_start.date()}")
                        for order in history_orders:
                            # 尝试获取与订单关联的交易
                            order_deals = mt5.history_deals_get(ticket=order.ticket)
                            if order_deals:
                                for deal in order_deals:
                                    # 将时间戳转换为UTC+2时区
                                    deal_time = pd.to_datetime(deal.time, unit='s', utc=True).tz_convert(UTC_PLUS_2)
                                    print(f"   检查订单关联交易时间: {deal_time.strftime('%Y-%m-%d %H:%M:%S')}, 今天日期: {today_start.date()}, 交易日期: {deal_time.date()}")
                                    if deal_time.date() == today_start.date():
                                        today_order_deals.append(deal)
                                        print(f"   ✅ 发现今天的订单关联交易: ID {deal.ticket}, 时间 {deal_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # 合并到主列表
                        if today_order_deals:
                            if history_deals is None:
                                history_deals = today_order_deals
                            else:
                                history_deals.extend(today_order_deals)
                        
                        print(f"✅ 从历史订单关联的交易中找到了 {len(today_order_deals) if today_order_deals else 0} 笔交易")
                except Exception as e:
                    print(f"⚠️ 历史订单查询出错: {str(e)}")
            
            # 获取当前账户信息
            account_info = mt5.account_info()
            if account_info is None:
                print("⚠️ 无法获取账户信息，使用当前账户余额作为今日初始余额")
                return self.startup_balance
            
            # 如果今天没有交易记录，则今日初始余额就是当前余额（减去持仓的浮动盈亏）
            if history_deals is None or len(history_deals) == 0:
                print("📊 今天没有交易记录，使用当前余额减去持仓浮动盈亏")
                # 如果没有持仓，则今日初始余额就是当前余额
                positions = mt5.positions_get(symbol=self.symbol)
                if positions is None or len(positions) == 0:

                    return account_info.balance
                else:
                    # 有持仓，需要计算持仓的浮动盈亏来反推初始余额
                    total_float_pnl = 0
                    print(f"📊 检测到 {len(positions)} 个持仓，计算浮动盈亏...")
                    for pos in positions:
                        # 计算每个持仓的浮动盈亏
                        print(f"   持仓单号: {pos.ticket}, 方向: {'做多' if pos.type == mt5.POSITION_TYPE_BUY else '做空'}, 开仓价: {pos.price_open}, 手数: {pos.volume}")
                        if pos.type == mt5.POSITION_TYPE_BUY:  # 做多
                            current_tick = mt5.symbol_info_tick(self.symbol)
                            if current_tick:
                                float_pnl = (current_tick.bid - pos.price_open) * pos.volume * 100
                                total_float_pnl += float_pnl
                                print(f"   做多持仓浮动盈亏: ({current_tick.bid:.2f} - {pos.price_open:.2f}) * {pos.volume} * 100 = {float_pnl:.2f}")
                        else:  # 做空
                            current_tick = mt5.symbol_info_tick(self.symbol)
                            if current_tick:
                                float_pnl = (pos.price_open - current_tick.ask) * pos.volume * 100
                                total_float_pnl += float_pnl
                                print(f"   做空持仓浮动盈亏: ({pos.price_open:.2f} - {current_tick.ask:.2f}) * {pos.volume} * 100 = {float_pnl:.2f}")
                    
                    print(f"📊 当前持仓总浮动盈亏: {total_float_pnl:.2f}")
                    # 今日初始余额 = 当前余额 - 持仓浮动盈亏
                    initial_balance = account_info.balance - total_float_pnl
                    print(f"📊 计算出的今日初始余额: 当前余额 {account_info.balance:.2f} - 持仓浮动盈亏 {total_float_pnl:.2f} = {initial_balance:.2f}")
                    return initial_balance
            else:
                        # 如果有交易记录，从历史交易中计算总盈亏来反推初始余额

                total_daily_pnl = 0
                for deal in history_deals:
                    total_daily_pnl += deal.profit
                # 今日初始余额 = 当前余额 - 今日盈亏
                initial_balance = account_info.balance - total_daily_pnl
                return initial_balance
        except Exception as e:
            print(f"⚠️ 获取今日初始余额时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            # 出错时，返回当前账户余额作为备用
            account_info = mt5.account_info()
            if account_info:
                return account_info.balance
            else:
                return self.startup_balance
    
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
    
    def get_latest_m1_data(self, count=250):
        """获取最新的M1数据"""
        print(f"\n📡 获取XAUUSD最新M1数据...")
        
        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return None
        
        # 检查交易品种
        symbol = "XAUUSD"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ 品种 {symbol} 不可用")
            mt5.shutdown()
            return None

        if not symbol_info.visible:
            print(f"✅ 启用品种 {symbol}...")
            if not mt5.symbol_select(symbol, True):
                print(f"❌ 启用品种失败")
                mt5.shutdown()
                return None

        # 获取最近的M1数据
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
        
        if rates is None or len(rates) == 0:
            print(f"❌ 未获取到最新的M1数据")
            mt5.shutdown()
            return None
        
        # 转换为DataFrame
        data = pd.DataFrame(rates)
        
        # 转换时间戳
        data['timestamp'] = pd.to_datetime(data['time'], unit='s')
        # 转换为UTC+2时区
        data['timestamp'] = data['timestamp'] + pd.Timedelta(hours=2)
        
        # 重命名列
        data = data.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread'
        })
        
        # 选择需要的列
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]
        
        print(f"✅ 获取到 {len(data)} 根M1 K线数据")
        
        # 断开MT5连接
        mt5.shutdown()
        
        return data
    
    def get_latest_trade(self):
        """查询最新的交易订单详情"""
        print(f"\n📋 查询最新一笔交易订单详情...")
        
        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return None
        
        try:
            # 获取所有的历史交易记录
            total_deals = mt5.history_deals_total(0, 100000)  # 获取最多100000条记录
            print(f"📊 总历史交易数: {total_deals}")
            
            if total_deals == 0:
                print("⚠️ 没有任何历史交易记录")
                mt5.shutdown()
                return None
            
            # 获取所有历史交易
            all_deals = mt5.history_deals_get(0, min(total_deals, 10000))  # 限制获取数量
            if all_deals is None:
                print("❌ 获取历史交易记录失败")
                mt5.shutdown()
                return None
            
            # 按时间排序（最新的在前）
            sorted_deals = sorted(all_deals, key=lambda x: x.time, reverse=True)
            
            # 取最新的一笔交易
            latest_deal = sorted_deals[0] if sorted_deals else None
            
            if latest_deal is None:
                print("⚠️ 未找到任何交易记录")
                return None
            
            # 将时间戳转换为UTC+2时区
            deal_time = pd.to_datetime(latest_deal.time, unit='s', utc=True).tz_convert(UTC_PLUS_2)
            
            print(f"✅ 最新交易订单详情:")
            print("-" * 50)
            print(f"交易号: {latest_deal.ticket}")
            print(f"交易时间(UTC+2): {deal_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"交易品种: {latest_deal.symbol}")
            print(f"交易类型: {'买入' if latest_deal.type == 0 else '卖出' if latest_deal.type == 1 else '其他'}")
            print(f"交易方向: {'开仓' if latest_deal.entry == 0 else '平仓' if latest_deal.entry == 1 else '其他'}")
            print(f"交易手数: {latest_deal.volume}")
            print(f"盈亏金额: {latest_deal.profit}")
            print(f"佣金: {latest_deal.commission}")
            print(f"成交价格: {latest_deal.price}")
            print(f"备注: {latest_deal.comment}")
            print("-" * 50)
            
            # 存储交易信息
            trade_info = {
                'ticket': latest_deal.ticket,
                'time': deal_time,
                'symbol': latest_deal.symbol,
                'type': latest_deal.type,
                'entry': latest_deal.entry,  # 添加entry字段
                'volume': latest_deal.volume,
                'profit': latest_deal.profit,
                'commission': latest_deal.commission,
                'price': latest_deal.price,
                'comment': latest_deal.comment
            }
            
            return trade_info
            
        except Exception as e:
            print(f"❌ 查询最新交易订单时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 断开MT5连接
            mt5.shutdown()
    
    def get_today_deals_and_positions(self):
        """查询今天的交易记录和当前持仓"""
        print(f"\n📊 查询今天的交易记录和当前持仓...")
        
        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return None
        
        try:
            local_tz = timezone(timedelta(hours=10))  # 东八区（北京/上海时区）

            # 2. 获取带本地时区的当前时间
            now = datetime.now(local_tz)


            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # 将时间转换为时间戳
            from_timestamp = int(today_start.timestamp())
            to_timestamp = int(now.timestamp())  # 可以设置为当前时间，或者留空表示查询到当前
            
            print(f"🔍 查询从 {today_start.strftime('%Y-%m-%d %H:%M:%S')} 到当前时间的交易记录")
            
            # 获取今天的交易历史
            history_deals = mt5.history_deals_get(today_start)
            
            if history_deals is None or len(history_deals) == 0:
                print("⚠️ 今天没有交易记录")
            else:
                print(f"📊 检测到今天有 {len(history_deals)} 笔交易记录")
                total_daily_pnl = 0
                open_deals = []  # 开仓记录
                close_deals = []  # 平仓记录
                
                for deal in history_deals:
                    # 将时间戳转换为UTC+2时区显示
                    try:
                        deal_time = pd.to_datetime(deal.time, unit='s', utc=True).tz_convert(UTC_PLUS_2)
                        print(f"   交易ID: {deal.ticket}, 类型: {'开仓' if deal.entry == 0 else '平仓' if deal.entry == 1 else '其他'}, 盈亏: {deal.profit:.2f}, 时间: {deal_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # 分类记录
                        if deal.entry == 0:  # 开仓
                            open_deals.append(deal)
                        elif deal.entry == 1:  # 平仓
                            close_deals.append(deal)
                            total_daily_pnl += deal.profit  # 累加平仓盈亏
                    except:
                        print(f"   交易ID: {deal.ticket}, 盈亏: {deal.profit:.2f}, 时间戳: {deal.time}")
                
                print(f"📊 今日平仓盈亏总计: {total_daily_pnl:.2f}")
                print(f"📊 今日开仓记录数: {len(open_deals)}")
                print(f"📊 今日平仓记录数: {len(close_deals)}")
            
            # 获取当前持仓
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None or len(positions) == 0:
                print("📊 当前没有持仓")
            else:
                print(f"📊 当前有 {len(positions)} 个持仓")
                total_float_pnl = 0
                
                for pos in positions:
                    # 获取当前市场价格
                    current_tick = mt5.symbol_info_tick(self.symbol)
                    if current_tick:
                        if pos.type == mt5.POSITION_TYPE_BUY:  # 做多
                            float_pnl = (current_tick.bid - pos.price_open) * pos.volume * 100
                            print(f"   持仓ID: {pos.ticket}, 方向: 做多, 开仓价: {pos.price_open}, 当前价: {current_tick.bid}, 浮动盈亏: {float_pnl:.2f}")
                        else:  # 做空
                            float_pnl = (pos.price_open - current_tick.ask) * pos.volume * 100
                            print(f"   持仓ID: {pos.ticket}, 方向: 做空, 开仓价: {pos.price_open}, 当前价: {current_tick.ask}, 浮动盈亏: {float_pnl:.2f}")
                        
                        total_float_pnl += float_pnl
                
                print(f"📊 当前持仓总浮动盈亏: {total_float_pnl:.2f}")
            
            return {
                'history_deals': history_deals,
                'positions': positions,
                'total_daily_pnl': total_daily_pnl if 'total_daily_pnl' in locals() else 0,
                'total_float_pnl': total_float_pnl if 'total_float_pnl' in locals() else 0
            }
            
        except Exception as e:
            print(f"❌ 查询今天交易记录和持仓时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 断开MT5连接
            mt5.shutdown()
    

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

                return latest_signal
            
            self.last_signal_time = signal_time
            self.current_signal = latest_signal
            
            # 获取反转概率信息
            reversal_probability = latest_signal.get('趋势反转概率(%)', 0)
                    
            # 检查信号是否有效
            signal_valid = latest_signal.get('信号有效性', '') == '✅ 有效'
                    
            # 信号已根据反转概率调整方向
            if signal_valid:
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
    
    def load_ai_models(self):
        """加载AI模型和scaler"""
        try:
            # 动态导入m1_data_analyzer_and_trainer模块以获取模型
            import os
            import joblib
            
            # 查找最新的模型文件
            model_dir = "trading_ai_models"
            if os.path.exists(model_dir):
                # 获取目录中所有pkl文件
                pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                
                if pkl_files:
                    # 按时间排序，获取最新的模型文件
                    latest_scaler = None
                    for file in sorted(pkl_files):
                        if file.startswith('scaler_'):
                            latest_scaler = file
                        
                    if latest_scaler:
                        scaler_path = os.path.join(model_dir, latest_scaler)
                        self.scaler = joblib.load(scaler_path)
                        print(f"✅ 成功加载scaler模型: {latest_scaler}")
                    else:
                        # 如果没有找到scaler，创建一个新的StandardScaler
                        from sklearn.preprocessing import StandardScaler
                        self.scaler = StandardScaler()
                        print("⚠️  未找到scaler模型，使用默认StandardScaler")
                else:
                    # 如果没有找到任何pkl文件，创建一个新的StandardScaler
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                    print("⚠️  未找到任何模型文件，使用默认StandardScaler")
            else:
                # 如果模型目录不存在，创建一个新的StandardScaler
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                print("⚠️  模型目录不存在，使用默认StandardScaler")
        except Exception as e:
            print(f"❌ 加载AI模型失败: {str(e)}")
            # 出错时创建一个新的StandardScaler
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            print("⚠️  使用默认StandardScaler")
    
    def check_loss_limits(self):
        """检查是否超过亏损限制"""
        # 更新每日统计数据
        self.update_daily_statistics()
        
        # 从当日统计中获取今日盈亏数据
        today_pnl = getattr(self, 'daily_pnl', 0)  # 今日盈亏
        current_balance = getattr(self, 'current_balance_from_stats', self.current_balance)  # 当前余额
        daily_start_balance = getattr(self, 'daily_start_balance_from_stats', self.daily_start_balance)  # 今日初始余额
        
        # 计算当前账户余额相对于今日初始余额的变化
        current_pnl = today_pnl  # 使用从统计中获取的今日盈亏
        if daily_start_balance != 0:
            loss_percentage = abs(current_pnl) / daily_start_balance * 100
        else:
            loss_percentage = 0
        
        # 检查是否超过每日亏损限制（只有亏损时才检查，盈利不管）
        if current_pnl < 0 and loss_percentage >= self.daily_loss_limit_percentage:
            print(f"🚨 严重警告: 当前亏损 {loss_percentage:.2f}% 已达到或超过每日最大亏损限制 {self.daily_loss_limit_percentage}%")
            print("🛑 为保护账户，系统将暂停交易并退出")
            self.trading_disabled = True
            self.stop_trading()
            return True
        
        # 检查浮亏是否超过限制（只考虑浮亏，盈利不管）
        floating_loss_exceeded = False
        if self.active_positions:
            total_floating_loss = 0
            for pos in self.active_positions:
                if 'entry_price' in pos and 'lot_size' in pos:
                    if pos['direction'] == '做多':
                        current_ask, current_bid = self.get_current_price()
                        if current_bid is not None:
                            unrealized_pnl = (current_bid - pos['entry_price']) * pos['lot_size'] * 100
                            if unrealized_pnl < 0:  # 只统计浮亏
                                total_floating_loss += abs(unrealized_pnl)
                    else:  # 做空
                        current_ask, current_bid = self.get_current_price()
                        if current_ask is not None:
                            unrealized_pnl = (pos['entry_price'] - current_ask) * pos['lot_size'] * 100
                            if unrealized_pnl < 0:  # 只统计浮亏
                                total_floating_loss += abs(unrealized_pnl)
            
            # 检查浮亏占今日初始本金的比例（只有浮亏时才检查，盈利不管）
            if daily_start_balance > 0:
                floating_loss_percentage = (total_floating_loss / daily_start_balance) * 100
                if floating_loss_percentage >= self.daily_loss_limit_percentage:
                    print(f"🚨 警告: 当前浮亏 ${total_floating_loss:.2f} 占今日初始本金的 {floating_loss_percentage:.2f}% 已达到或超过限制 {self.daily_loss_limit_percentage}%")
                    print("⚠️  系统将暂停新交易，但会继续管理现有持仓")
                    self.trading_disabled = True
                    floating_loss_exceeded = True
        
        return floating_loss_exceeded

    def update_daily_statistics(self):
        """更新每日统计数据，从历史统计中获取今日盈亏、当前余额等信息"""
        try:
            # 初始化MT5连接
            if not mt5.initialize():
                print(f"MT5初始化失败，错误：{mt5.last_error()}")
                return

            # 获取账户信息以获取当前余额
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ 无法获取账户信息")
                mt5.shutdown()
                return

            current_balance = account_info.balance
            
            # 获取今天的日期范围
            today = datetime.now(UTC_PLUS_2).date()
            start_time = datetime.combine(today, datetime.min.time()).replace(tzinfo=UTC_PLUS_2)
            end_time = datetime.combine(today, datetime.max.time()).replace(tzinfo=UTC_PLUS_2)
            
            # 转换为时间戳
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            # 查询交易记录
            all_deals = mt5.history_deals_get(start_ts, end_ts)
            
            # 过滤XAUUSD的交易记录，只包含盈亏不为0的记录
            daily_profit_loss = 0  # 今日盈亏
            daily_losses = []  # 今日亏损记录（绝对值）

            if all_deals and len(all_deals) > 0:
                for deal in all_deals:
                    # 过滤XAUUSD相关品种
                    symbol = deal.symbol.upper()
                    if "XAUUSD" in symbol and deal.profit != 0:  # 只添加盈亏不为0的记录
                        # 累加今日盈亏
                        daily_profit_loss += deal.profit

                        # 记录亏损（绝对值）
                        if deal.profit < 0:
                            daily_losses.append(abs(deal.profit))
            
            # 计算今日初始余额（当前余额 - 今日盈亏）
            daily_start_balance = current_balance - daily_profit_loss
            
            # 计算今日最大亏损
            max_daily_loss = max(daily_losses) if daily_losses else 0
            
            # 更新实例变量
            self.current_balance_from_stats = current_balance
            self.daily_start_balance_from_stats = daily_start_balance
            self.daily_pnl = daily_profit_loss
            self.max_daily_loss = max_daily_loss
            
            # 断开MT5连接
            mt5.shutdown()
            
            # print(f"📊 从当日统计获取数据 - 当前余额: {current_balance:.2f}USD, 今日盈亏: {daily_profit_loss:.2f}USD, 今日初始余额: {daily_start_balance:.2f}USD")
            #
        except Exception as e:
            print(f"❌ 更新每日统计数据时出错: {str(e)}")
            # 出错时使用原始方法更新数据
            try:
                mt5.shutdown()
            except:
                pass
    def place_order(self, signal):
        """根据信号下单"""
        # 检查是否因亏损限制而禁用交易
        if self.trading_disabled:
            print("⚠️  由于亏损限制，新交易已被禁用，无法下单")
            return False

        # 检查是否处于冷却期
        if self.cooling_period_until and datetime.now(UTC_PLUS_2) < self.cooling_period_until:
            print(f"❄️  处于冷却期，直到 {self.cooling_period_until.strftime('%Y-%m-%d %H:%M:%S')}，暂停新订单")
            return False
        
        # 清除冷却期（如果过了冷却期，可以下单）
        if self.cooling_period_until and datetime.now(UTC_PLUS_2) >= self.cooling_period_until:
            self.cooling_period_until = None
            print("✅ 冷却期结束，恢复交易")
        
        # 再次检查亏损限制
        if self.check_loss_limits():
            print("⚠️  由于亏损限制，新交易已被禁用，无法下单")
            return False
            
        direction = signal.get('实际方向', '')
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
                # print(f"⚠️ MT5中已有{existing_direction}持仓，无法开立同方向新仓")
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
                close_success = self.close_position_directly(existing_position, "", 0)  # 临时利润为0，实际盈亏会在close_position中计算
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
            print(f"❌ 未知的交易方向: '{direction}'")
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
                
                # 检查该持仓是否因止盈或止损被平仓
                # 获取最新的交易历史来判断平仓原因
                try:
                    # 获取最近的交易记录
                    from_time = datetime.now() - timedelta(minutes=10)  # 检查最近10分钟的交易
                    recent_deals = mt5.history_deals_get(from_time, datetime.now())
                    
                    # 查找与该持仓相关的平仓交易
                    close_reason = "时间到期"  # 默认原因
                    for deal in recent_deals or []:
                        if deal.position_id == position['ticket']:  # 找到相关的平仓记录
                            # 根据成交价格判断平仓原因
                            if abs(deal.price - position['take_profit']) < 0.1:  # 接近止盈价
                                close_reason = "止盈"
                            elif abs(deal.price - position['stop_loss']) < 0.1:  # 接近止损价
                                close_reason = "止损"
                            else:
                                close_reason = "时间到期"
                            break
                    
                    # 如果是止盈或止损平仓，则设置5分钟冷却时间
                    if close_reason in ['止盈', '止损']:
                        self.cooling_period_until = datetime.now(UTC_PLUS_2) + timedelta(minutes=5)
                        print(f"⏰ 检测到{close_reason}平仓，进入5分钟冷却期，直到 {self.cooling_period_until.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    print(f"⚠️ 检查平仓原因时出错: {str(e)}")
                    # 出错时仍按正常流程处理
                    close_reason = "未知"
                
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
            
            # 检查持仓是否包含必要的字段
            has_tp_sl = 'take_profit' in position and 'stop_loss' in position
            has_expected_close_time = 'expected_close_time' in position
            has_lot_size = 'lot_size' in position
            
            # 检查是否达到预计持仓时长（时间到期平仓）
            # 如果剩余时间小于等于0，则执行平仓
            should_close = False
            close_reason = ""
            
            if not should_close and has_expected_close_time:
                time_left = position['expected_close_time'] - datetime.now(UTC_PLUS_2)  # 使用UTC+2时区
                if time_left.total_seconds() <= 0:
                    should_close = True
                    close_reason = "时间到期"
            
            if should_close:
                # 计算盈亏
                lot_size = position.get('lot_size', position.get('volume', 0))
                if position['direction'] == '做多':
                    profit = (current_bid - position['entry_price']) * lot_size * 100
                else:
                    profit = (position['entry_price'] - current_ask) * lot_size * 100
                
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
        
        # 如果是止盈或止损平仓，则设置5分钟冷却时间
        if reason in ['止盈', '止损']:
            self.cooling_period_until = datetime.now(UTC_PLUS_2) + timedelta(minutes=5)
            print(f"⏰ 触发{reason}平仓，进入5分钟冷却期，直到 {self.cooling_period_until.strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        
        # 如果是止盈或止损平仓，则设置5分钟冷却时间
        if reason in ['止盈', '止损']:
            self.cooling_period_until = datetime.now(UTC_PLUS_2) + timedelta(minutes=5)
            print(f"⏰ 触发{reason}平仓，进入5分钟冷却期，直到 {self.cooling_period_until.strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        
        # 更新每日统计数据
        self.update_daily_statistics()
        
        # 使用从统计中获取的数据
        current_balance = getattr(self, 'current_balance_from_stats', self.current_balance)
        daily_start_balance = getattr(self, 'daily_start_balance_from_stats', self.daily_start_balance)
        daily_pnl = getattr(self, 'daily_pnl', self.current_balance - self.daily_start_balance)
        
        # 重新计算百分比
        if daily_start_balance != 0:
            pnl_percentage = (daily_pnl / daily_start_balance) * 100
        else:
            pnl_percentage = 0

        print(f"当前账户余额: ${current_balance:.2f} | 程序启动时余额: ${daily_start_balance:.2f}|  今日盈亏: {daily_pnl:+.2f}$ ({pnl_percentage:+.2f}%)")

        if self.active_positions:
            for i, pos in enumerate(self.active_positions):
                # 检查MT5连接状态，如果未连接则初始化
                if mt5.account_info() is None:
                    if not mt5.initialize():
                        print(f"  持仓详情: #{i+1} {pos['direction']} 无法获取价格信息")
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
                    print(f"  #{i+1} {pos['direction']} {unrealized_pnl:+.2f}$ 方向:{pos['direction']}{remaining_time}")
                else:
                    print(f"  #{i+1} {pos['direction']} 暂无法计算盈亏")
        else:
            print("   持仓详情: 无")

    def run_trading_cycle(self):
        """运行交易循环"""
        while not self.stop_event.is_set():
            try:
                # 检查亏损限制
                if self.check_loss_limits():
                    print("🚨 由于亏损限制，交易已停止")
                    break
                
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