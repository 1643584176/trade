import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class FTMOMaxConsecutiveLossStrategy:
    def __init__(self, csv_file, initial_capital=100000):
        """
        初始化针对连续亏损优化的FTMO交易策略

        参数:
        csv_file: CSV文件路径
        initial_capital: 初始资本 (默认10万美元)
        """
        self.csv_file = csv_file
        self.initial_capital = initial_capital
        self.data = None
        self.load_data()

    def load_data(self):
        """
        加载并预处理数据
        """
        # 读取CSV文件
        self.data = pd.read_csv(self.csv_file, encoding='utf-8-sig')

        # 转换时间列格式
        self.data['时间点'] = pd.to_datetime(self.data['时间点'])

        # 按时间排序
        self.data = self.data.sort_values('时间点').reset_index(drop=True)

        # 添加星期几和小时信息
        self.data['星期几'] = self.data['时间点'].dt.day_name()
        self.data['小时'] = self.data['时间点'].dt.hour
        self.data['日期'] = self.data['时间点'].dt.date
        self.data['星期数'] = self.data['时间点'].dt.weekday  # 0=Monday, 6=Sunday
        self.data['周数'] = self.data['时间点'].dt.isocalendar().week  # 添加周数信息

        # 计算技术指标
        self.calculate_indicators()

    def calculate_indicators(self):
        """
        计算技术指标
        """
        # 计算ATR
        high_low = self.data['当前最高价'] - self.data['当前最低价']
        high_close = np.abs(self.data['当前最高价'] - self.data['当前收盘价'].shift())
        low_close = np.abs(self.data['当前最低价'] - self.data['当前收盘价'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.data['ATR'] = pd.Series(true_range).rolling(14).mean()

        # 计算RSI指标
        delta = self.data['当前收盘价'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # 计算开盘方向相关指标
        # 按日期分组，计算每日开盘价（当天第一根K线的开盘价）
        daily_open = self.data.groupby('日期')['当前开盘价'].first()
        self.data['当日开盘价'] = self.data['日期'].map(daily_open)

        # 计算当前价格相对于当日开盘价的变化
        self.data['相对开盘价变化'] = self.data['当前收盘价'] - self.data['当日开盘价']

        # 计算K线实体和影线
        self.data['实体'] = abs(self.data['当前收盘价'] - self.data['当前开盘价'])
        self.data['上影线'] = self.data['当前最高价'] - np.maximum(self.data['当前开盘价'], self.data['当前收盘价'])
        self.data['下影线'] = np.minimum(self.data['当前开盘价'], self.data['当前收盘价']) - self.data['当前最低价']
        self.data['是否阳线'] = self.data['当前收盘价'] > self.data['当前开盘价']

        # 计算短期价格动量
        self.data['1周期动量'] = self.data['当前收盘价'] - self.data['当前收盘价'].shift(1)
        self.data['3周期动量'] = self.data['当前收盘价'] - self.data['当前收盘价'].shift(3)
        self.data['5周期动量'] = self.data['当前收盘价'] - self.data['当前收盘价'].shift(5)
        self.data['10周期动量'] = self.data['当前收盘价'] - self.data['当前收盘价'].shift(10)

        # 计算价格波动率
        self.data['波动率'] = self.data['当前收盘价'].rolling(window=14).std()

        # 计算周间关联性
        self.calculate_weekly_correlations()

    def calculate_weekly_correlations(self):
        """
        计算周间关联性，特别关注星期几的特殊性
        """
        # 计算每周的收盘价
        weekly_close = self.data.groupby('周数')['当前收盘价'].last().reset_index()
        weekly_close.columns = ['周数', '周收盘价']

        # 计算前一周收盘价
        weekly_close['前一周收盘价'] = weekly_close['周收盘价'].shift(1)

        # 计算周间价格变化
        weekly_close['周间价格变化'] = weekly_close['周收盘价'] - weekly_close['前一周收盘价']
        weekly_close['周间变化方向'] = np.where(weekly_close['周间价格变化'] > 0, 1, -1)

        # 将周间关联性数据合并到原始数据中
        self.data = self.data.merge(weekly_close[['周数', '前一周收盘价', '周间价格变化', '周间变化方向']],
                                    on='周数', how='left')

        # 计算星期几的历史倾向
        weekday_analysis = self.data.groupby('星期数').agg({
            '当前增减': ['mean', 'std'],
            '相对开盘价变化': 'mean'
        }).reset_index()
        weekday_analysis.columns = ['星期数', '星期几平均涨跌', '星期几涨跌标准差', '相对开盘价平均变化']
        self.data = self.data.merge(weekday_analysis, on='星期数', how='left')

    def get_session_type(self, hour):
        """
        根据小时判断交易时段类型

        参数:
        hour: 小时数 (0-23)

        返回:
        session_type: 交易时段类型
        """
        if 0 <= hour <= 6:  # 亚洲时段 (GMT)
            return '亚洲'
        elif 7 <= hour <= 14:  # 欧洲时段 (GMT)
            return '欧洲'
        elif 15 <= hour <= 23:  # 美洲时段 (GMT)
            return '美洲'
        else:
            return '其他'

    def analyze_weekday_patterns(self):
        """
        分析星期几模式，找出各星期几的交易倾向
        """
        weekday_analysis = self.data.groupby('星期几').agg({
            '当前增减': ['mean', 'count', 'std'],
            '相对开盘价变化': 'mean'
        }).round(4)

        weekday_analysis.columns = ['平均涨跌', '交易次数', '涨跌标准差', '相对开盘价平均变化']
        weekday_analysis['交易倾向'] = np.where(weekday_analysis['平均涨跌'] > 0, 1, -1)
        return weekday_analysis

    def analyze_session_patterns(self):
        """
        分析不同时段的交易倾向
        """
        self.data['时段类型'] = self.data['小时'].apply(self.get_session_type)
        session_analysis = self.data.groupby('时段类型').agg({
            '当前增减': ['mean', 'count', 'std'],
            '相对开盘价变化': 'mean'
        }).round(4)

        session_analysis.columns = ['平均涨跌', '交易次数', '涨跌标准差', '相对开盘价平均变化']
        session_analysis['交易倾向'] = np.where(session_analysis['平均涨跌'] > 0, 1, -1)
        return session_analysis

    def is_reversal_pattern(self, index):
        """
        判断是否出现反转形态

        参数:
        index: 当前数据索引

        返回:
        str: 反转类型 ('bullish' 或 'bearish') 或 None
        """
        if index < 2:
            return None

        current_row = self.data.iloc[index]
        previous_row = self.data.iloc[index - 1]
        previous_2_row = self.data.iloc[index - 2]

        # 锤子线（看涨反转）
        if (current_row['下影线'] > current_row['实体'] * 2 and
                current_row['上影线'] < current_row['实体'] * 0.5 and
                current_row['是否阳线']):
            return 'bullish'

        # 上吊线（看跌反转）
        if (current_row['下影线'] > current_row['实体'] * 2 and
                current_row['上影线'] < current_row['实体'] * 0.5 and
                not current_row['是否阳线']):
            return 'bearish'

        # 吞没形态
        # 看涨吞没
        if (previous_row['当前收盘价'] < previous_row['当前开盘价'] and  # 前一根为阴线
                current_row['当前收盘价'] > current_row['当前开盘价'] and  # 当前为阳线
                current_row['当前开盘价'] < previous_row['当前收盘价'] and  # 当前开盘价低于前收盘价
                current_row['当前收盘价'] > previous_row['当前开盘价']):  # 当前收盘价高于前开盘价
            return 'bullish'

        # 看跌吞没
        if (previous_row['当前收盘价'] > previous_row['当前开盘价'] and  # 前一根为阳线
                current_row['当前收盘价'] < current_row['当前开盘价'] and  # 当前为阴线
                current_row['当前开盘价'] > previous_row['当前收盘价'] and  # 当前开盘价高于前收盘价
                current_row['当前收盘价'] < previous_row['当前开盘价']):  # 当前收盘价低于前开盘价
            return 'bearish'

        # 星形形态（早晨之星/黄昏之星）
        if index >= 3:
            previous_3_row = self.data.iloc[index - 3]
            # 早晨之星（看涨反转）
            if (previous_3_row['当前收盘价'] < previous_3_row['当前开盘价'] and  # 第一根阴线
                    abs(previous_2_row['当前收盘价'] - previous_2_row['当前开盘价']) < previous_3_row[
                        '实体'] * 0.3 and  # 第二根小实体
                    previous_2_row['当前最低价'] < min(previous_3_row['当前收盘价'], previous_3_row['当前开盘价']) and
                    current_row['当前收盘价'] > current_row['当前开盘价'] and  # 第三根阳线
                    current_row['当前收盘价'] > previous_2_row['当前最高价']):  # 收盘价高于第二根K线
                return 'bullish'

            # 黄昏之星（看跌反转）
            if (previous_3_row['当前收盘价'] > previous_3_row['当前开盘价'] and  # 第一根阳线
                    abs(previous_2_row['当前收盘价'] - previous_2_row['当前开盘价']) < previous_3_row[
                        '实体'] * 0.3 and  # 第二根小实体
                    previous_2_row['当前最高价'] > max(previous_3_row['当前收盘价'], previous_3_row['当前开盘价']) and
                    current_row['当前收盘价'] < current_row['当前开盘价'] and  # 第三根阴线
                    current_row['当前收盘价'] < previous_2_row['当前最低价']):  # 收盘价低于第二根K线
                return 'bearish'

        return None

    def generate_signals(self):
        """
        基于多因子生成交易信号，特别考虑星期几关联性
        """
        signals = []
        signal_strengths = []
        signal_reasons = []

        # 分析星期几和时段模式
        weekday_analysis = self.analyze_weekday_patterns()
        session_analysis = self.analyze_session_patterns()

        print("=== 星期几分析 ===")
        print(weekday_analysis[['平均涨跌', '交易倾向']])
        print("\n=== 时段分析 ===")
        print(session_analysis[['平均涨跌', '交易倾向']])
        print()

        for i in range(len(self.data)):
            current_row = self.data.iloc[i]
            hour = current_row['小时']
            weekday = current_row['星期几']
            weekday_num = current_row['星期数']
            week_num = current_row['周数']
            session_type = self.get_session_type(hour)

            signal = 0
            strength = 0
            reasons = []

            # 获取星期几和时段的交易倾向
            weekday_trend = weekday_analysis.loc[weekday, '交易倾向']
            session_trend = session_analysis.loc[session_type, '交易倾向']

            # 基于开盘方向判断
            open_direction = 1 if current_row['相对开盘价变化'] > 0 else -1

            # 当前价格相对于开盘价的位置
            price_change_from_open = current_row['相对开盘价变化']

            # 动量确认
            momentum_1 = 1 if current_row['1周期动量'] > 0 else -1
            momentum_3 = 1 if current_row['3周期动量'] > 0 else -1
            momentum_5 = 1 if current_row['5周期动量'] > 0 else -1
            momentum_10 = 1 if current_row['10周期动量'] > 0 else -1

            # 反转形态确认
            reversal = self.is_reversal_pattern(i)

            # 时段过滤（只在特定时段交易）
            if session_type not in ['欧洲', '美洲']:
                signals.append(0)
                signal_strengths.append(0)
                signal_reasons.append(['非交易时段'])
                continue

            # 星期几特殊性分析
            # 周五：单边行情可能性大
            if weekday_num == 4:  # Friday (0=Monday, 4=Friday)
                # 增加趋势信号强度
                if open_direction == momentum_1 == momentum_3:
                    signal = open_direction
                    strength += 3  # 增加强度
                    reasons.append(f'周五趋势强化({open_direction})')

            # 周三：反转可能性大
            elif weekday_num == 2:  # Wednesday (0=Monday, 2=Wednesday)
                # 给反转信号更高权重
                if reversal == 'bullish':
                    signal = 1
                    strength += 4  # 增加强度
                    reasons.append('周三看涨反转强化')
                elif reversal == 'bearish':
                    signal = -1
                    strength += 4  # 增加强度
                    reasons.append('周三看跌反转强化')

            # 周一周二关联性强
            elif weekday_num in [0, 1]:  # Monday or Tuesday
                # 参考前一天的收盘情况
                if not np.isnan(current_row['周间变化方向']):
                    weekly_trend = current_row['周间变化方向']
                    if weekly_trend != open_direction:
                        signal = weekly_trend
                        strength += 2
                        reasons.append(f'周间趋势修正({weekly_trend})')

            # 周四周五关联性强
            elif weekday_num in [3, 4]:  # Thursday or Friday
                # 参考周间趋势
                if not np.isnan(current_row['周间变化方向']):
                    weekly_trend = current_row['周间变化方向']
                    if weekly_trend == open_direction:
                        signal = open_direction
                        strength += 2
                        reasons.append(f'周间趋势确认({weekly_trend})')

            # 如果还没有信号，使用基础多因子策略
            if signal == 0:
                # 多因子信号生成
                # 1. 基本方向信号（开盘方向）
                if abs(price_change_from_open) > current_row['ATR'] * 0.5:
                    if open_direction == momentum_1 == momentum_3:
                        signal = open_direction
                        strength += 2
                        reasons.append(f'开盘方向{open_direction}与动量一致')

                # 2. 星期几修正信号
                if weekday_trend != open_direction:
                    # 如果星期几倾向与开盘方向相反，则修正信号
                    signal = weekday_trend
                    strength += 1
                    reasons.append(f'星期几倾向修正({weekday_trend})')

                # 3. 时段倾向修正信号
                if session_trend != open_direction:
                    # 如果时段倾向与开盘方向相反，则修正信号
                    signal = session_trend
                    strength += 1
                    reasons.append(f'时段倾向修正({session_trend})')

                # 4. 反转形态信号（优先级最高）
                if reversal == 'bullish':
                    signal = 1
                    strength += 3
                    reasons.append('看涨反转形态')
                elif reversal == 'bearish':
                    signal = -1
                    strength += 3
                    reasons.append('看跌反转形态')

                # 5. 价格回到开盘价附近的信号
                if abs(price_change_from_open) < current_row['ATR'] * 0.3:
                    if reversal == 'bullish':
                        signal = 1
                        strength += 2
                        reasons.append('价格回归开盘价+看涨形态')
                    elif reversal == 'bearish':
                        signal = -1
                        strength += 2
                        reasons.append('价格回归开盘价+看跌形态')

            # 波动率过滤
            avg_atr = self.data['ATR'].mean()
            if current_row['ATR'] < avg_atr * 0.5:
                signal = 0  # 波动率过低
                reasons.append('波动率过低')
            elif current_row['ATR'] > avg_atr * 2:
                signal = 0  # 波动率过高
                reasons.append('波动率过高')

            signals.append(signal)
            signal_strengths.append(strength if signal != 0 else 0)
            signal_reasons.append(reasons)

        self.data['Signal'] = signals
        self.data['Signal_Strength'] = signal_strengths
        self.data['Signal_Reasons'] = signal_reasons

    def backtest_strategy(self):
        """
        回测针对连续亏损优化的策略
        """
        self.generate_signals()

        # 初始化变量
        capital = self.initial_capital
        position = 0  # 0表示无仓位，1表示多头，-1表示空头
        entry_price = 0
        entry_index = -10  # 记录入场索引，用于时间止损
        lot_size = 1  # 基础手数为1手
        max_position = 2  # 最大持仓不超过2手
        trades = []
        equity_curve = [capital]
        daily_trades = {}  # 记录每日交易数
        daily_losses = {}  # 记录每日亏损
        consecutive_losses = 0  # 连续亏损次数
        max_consecutive_losses = 2  # 最大连续亏损次数（降低到2次）
        consecutive_loss_pause = 0  # 连续亏损暂停交易周期数

        # 风险管理参数
        daily_max_loss = 1200  # 降低每日最大亏损到1200
        total_max_loss = 1800  # 降低总最大亏损到1800
        min_daily_trades = 1  # 每日最少交易数降低到1
        max_daily_trades = 3  # 每日最多交易数降低到3

        # 动态止损止盈参数
        base_stop_loss_multiplier = 0.8  # 减少止损距离
        base_take_profit_multiplier = 1.6  # 降低止盈目标，提高成功率

        for i in range(1, len(self.data)):  # 从第二根K线开始
            current_row = self.data.iloc[i]
            date = current_row['时间点'].date()

            # 初始化每日统计数据
            if date not in daily_trades:
                daily_trades[date] = 0
            if date not in daily_losses:
                daily_losses[date] = 0

            # 检查是否超过总最大亏损
            total_loss = self.initial_capital - capital
            if total_loss >= total_max_loss:
                print(f"达到总最大亏损限制: {total_loss:.2f}")
                break

            # 检查是否处于连续亏损暂停期
            if consecutive_loss_pause > 0:
                consecutive_loss_pause -= 1
                continue

            # 检查是否超过连续亏损限制
            if consecutive_losses >= max_consecutive_losses:
                print(f"达到最大连续亏损限制: {consecutive_losses}，暂停交易")
                consecutive_loss_pause = 10  # 暂停10个周期再交易
                consecutive_losses = 0  # 重置连续亏损计数

            # 检查是否需要强制交易（每日交易数不足）
            force_trade = False
            if i < len(self.data) - 1:  # 确保不是最后一笔数据
                next_row = self.data.iloc[i + 1]
                next_date = next_row['时间点'].date()
                # 如果到了新的一天，且今天交易数不足，且亏损未超限
                if next_date != date and \
                        daily_trades[date] < min_daily_trades and \
                        daily_losses[date] < daily_max_loss * 0.3:  # 只在亏损较少时强制交易
                    force_trade = True

            # 如果有信号且当前无仓位，或需要强制交易
            if ((current_row['Signal'] != 0 and position == 0) or force_trade) and \
                    daily_trades[date] < max_daily_trades:
                # 检查当日亏损是否超过限制
                if daily_losses[date] < daily_max_loss:
                    # 开仓
                    if force_trade and current_row['Signal'] == 0:
                        # 强制交易使用更严格的趋势判断
                        if (current_row['5周期动量'] > 0 and
                                current_row['10周期动量'] > 0 and
                                current_row['RSI'] < 65):  # 避免超买
                            position = 1  # 做多
                        elif (current_row['5周期动量'] < 0 and
                              current_row['10周期动量'] < 0 and
                              current_row['RSI'] > 35):  # 避免超卖
                            position = -1  # 做空
                        else:
                            position = 0  # 条件不满足，不交易
                    else:
                        # 只有信号强度足够时才交易（提高门槛）
                        if current_row['Signal_Strength'] >= 3:
                            position = current_row['Signal']
                        else:
                            position = 0  # 信号强度不足，不交易

                    # 只有在确定有仓位时才执行开仓操作
                    if position != 0:
                        entry_price = current_row['当前收盘价']  # 使用收盘价作为成交价
                        entry_index = i  # 记录入场索引
                        lots = min(lot_size, max_position)  # 控制仓位

                        # 更新每日交易计数
                        daily_trades[date] += 1

                        trade = {
                            'entry_time': current_row['时间点'],
                            'direction': 'BUY' if position > 0 else 'SELL',
                            'entry_price': entry_price,
                            'lots': lots,
                            'signal_strength': current_row['Signal_Strength'] if not force_trade else 0,
                            'session': self.get_session_type(current_row['小时']),
                            'reasons': ', '.join(current_row['Signal_Reasons']) if not force_trade else '强制交易'
                        }

                        print(
                            f"开仓: {trade['direction']} {lots}手 @ {entry_price:.2f} 时间: {current_row['时间点']} 时段: {trade['session']}")
                        if trade['reasons']:
                            print(f"  开仓理由: {trade['reasons']}")

            # 如果有仓位，检查是否需要平仓
            elif position != 0:
                # 计算动态止损止盈点位 (基于ATR)
                atr = current_row['ATR'] if not np.isnan(current_row['ATR']) else 10  # 默认ATR为10点
                stop_loss_points = atr * base_stop_loss_multiplier  # 缩小止损
                take_profit_points = atr * base_take_profit_multiplier  # 降低止盈目标，提高成功率

                exit_reason = ""
                exit_price = 0

                # 平仓条件
                if position > 0:  # 多头仓位
                    # 止盈条件
                    if current_row['当前最高价'] >= entry_price + take_profit_points:
                        exit_price = entry_price + take_profit_points
                        exit_reason = "止盈"
                    # 止损条件
                    elif current_row['当前最低价'] <= entry_price - stop_loss_points:
                        exit_price = entry_price - stop_loss_points
                        exit_reason = "止损"
                    # 时间止损（缩短至10个周期）
                    elif i - entry_index >= 10:
                        exit_price = current_row['当前收盘价']
                        exit_reason = "时间平仓"

                else:  # 空头仓位
                    # 止盈条件
                    if current_row['当前最低价'] <= entry_price - take_profit_points:
                        exit_price = entry_price - take_profit_points
                        exit_reason = "止盈"
                    # 止损条件
                    elif current_row['当前最高价'] >= entry_price + stop_loss_points:
                        exit_price = entry_price + stop_loss_points
                        exit_reason = "止损"
                    # 时间止损（缩短至10个周期）
                    elif i - entry_index >= 10:
                        exit_price = current_row['当前收盘价']
                        exit_reason = "时间平仓"

                # 如果触发平仓条件
                if exit_reason:
                    # 计算盈亏 (假设每点价值为1美元)
                    points = (exit_price - entry_price) * position
                    profit = points * lots * 100  # 假设每手100单位

                    # 更新资本和亏损记录
                    capital += profit
                    if profit < 0:
                        daily_losses[date] += abs(profit)
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0  # 重置连续亏损计数

                    trade['exit_time'] = current_row['时间点']
                    trade['exit_price'] = exit_price
                    trade['profit'] = profit
                    trade['exit_reason'] = exit_reason
                    trades.append(trade)

                    print(f"平仓: {exit_reason} @ {exit_price:.2f} 时间: {current_row['时间点']} 盈亏: {profit:.2f}")

                    # 重置仓位
                    position = 0
                    entry_price = 0
                    entry_index = -10  # 重置入场索引

            # 更新权益曲线
            equity_curve.append(capital)

        # 输出回测结果
        self.print_backtest_results(trades, equity_curve, daily_trades)

    def print_backtest_results(self, trades, equity_curve, daily_trades):
        """
        打印回测结果
        """
        if not trades:
            print("没有完成的交易")
            return

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        losing_trades = len([t for t in trades if t['profit'] < 0])
        total_profit = sum([t['profit'] for t in trades])
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # 计算盈利交易和亏损交易的平均盈亏
        avg_winning_trade = np.mean([t['profit'] for t in trades if t['profit'] > 0]) if winning_trades > 0 else 0
        avg_losing_trade = np.mean([abs(t['profit']) for t in trades if t['profit'] < 0]) if losing_trades > 0 else 0

        # 计算盈亏比
        profit_factor = avg_winning_trade / avg_losing_trade if avg_losing_trade > 0 else 0

        # 计算每日交易统计
        daily_trade_counts = list(daily_trades.values())
        avg_daily_trades = np.mean(daily_trade_counts) if daily_trade_counts else 0

        print("\n=== 回测结果 ===")
        print(f"总交易数: {total_trades}")
        print(f"盈利交易数: {winning_trades}")
        print(f"亏损交易数: {losing_trades}")
        print(f"胜率: {win_rate:.2f}%")
        print(f"总盈亏: ${total_profit:.2f}")
        print(f"平均盈亏: ${avg_profit:.2f}")
        print(f"平均盈利交易: ${avg_winning_trade:.2f}")
        print(f"平均亏损交易: ${avg_losing_trade:.2f}")
        print(f"盈亏比: {profit_factor:.2f}")
        print(f"初始资本: ${self.initial_capital:.2f}")
        print(f"最终资本: ${equity_curve[-1]:.2f}")
        print(f"收益率: {(equity_curve[-1] - self.initial_capital) / self.initial_capital * 100:.2f}%")
        print(f"平均每日交易数: {avg_daily_trades:.2f}")

        # 按交易原因分类统计
        stop_loss_trades = len([t for t in trades if t['exit_reason'] == '止损'])
        take_profit_trades = len([t for t in trades if t['exit_reason'] == '止盈'])
        time_exit_trades = len([t for t in trades if t['exit_reason'] == '时间平仓'])

        print(f"\n=== 交易分类 ===")
        print(f"止损交易数: {stop_loss_trades}")
        print(f"止盈交易数: {take_profit_trades}")
        print(f"时间平仓数: {time_exit_trades}")

        # 按信号强度统计
        strong_signals = len([t for t in trades if t.get('signal_strength', 0) >= 4])
        weak_signals = len([t for t in trades if t.get('signal_strength', 0) < 4 and t.get('signal_strength', 0) >= 3])
        forced_trades = len([t for t in trades if t.get('signal_strength', 0) == 0])

        print(f"\n=== 信号强度分析 ===")
        print(f"强信号交易数: {strong_signals}")
        print(f"中等信号交易数: {weak_signals}")
        print(f"强制交易数: {forced_trades}")

        # 按时段统计
        asia_trades = len([t for t in trades if t.get('session') == '亚洲'])
        europe_trades = len([t for t in trades if t.get('session') == '欧洲'])
        america_trades = len([t for t in trades if t.get('session') == '美洲'])

        print(f"\n=== 时段分析 ===")
        print(f"亚洲时段交易数: {asia_trades}")
        print(f"欧洲时段交易数: {europe_trades}")
        print(f"美洲时段交易数: {america_trades}")

        # 按星期几统计
        weekday_stats = {}
        for t in trades:
            weekday = t['entry_time'].strftime('%A')
            if weekday not in weekday_stats:
                weekday_stats[weekday] = {'count': 0, 'profit': 0}
            weekday_stats[weekday]['count'] += 1
            weekday_stats[weekday]['profit'] += t['profit']

        print(f"\n=== 星期几分析 ===")
        for weekday, stats in weekday_stats.items():
            print(f"{weekday}: 交易数 {stats['count']}, 盈亏 ${stats['profit']:.2f}")

        # 按月份统计盈亏
        monthly_stats = {}
        for t in trades:
            month = t['entry_time'].strftime('%Y-%m')
            if month not in monthly_stats:
                monthly_stats[month] = {'count': 0, 'profit': 0}
            monthly_stats[month]['count'] += 1
            monthly_stats[month]['profit'] += t['profit']

        print(f"\n=== 月度分析 ===")
        total_months = len(monthly_stats)
        positive_months = len([m for m in monthly_stats.values() if m['profit'] > 0])
        print(f"有交易的月份: {total_months}")
        print(f"盈利月份: {positive_months}")
        print(f"亏损月份: {total_months - positive_months}")

        # 按月份排序并显示
        sorted_months = sorted(monthly_stats.items())
        cumulative_profit = 0
        for month, stats in sorted_months:
            cumulative_profit += stats['profit']
            print(f"{month}: 交易数 {stats['count']}, 盈亏 ${stats['profit']:.2f}, 累计盈亏 ${cumulative_profit:.2f}")

    def run_strategy(self):
        """
        运行完整策略
        """
        print("开始运行针对连续亏损优化的FTMO交易策略...")
        self.backtest_strategy()


def main():
    """
    主函数
    """
    # 设置CSV文件路径
    csv_file = r"C:\Users\孙明辉\PycharmProjects\PythonProject\.venv\Include\AI策略\XAUUSD\XAUUSD_H1_历史数据.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 文件 {csv_file} 不存在")
        return

    # 创建策略实例
    strategy = FTMOMaxConsecutiveLossStrategy(csv_file)

    # 运行策略
    strategy.run_strategy()


if __name__ == "__main__":
    main()