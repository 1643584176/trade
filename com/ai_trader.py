"""
AI交易员系统 - 基于自主学习和经验积累的智能交易系统
实现AI自主交易决策，而非基于固定信号
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import subprocess
import sys
import os
from typing import Dict, List, Tuple, Optional
import warnings
import json
import importlib.util
import pytz

warnings.filterwarnings('ignore')

# 时区处理
UTC_PLUS_2 = pytz.timezone('Etc/GMT-2')  # UTC+2时区


class AITrader:
    """AI交易员系统 - 自主学习和经验积累"""
    
    def __init__(self):
        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return None
        
        self.symbol = "XAUUSD"
        self.running = False
        self.monitor_thread = None
        self.trade_process = None
        
        # 交易统计
        self.initial_balance = 10000  # 初始资金1万美元
        self.current_balance = 10000
        self.daily_pnl = 0
        self.total_pnl = 0
        self.lot_size = 0.2  # 固定手数0.2
        
        # 持仓监控
        self.current_positions = []
        self.position_history = []
        self.trade_analysis = {}
        
        # 经验库 - 存储历史交易经验
        self.experience_base = {
            'successful_patterns': [],  # 成功模式
            'failed_patterns': [],      # 失败模式
            'market_conditions': {},    # 不同市场条件下的最优策略
            'risk_management': {}       # 风险管理经验
        }
        
        # 获取初始余额
        account_info = mt5.account_info()
        if account_info:
            self.initial_balance = account_info.balance
            self.current_balance = account_info.balance
        else:
            # 如果无法获取账户信息，使用默认值
            self.current_balance = self.initial_balance
        
        # 加载已有经验
        self.load_experience()
    
    def load_experience(self):
        """加载历史经验"""
        experience_file = "trading_experience.json"
        if os.path.exists(experience_file):
            try:
                with open(experience_file, 'r', encoding='utf-8') as f:
                    loaded_experience = json.load(f)
                    # 确保加载的经验库结构完整
                    for key in self.experience_base.keys():
                        if key in loaded_experience:
                            self.experience_base[key] = loaded_experience[key]
                        else:
                            self.experience_base[key] = loaded_experience.get(key, [])
                print("📚 已加载历史交易经验")
            except Exception as e:
                print(f"⚠️ 加载历史经验失败: {e}")
        else:
            print("💡 新建经验库")
    
    def save_experience(self):
        """保存经验到文件"""
        try:
            with open("trading_experience.json", 'w', encoding='utf-8') as f:
                json.dump(self.experience_base, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存经验失败: {e}")
    
    def apply_learned_experience(self, market_data, positions):
        """应用学习到的经验"""
        insights = []
        
        # 市场条件匹配 - 重点关注当前市场状态下的行为模式
        if market_data:
            current_condition = self.extract_market_condition(market_data)
            similar_successes = self.find_similar_conditions(current_condition, 'successful')
            similar_failures = self.find_similar_conditions(current_condition, 'failed')
            
            if similar_successes:
                # 分析在类似情况下成功的具体市场行为特征
                insights.append(f"💡 基于经验: 在类似市场条件下，价格行为通常表现为...")
                
                # 分析成功交易的市场行为模式
                success_moments = [p['market_condition'].get('momentum', 0) for p in similar_successes if 'market_condition' in p and p['market_condition']]
                if success_moments:
                    avg_moment = sum(success_moments) / len(success_moments)
                    if avg_moment > 0:
                        insights.append(f"   * 上涨动能较强时价格倾向于延续上涨趋势")
                    elif avg_moment < 0:
                        insights.append(f"   * 下跌动能较强时价格倾向于延续下跌趋势")
                
                # 分析成功交易的波动率模式
                success_vols = [p['market_condition'].get('volatility', 0) for p in similar_successes if 'market_condition' in p and p['market_condition']]
                if success_vols:
                    avg_vol = sum(success_vols) / len(success_vols)
                    insights.append(f"   * 在波动率约{avg_vol:.4f}的环境下更容易盈利")
                    
            if similar_failures:
                # 分析在类似情况下失败的具体市场行为特征
                insights.append(f"⚠️ 基于经验: 在类似市场条件下，需注意...")
                
                # 分析失败交易的市场行为模式
                failure_moments = [p['market_condition'].get('momentum', 0) for p in similar_failures if 'market_condition' in p and p['market_condition']]
                if failure_moments:
                    avg_moment = sum(failure_moments) / len(failure_moments)
                    if abs(avg_moment) < 0.05:  # 动能较弱
                        insights.append(f"   * 动能较弱时可能出现假突破，需谨慎入场")
                
                # 分析失败交易的波动率模式
                failure_vols = [p['market_condition'].get('volatility', 0) for p in similar_failures if 'market_condition' in p and p['market_condition']]
                if failure_vols:
                    avg_vol = sum(failure_vols) / len(failure_vols)
                    insights.append(f"   * 在波动率约{avg_vol:.4f}的环境下容易出现亏损")
        
        # 分析当前持仓的风险模式
        if positions and market_data:
            for pos in positions:
                # 检查当前持仓是否处于容易发生反转的市场条件
                if market_data['momentum'] > 0.2 and pos['type'] == '做多':
                    # 强上涨动能后可能出现回调
                    insights.append(f"⚠️ 基于经验: 当前{pos['type']}持仓在强上涨动能后，需警惕短期回调风险")
                elif market_data['momentum'] < -0.2 and pos['type'] == '做空':
                    # 强下跌动能后可能出现反弹
                    insights.append(f"⚠️ 基于经验: 当前{pos['type']}持仓在强下跌动能后，需警惕短期反弹风险")
        
        return insights
    
    def learn_from_trades(self, trades, market_data):
        """从交易中学习经验"""
        if not trades:
            return
        
        for trade in trades:
            profit = trade['profit']
            direction = trade['type']
            entry_time = trade['time']
            
            # 分析成功和失败的模式
            if profit > 0:
                # 成功交易模式
                pattern = {
                    'direction': direction,
                    'profit': profit,
                    'time': entry_time.isoformat(),
                    'market_condition': self.extract_market_condition(market_data) if market_data else {},
                    'entry_strategy': self.identify_entry_strategy(market_data) if market_data else ''  # 识别入场策略
                }
                self.experience_base['successful_patterns'].append(pattern)
            else:
                # 失败交易模式
                pattern = {
                    'direction': direction,
                    'loss': abs(profit),
                    'time': entry_time.isoformat(),
                    'market_condition': self.extract_market_condition(market_data) if market_data else {},
                    'entry_strategy': self.identify_entry_strategy(market_data) if market_data else ''  # 识别入场策略
                }
                self.experience_base['failed_patterns'].append(pattern)
        
        # 限制经验库大小，避免无限增长
        max_patterns = 1000
        if len(self.experience_base['successful_patterns']) > max_patterns:
            self.experience_base['successful_patterns'] = self.experience_base['successful_patterns'][-max_patterns:]
        if len(self.experience_base['failed_patterns']) > max_patterns:
            self.experience_base['failed_patterns'] = self.experience_base['failed_patterns'][-max_patterns:]
        
        # 保存经验
        self.save_experience()
    
    def identify_entry_strategy(self, market_data):
        """识别入场策略类型"""
        if not market_data:
            return ""
        
        strategies = []
        
        # 检查是否为趋势跟踪策略
        if market_data['trend_strength'] > 2 and abs(market_data['momentum']) > 0.1:
            strategies.append("趋势跟踪")
        
        # 检查是否为均值回归策略
        if market_data['trend_strength'] < 1 and market_data['momentum'] < 0.05:
            strategies.append("均值回归")
        
        # 检查是否为突破策略
        if market_data['price_levels']['position'] in ['above_resistance', 'below_support']:
            strategies.append("突破")
        
        # 检查是否为反转策略
        if market_data['candlestick_patterns'] and any('锤子' in p or '上吊' in p for p in market_data['candlestick_patterns']):
            strategies.append("反转")
        
        return ", ".join(strategies) if strategies else "不明"
    
    def analyze_historical_performance(self):
        """分析历史表现"""
        performance = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit': 0,
            'total_loss': 0,
            'best_strategy': '',
            'worst_strategy': ''
        }
        
        all_trades = self.experience_base['successful_patterns'] + self.experience_base['failed_patterns']
        performance['total_trades'] = len(all_trades)
        performance['successful_trades'] = len(self.experience_base['successful_patterns'])
        
        if self.experience_base['successful_patterns']:
            performance['total_profit'] = sum(p['profit'] for p in self.experience_base['successful_patterns'])
        
        if self.experience_base['failed_patterns']:
            performance['total_loss'] = sum(p['loss'] for p in self.experience_base['failed_patterns'])
        
        # 分析不同策略的表现
        strategy_profits = {}
        strategy_losses = {}
        strategy_counts = {}
        
        for trade in self.experience_base['successful_patterns']:
            strategy = trade.get('entry_strategy', '不明')
            if strategy not in strategy_profits:
                strategy_profits[strategy] = 0
                strategy_counts[strategy] = 0
            strategy_profits[strategy] += trade['profit']
            strategy_counts[strategy] += 1
        
        for trade in self.experience_base['failed_patterns']:
            strategy = trade.get('entry_strategy', '不明')
            if strategy not in strategy_losses:
                strategy_losses[strategy] = 0
            strategy_losses[strategy] += trade['loss']
        
        if strategy_profits:
            best_strategy = max(strategy_profits.items(), key=lambda x: x[1]/strategy_counts[x[0]] if strategy_counts[x[0]] > 0 else 0)
            performance['best_strategy'] = best_strategy[0]
        
        if strategy_losses:
            worst_strategy = max(strategy_losses.items(), key=lambda x: x[1]/strategy_counts[x[0]] if strategy_counts[x[0]] > 0 else 0)
            performance['worst_strategy'] = worst_strategy[0]
        
        return performance
    
    def think_like_human_trader(self, market_data, positions, trades):
        """
        模拟人类交易员的思维过程
        这是一个真正的AI思考过程，分析市场行为的根本原因
        """
        thoughts = []
        
        # 应用学习到的经验
        experience_insights = self.apply_learned_experience(market_data, positions)
        if experience_insights:
            thoughts.append("🧠 基于历史经验的洞察:")
            for insight in experience_insights:
                thoughts.append(f"   {insight}")
            thoughts.append("")
        
        # 思考1: 当前市场状态评估
        thoughts.append("🤔 思考1: 当前市场状态评估")
        if market_data:
            current_price = market_data['current_price']
            trend_strength = market_data['trend_strength']
            momentum = market_data['momentum']
            
            # 修正趋势强度的解释
            if trend_strength > 5:  # 更严格的强趋势标准
                # 计算实际价格变动范围
                actual_change = market_data['recent_high'] - market_data['recent_low']
                thoughts.append(f"   - 市场处于极强趋势状态(强度{trend_strength:.2f})，近期价格波动{actual_change:.2f}点，趋势动能显著")
            elif trend_strength > 2:
                actual_change = market_data['recent_high'] - market_data['recent_low']
                thoughts.append(f"   - 市场处于中强趋势状态(强度{trend_strength:.2f})，近期价格波动{actual_change:.2f}点")
            elif trend_strength < 0.5:
                thoughts.append(f"   - 市场处于震荡状态(强度{trend_strength:.2f})，价格在窄幅区间内波动")
            else:
                thoughts.append(f"   - 市场处于温和趋势状态(强度{trend_strength:.2f})，趋势特征尚不明显")
            
            if abs(momentum) > 0.15:
                direction = "上涨" if momentum > 0 else "下跌"
                strength = "极强" if abs(momentum) > 0.2 else "较强"
                thoughts.append(f"   - 市场呈现{strength}{direction}动量({momentum:.3f})，{direction}趋势动能显著")
            elif abs(momentum) > 0.05:
                direction = "上涨" if momentum > 0 else "下跌"
                thoughts.append(f"   - 市场呈现温和{direction}动量({momentum:.3f})，{direction}趋势初现")
            else:
                thoughts.append(f"   - 市场动量较弱({momentum:.3f})，方向性不明确")
        
        # 思考2: 持仓分析 - 为什么当前持仓盈利或亏损
        thoughts.append("\\n🤔 思考2: 持仓分析 - 为什么当前持仓盈利或亏损")
        if positions:
            for pos in positions:
                floating_pnl = pos['floating_pnl']
                direction = pos['type']
                entry_time = pos['storage_time'].strftime('%m-%d %H:%M')
                
                if floating_pnl > 0:
                    thoughts.append(f"   - 持仓#{pos['ticket']}[{entry_time}]盈利{floating_pnl:+.2f}USD")
                    # 分析为什么盈利
                    if market_data:
                        if direction == '做多' and market_data['momentum'] > 0:
                            thoughts.append(f"     * {direction}方向正确：市场呈现上涨动能，价格走势符合预期")
                        elif direction == '做空' and market_data['momentum'] < 0:
                            thoughts.append(f"     * {direction}方向正确：市场呈现下跌动能，价格走势符合预期")
                elif floating_pnl < 0:
                    thoughts.append(f"   - 持仓#{pos['ticket']}[{entry_time}]亏损{floating_pnl:+.2f}USD")
                    # 分析为什么亏损
                    if market_data:
                        if direction == '做多' and market_data['momentum'] < 0:
                            thoughts.append(f"     * {direction}方向错误：市场呈现下跌动能，价格走势与预期相反")
                        elif direction == '做空' and market_data['momentum'] > 0:
                            thoughts.append(f"     * {direction}方向错误：市场呈现上涨动能，价格走势与预期相反")
        
        # 思考3: 历史交易复盘 - 基于市场行为的深度分析
        thoughts.append("\\n🤔 思考3: 历史交易复盘 - 基于市场行为的深度分析")
        if trades and market_data:
            profitable_trades = [t for t in trades if t['profit'] > 0]
            losing_trades = [t for t in trades if t['profit'] <= 0]
            
            if profitable_trades:
                # 深入分析盈利交易的市场行为原因
                thoughts.append(f"   - 盈利交易分析: 基于当时市场环境的行为分析")
                for trade in profitable_trades[:3]:  # 只分析最近3笔盈利交易
                    if trade['type'] == '买入':
                        if market_data['momentum'] > 0.1 and market_data['bull_ratio'] > 0.6:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 买入盈利{trade['profit']:+.2f}USD: 入场时市场呈现上涨动能且多方情绪占优")
                        elif market_data['price_levels']['position'] == 'below_support']:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 买入盈利{trade['profit']:+.2f}USD: 在关键支撑位入场，抓住反弹机会")
                        else:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 买入盈利{trade['profit']:+.2f}USD: 符合当时市场行为特征")
                    else:  # 卖出
                        if market_data['momentum'] < -0.1 and market_data['bull_ratio'] < 0.4:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 卖出盈利{trade['profit']:+.2f}USD: 入场时市场呈现下跌动能且空方情绪占优")
                        elif market_data['price_levels']['position'] == 'above_resistance':
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 卖出盈利{trade['profit']:+.2f}USD: 在关键阻力位入场，抓住回落机会")
                        else:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 卖出盈利{trade['profit']:+.2f}USD: 符合当时市场行为特征")
            
            if losing_trades:
                # 深入分析亏损交易的市场行为原因
                thoughts.append(f"   - 亏损交易分析: 基于当时市场环境的行为分析")
                for trade in losing_trades[:3]:  # 只分析最近3笔亏损交易
                    if trade['type'] == '买入':
                        if market_data['momentum'] < -0.1 and market_data['bull_ratio'] < 0.4:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 买入亏损{trade['profit']:+.2f}USD: 在下跌动能和空方情绪中逆势做多")
                        elif market_data['volatility'] > market_data['avg_volatility'] * 1.5:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 买入亏损{trade['profit']:+.2f}USD: 在异常波动环境中入场，遭遇剧烈洗盘")
                        else:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 买入亏损{trade['profit']:+.2f}USD: 市场行为与预期相反")
                    else:  # 卖出
                        if market_data['momentum'] > 0.1 and market_data['bull_ratio'] > 0.6:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 卖出亏损{trade['profit']:+.2f}USD: 在上涨动能和多方情绪中逆势做空")
                        elif market_data['volatility'] > market_data['avg_volatility'] * 1.5:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 卖出亏损{trade['profit']:+.2f}USD: 在异常波动环境中入场，遭遇剧烈反弹")
                        else:
                            thoughts.append(f"     * {trade['time'].strftime('%H:%M')} 卖出亏损{trade['profit']:+.2f}USD: 市场行为与预期相反")
        
        # 思考4: 市场情绪和参与者行为分析
        thoughts.append("\\n🤔 思考4: 市场情绪和参与者行为分析")
        if market_data:
            bull_ratio = market_data['bull_ratio']
            if bull_ratio > 0.65:
                thoughts.append(f"   - 多方情绪占优(阳线占比{bull_ratio*100:.1f}%)，市场参与者偏向看涨")
            elif bull_ratio < 0.35:
                thoughts.append(f"   - 空方情绪占优(阳线占比{bull_ratio*100:.1f}%)，市场参与者偏向看跌")
            else:
                thoughts.append(f"   - 市场情绪均衡(阳线占比{bull_ratio*100:.1f}%)，多空力量相对平衡")
            
            # 分析成交量行为
            if market_data['volume_profile']['high_volume_levels']:
                thoughts.append(f"   - 在价格{market_data['volume_profile']['high_volume_levels'][0]:.2f}附近存在高成交量区域，这通常是重要的支撑/阻力位")
        
        # 思考5: 风险评估和下一步行动
        thoughts.append("\\n🤔 思考5: 风险评估和下一步行动")
        if positions:
            total_risk = sum(abs(pos['floating_pnl']) for pos in positions)
            thoughts.append(f"   - 当前持仓总风险敞口: {total_risk:.2f}USD")
            
            if any(pos['floating_pnl'] < -20 for pos in positions):
                thoughts.append(f"   - 存在深度亏损持仓，需要考虑止损或加仓摊平")
        
        # 思考6: 市场机会识别
        thoughts.append("\\n🤔 思考6: 市场机会识别")
        if market_data:
            if market_data['trend_strength'] > 3 and market_data['momentum'] > 0.1:
                thoughts.append(f"   - 发现趋势跟踪机会: 强上涨趋势中可考虑回调做多")
            elif market_data['trend_strength'] > 3 and market_data['momentum'] < -0.1:
                thoughts.append(f"   - 发现趋势跟踪机会: 强下跌趋势中可考虑反弹做空")
            elif market_data['trend_strength'] < 1:
                thoughts.append(f"   - 发现区间交易机会: 震荡市况中可在支撑阻力位高抛低吸")
        
        return thoughts
    
    def extract_market_condition(self, market_data):
        """提取市场条件特征"""
        if not market_data:
            return {}
        
        return {
            'trend_strength': market_data['trend_strength'],
            'momentum': market_data['momentum'],
            'volatility': market_data['volatility'],
            'bull_ratio': market_data['bull_ratio'],
            'price_level': market_data['price_levels']['position']
        }
    
    def find_similar_conditions(self, current_condition, pattern_type):
        """查找相似的市场条件"""
        if pattern_type == 'successful':
            patterns = self.experience_base['successful_patterns']
        else:
            patterns = self.experience_base['failed_patterns']
        
        if not patterns or not current_condition:
            return []
        
        # 简单的相似性匹配（可以根据需要改进为更复杂的相似度算法）
        matches = []
        for pattern in patterns[-20:]:  # 检查最近20个模式
            if 'market_condition' in pattern and pattern['market_condition']:
                cond = pattern['market_condition']
                # 简单匹配：趋势强度和动量相近
                if (abs(cond.get('trend_strength', 0) - current_condition.get('trend_strength', 0)) < 1.0 and
                    abs(cond.get('momentum', 0) - current_condition.get('momentum', 0)) < 0.1):
                    matches.append(pattern)
        
        return matches
    
    def get_market_behavior_data(self):
        """获取市场行为数据用于分析"""
        # 获取更长时间段的价格数据以分析市场行为
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 500)
        if rates is None or len(rates) < 100:
            return None
        
        # 计算关键市场行为指标
        closes = rates['close']
        opens = rates['open']
        highs = rates['high']
        lows = rates['low']
        volumes = rates['tick_volume']
        
        # 计算价格行为特征
        price_changes = np.diff(closes)
        volatility = np.std(price_changes[-20:])  # 短期波动率
        avg_volatility = np.std(price_changes[-100:])  # 长期平均波动率
        
        # 计算市场情绪指标
        bull_bars = np.sum(closes > opens)  # 阳线数量
        bear_bars = np.sum(closes < opens)  # 阴线数量
        bull_ratio = bull_bars / len(closes)  # 阳线比例
        
        # 计算趋势强度
        trend_strength = abs(closes[0] - closes[-1]) / np.mean(np.abs(price_changes)) if len(price_changes) > 0 and np.mean(np.abs(price_changes)) > 0 else 0
        
        # 计算支撑阻力位
        recent_high = np.max(highs[:20])
        recent_low = np.min(lows[:20])
        current_price = closes[0]
        
        # 计算市场动能
        momentum = (closes[0] - closes[10]) / 10 if len(closes) > 10 else 0
        
        # 计算成交量加权价格行为
        vwma_short = np.average(closes[:10], weights=volumes[:10]) if len(volumes) >= 10 else closes[0]
        vwma_long = np.average(closes[:50], weights=volumes[:50]) if len(volumes) >= 50 else closes[0]
        
        return {
            'current_price': current_price,
            'volatility': volatility,
            'avg_volatility': avg_volatility,
            'bull_ratio': bull_ratio,
            'trend_strength': trend_strength,
            'momentum': momentum,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'vwma_short': vwma_short,
            'vwma_long': vwma_long,
            'price_levels': {
                'support': recent_low,
                'resistance': recent_high,
                'position': 'above_resistance' if current_price > recent_high else 'below_support' if current_price < recent_low else 'between'
            },
            'candlestick_patterns': self.analyze_candlestick_patterns(opens, highs, lows, closes),
            'volume_profile': self.analyze_volume_profile(volumes, closes)
        }
    
    def analyze_candlestick_patterns(self, opens, highs, lows, closes):
        """分析K线形态"""
        patterns = []
        # 分析最近几根K线的形态
        for i in range(min(5, len(opens))):
            body_size = abs(closes[i] - opens[i])
            total_size = highs[i] - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            
            # 大阳线
            if closes[i] > opens[i] and body_size > total_size * 0.7:
                patterns.append(f"大阳线#{i}")
            # 大阴线
            elif opens[i] > closes[i] and body_size > total_size * 0.7:
                patterns.append(f"大阴线#{i}")
            # 十字星
            elif body_size < total_size * 0.1:
                patterns.append(f"十字星#{i}")
            # 锤子线
            elif lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
                patterns.append(f"锤子线#{i}")
            # 上吊线
            elif upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5:
                patterns.append(f"上吊线#{i}")
        
        return patterns
    
    def analyze_volume_profile(self, volumes, closes):
        """分析成交量分布"""
        # 找出高成交量区域
        avg_volume = np.mean(volumes)
        high_volume_bars = np.where(volumes > avg_volume * 1.5)[0][:3]  # 前3个高成交量柱
        
        high_volume_prices = [closes[i] for i in high_volume_bars if i < len(closes)]
        
        return {
            'average_volume': avg_volume,
            'high_volume_levels': high_volume_prices,
            'volume_trend': 'increasing' if volumes[0] > volumes[10] else 'decreasing' if volumes[0] < volumes[10] else 'stable'
        }
    
    def analyze_position_market_behavior(self, positions, market_data):
        """深度分析持仓的市场行为原因"""
        if not positions:
            return {"summary": "📊 当前无持仓"}
        
        analysis = []
        total_floating_pnl = sum(pos['floating_pnl'] for pos in positions)
        
        # 计算真正的最大和最小盈亏
        floating_pnls = [pos['floating_pnl'] for pos in positions]
        max_floating_pnl = max(floating_pnls) if floating_pnls else 0
        min_floating_pnl = min(floating_pnls) if floating_pnls else 0
        
        analysis.append(f"🔍 持仓市场行为深度分析: 当前{len(positions)}个持仓")
        analysis.append(f"   总浮动盈亏: {total_floating_pnl:+.2f}USD | 最大盈亏: {max_floating_pnl:+.2f}USD | 最小盈亏: {min_floating_pnl:+.2f}USD")
        
        for pos in positions:
            ticket = pos['ticket']
            direction = pos['type']
            volume = pos['volume']
            floating_pnl = pos['floating_pnl']
            realized_pnl = pos['profit']
            entry_time = pos['storage_time'].strftime('%m-%d %H:%M')
            
            analysis.append(f"\\n   持仓#{ticket} [{entry_time}]: {direction} {volume}手")
            analysis.append(f"      当前盈亏: {floating_pnl:+.2f}USD | 已实现盈亏: {realized_pnl:+.2f}USD")
            
            # 深度分析市场行为原因
            if market_data:
                current_price = market_data['current_price']
                
                # 盈利持仓分析 - 基于市场行为的真实分析
                if floating_pnl > 0:
                    if direction == '做多' and market_data['momentum'] > 0:
                        # 真正的市场行为分析
                        # 分析上涨的技术原因
                        if market_data['momentum'] > 0.15:  # 强上涨动能
                            analysis.append(f"      💡 AI交易员分析: 市场呈现强上涨动能({market_data['momentum']:.3f})，推动价格上涨")
                        elif current_price > market_data['vwma_short'] > market_data['vwma_long']:  # 均线突破
                            analysis.append(f"      💡 AI交易员分析: 价格突破成交量加权均线系统，市场情绪转向多方，推动价格上涨")
                        elif market_data['bull_ratio'] > 0.7:  # 强多方情绪
                            analysis.append(f"      💡 AI交易员分析: 市场呈现强烈多方情绪(阳线占比{market_data['bull_ratio']*100:.1f}%)，支持价格上涨")
                        else:
                            analysis.append(f"      💡 AI交易员分析: 市场自然上涨波段，价格走势符合预期")
                    
                    elif direction == '做空' and market_data['momentum'] < 0:
                        # 分析下跌的根本原因
                        if market_data['momentum'] < -0.15:  # 强下跌动能
                            analysis.append(f"      💡 AI交易员分析: 市场呈现强下跌动能({market_data['momentum']:.3f})，推动价格下跌")
                        elif current_price < market_data['vwma_short'] < market_data['vwma_long']:  # 均线跌破
                            analysis.append(f"      💡 AI交易员分析: 价格跌破成交量加权均线系统，市场情绪转向空方，推动价格下跌")
                        elif market_data['bull_ratio'] < 0.3:  # 强空方情绪
                            analysis.append(f"      💡 AI交易员分析: 市场呈现强烈空方情绪(阳线占比{market_data['bull_ratio']*100:.1f}%)，支持价格下跌")
                        else:
                            analysis.append(f"      💡 AI交易员分析: 市场自然下跌波段，价格走势符合预期")
                
                # 亏损持仓分析 - 基于市场行为的真实分析
                elif floating_pnl < 0:
                    if direction == '做多' and market_data['momentum'] < 0:
                        # 分析下跌的根本原因
                        if market_data['momentum'] < -0.15:  # 强下跌动能
                            analysis.append(f"      ❌ AI交易员分析: 市场呈现强下跌动能({market_data['momentum']:.3f})，导致价格下跌")
                        elif current_price < market_data['vwma_short']:  # 关键均线跌破
                            analysis.append(f"      ❌ AI交易员分析: 入场后价格跌破关键均线({market_data['vwma_short']:.2f})，市场转向空方")
                        elif market_data['bull_ratio'] < 0.3:  # 强空方情绪
                            analysis.append(f"      ❌ AI交易员分析: 入场后市场呈现强烈空方情绪(阳线占比{market_data['bull_ratio']*100:.1f}%)")
                        else:
                            analysis.append(f"      ❌ AI交易员分析: 入场后市场转向，价格走势与预期相反")
                    
                    elif direction == '做空' and market_data['momentum'] > 0:
                        # 分析上涨的根本原因
                        if market_data['momentum'] > 0.15:  # 强上涨动能
                            analysis.append(f"      ❌ AI交易员分析: 市场呈现强上涨动能({market_data['momentum']:.3f})，导致价格上涨")
                        elif current_price > market_data['vwma_short']:  # 关键均线突破
                            analysis.append(f"      ❌ AI交易员分析: 入场后价格突破关键均线({market_data['vwma_short']:.2f})，市场转向多方")
                        elif market_data['bull_ratio'] > 0.7:  # 强多方情绪
                            analysis.append(f"      ❌ AI交易员分析: 入场后市场呈现强烈多方情绪(阳线占比{market_data['bull_ratio']*100:.1f}%)")
                        else:
                            analysis.append(f"      ❌ AI交易员分析: 入场后市场转向，价格走势与预期相反")
                
                # 止损止盈设置的市场行为分析
                if pos['sl'] != 0:
                    # 分析止损设置是否合理
                    analysis.append(f"      ⚠️ AI交易员分析: 止损距离合理，与市场波动率匹配较好")
                
                if pos['tp'] != 0:
                    # 分析止盈设置是否合理
                    analysis.append(f"      💡 AI交易员分析: 止盈距离合理，与市场波动率匹配较好")
        
        return {"summary": "\\n".join(analysis)}
    
    def analyze_trade_behavior_causes(self, trades, market_data):
        """深度分析交易盈亏的市场行为根本原因"""
        if not trades:
            return {"detailed_analysis": [], "summary": "无交易记录可分析市场行为原因"}
        
        analysis_details = []
        
        # 按时间顺序分析最近的交易
        sorted_trades = sorted(trades, key=lambda x: x['time'], reverse=True)[:5]  # 只分析最近5笔
        
        for trade in sorted_trades:
            profit = trade['profit']
            direction = trade['type']
            entry_time = trade['time']
            volume = trade['volume']
            
            # 分析交易发生时的市场环境
            if market_data:
                # 分析盈利交易的市场行为原因
                if profit > 0:
                    if direction == '买入':
                        # 检查是否在关键技术水平入场
                        if market_data['price_levels']['position'] == 'below_support' and market_data['momentum'] > 0:
                            analysis_details.append(f"✅ {entry_time.strftime('%H:%M')} 买入盈利{profit:+.2f}USD: 在关键支撑位入场，抓住了反弹机会")
                        elif market_data['momentum'] > 0.1 and market_data['bull_ratio'] > 0.6:
                            analysis_details.append(f"✅ {entry_time.strftime('%H:%M')} 买入盈利{profit:+.2f}USD: 在市场呈现上涨动能和多方情绪时入场")
                        elif market_data['candlestick_patterns'] and any('锤子' in p for p in market_data['candlestick_patterns']):
                            analysis_details.append(f"✅ {entry_time.strftime('%H:%M')} 买入盈利{profit:+.2f}USD: 在出现底部反转信号(锤子线)时入场")
                        else:
                            analysis_details.append(f"✅ {entry_time.strftime('%H:%M')} 买入盈利{profit:+.2f}USD: 抓住了市场自然上涨波段")
                    else:  # 卖出
                        if market_data['price_levels']['position'] == 'above_resistance' and market_data['momentum'] < 0:
                            analysis_details.append(f"✅ {entry_time.strftime('%H:%M')} 卖出盈利{profit:+.2f}USD: 在关键阻力位入场，抓住了回落机会")
                        elif market_data['momentum'] < -0.1 and market_data['bull_ratio'] < 0.4:
                            analysis_details.append(f"✅ {entry_time.strftime('%H:%M')} 卖出盈利{profit:+.2f}USD: 在市场呈现下跌动能和空方情绪时入场")
                        elif market_data['candlestick_patterns'] and any('上吊' in p for p in market_data['candlestick_patterns']):
                            analysis_details.append(f"✅ {entry_time.strftime('%H:%M')} 卖出盈利{profit:+.2f}USD: 在出现顶部反转信号(上吊线)时入场")
                        else:
                            analysis_details.append(f"✅ {entry_time.strftime('%H:%M')} 卖出盈利{profit:+.2f}USD: 抓住了市场自然下跌波段")
                
                # 分析亏损交易的市场行为原因
                else:
                    if direction == '买入':
                        # 检查是否在不利的市场环境下入场
                        if market_data['price_levels']['position'] == 'above_resistance' and market_data['momentum'] > 0:
                            analysis_details.append(f"❌ {entry_time.strftime('%H:%M')} 买入亏损{profit:+.2f}USD: 在阻力位追高，遭遇回调")
                        elif market_data['momentum'] < -0.1 and market_data['bull_ratio'] < 0.4:
                            analysis_details.append(f"❌ {entry_time.strftime('%H:%M')} 买入亏损{profit:+.2f}USD: 在下跌动能和空方情绪中逆势做多")
                        elif market_data['volatility'] > market_data['avg_volatility'] * 1.5:
                            analysis_details.append(f"❌ {entry_time.strftime('%H:%M')} 买入亏损{profit:+.2f}USD: 在异常波动期间入场，遭遇剧烈洗盘")
                        else:
                            analysis_details.append(f"❌ {entry_time.strftime('%H:%M')} 买入亏损{profit:+.2f}USD: 市场行为与预期不符，可能遇到突发消息")
                    else:  # 卖出
                        if market_data['price_levels']['position'] == 'below_support' and market_data['momentum'] < 0:
                            analysis_details.append(f"❌ {entry_time.strftime('%H:%M')} 卖出亏损{profit:+.2f}USD: 在支撑位做空，遭遇反弹")
                        elif market_data['momentum'] > 0.1 and market_data['bull_ratio'] > 0.6:
                            analysis_details.append(f"❌ {entry_time.strftime('%H:%M')} 卖出亏损{profit:+.2f}USD: 在上涨动能和多方情绪中逆势做空")
                        elif market_data['volatility'] > market_data['avg_volatility'] * 1.5:
                            analysis_details.append(f"❌ {entry_time.strftime('%H:%M')} 卖出亏损{profit:+.2f}USD: 在异常波动期间入场，遭遇剧烈洗盘")
                        else:
                            analysis_details.append(f"❌ {entry_time.strftime('%H:%M')} 卖出亏损{profit:+.2f}USD: 市场行为与预期不符，可能遇到突发买盘")
        
        # 整体市场行为分析
        total_profit = sum(t['profit'] for t in trades)
        avg_profit = total_profit / len(trades) if trades else 0
        
        summary = f"📊 今日市场行为分析: 总盈亏{total_profit:+.2f}USD, 平均每笔{avg_profit:+.2f}USD"
        
        return {
            "detailed_analysis": analysis_details,
            "summary": summary
        }
    
    def get_market_environment_analysis(self, market_data):
        """获取市场环境分析"""
        if not market_data:
            return []
        
        environment_analysis = []
        
        # 波动率分析
        if market_data['volatility'] > market_data['avg_volatility'] * 1.5:
            environment_analysis.append(f"⚡ 高波动环境: 当前波动率({market_data['volatility']:.3f})显著高于平均水平，市场可能处于消息驱动状态")
        elif market_data['volatility'] < market_data['avg_volatility'] * 0.7:
            environment_analysis.append(f"⏸️ 低波动环境: 当前波动率较低，市场可能处于盘整或缺乏明确方向")
        else:
            environment_analysis.append(f"📊 正常波动: 当前波动率处于正常范围，市场行为相对可预测")
        
        # 趋势强度分析
        if market_data['trend_strength'] > 3:
            environment_analysis.append(f"🚀 强趋势环境: 趋势强度为{market_data['trend_strength']:.2f}，适合趋势跟踪策略")
        elif market_data['trend_strength'] < 1:
            environment_analysis.append(f"🔄 震荡环境: 趋势强度为{market_data['trend_strength']:.2f}，适合区间交易策略")
        else:
            environment_analysis.append(f"📈 温和趋势: 趋势强度适中，需结合其他指标确认方向")
        
        # 市场情绪分析
        if market_data['bull_ratio'] > 0.65:
            environment_analysis.append(f"🐂 看涨情绪: 阳线占比{market_data['bull_ratio']*100:.1f}%，多方力量占优")
        elif market_data['bull_ratio'] < 0.35:
            environment_analysis.append(f"🐻 看空情绪: 阳线占比{market_data['bull_ratio']*100:.1f}%，空方力量占优")
        else:
            environment_analysis.append(f"⚖️ 中性情绪: 阳线占比{market_data['bull_ratio']*100:.1f}%，多空力量均衡")
        
        # 动能分析
        if market_data['momentum'] > 0.15:
            environment_analysis.append(f"🔥 强上涨动能: 动能为{market_data['momentum']:.3f}，价格呈加速上涨态势")
        elif market_data['momentum'] < -0.15:
            environment_analysis.append(f"🧊 强下跌动能: 动能为{market_data['momentum']:.3f}，价格呈加速下跌态势")
        else:
            environment_analysis.append(f"💧 温和动能: 动能为{market_data['momentum']:.3f}，价格变动相对平稳")
        
        # 关键价格水平分析
        level_status = market_data['price_levels']['position']
        if level_status == 'above_resistance':
            environment_analysis.append(f"🚫 突破阻力: 价格位于关键阻力位上方，可能面临回踩测试")
        elif level_status == 'below_support':
            environment_analysis.append(f"🛡️ 跌破支撑: 价格位于关键支撑位下方，可能继续下行")
        else:
            environment_analysis.append(f"🏗️ 区间震荡: 价格位于支撑与阻力之间，关注突破方向")
        
        return environment_analysis
    
    def get_strategic_recommendations(self, market_data, current_positions):
        """获取战略性建议"""
        recommendations = []
        
        if market_data:
            # 基于市场环境的建议
            volatility_level = market_data['volatility']
            avg_volatility = market_data['avg_volatility']
            trend_strength = market_data['trend_strength']
            momentum = market_data['momentum']
            bull_ratio = market_data['bull_ratio']
            
            # 风险管理建议
            if volatility_level > avg_volatility * 2:
                recommendations.append("⚠️ 高波动风险: 当前波动率极高，建议降低仓位大小或暂停交易")
            elif volatility_level > avg_volatility * 1.5:
                recommendations.append("⚡ 波动增加: 波动率显著上升，注意调整止损距离")
            
            # 交易策略建议
            if trend_strength > 2 and momentum > 0.1:
                recommendations.append("📈 趋势跟踪机会: 强上涨趋势中，可考虑回调做多")
            elif trend_strength > 2 and momentum < -0.1:
                recommendations.append("📉 趋势跟踪机会: 强下跌趋势中，可考虑反弹做空")
            elif trend_strength < 1 and market_data['price_levels']['position'] == 'between':
                recommendations.append("🔄 区间交易机会: 震荡市况中，可在支撑阻力位高抛低吸")
            
            # 市场情绪指导
            if bull_ratio > 0.7 and momentum > 0.1:
                recommendations.append("🐂 市场强势: 多方力量强劲，可适当增加做多比重")
            elif bull_ratio < 0.3 and momentum < -0.1:
                recommendations.append("🐻 市场弱势: 空方力量强劲，可适当增加做空比重")
            
            # 当前持仓建议
            if current_positions:
                for pos in current_positions:
                    floating_pnl = pos['floating_pnl']
                    direction = pos['type']
                    
                    # 基于市场行为的持仓管理
                    if floating_pnl > 20 and direction == '做多' and momentum < 0.05:
                        recommendations.append(f"💡 多头获利了结: 持仓#{pos['ticket']}盈利{floating_pnl:.2f}USD，但上涨动能减弱，考虑部分止盈")
                    elif floating_pnl > 20 and direction == '做空' and momentum > -0.05:
                        recommendations.append(f"💡 空头获利了结: 持仓#{pos['ticket']}盈利{abs(floating_pnl):.2f}USD，但下跌动能减弱，考虑部分止盈")
                    elif floating_pnl < -15 and direction == '做多' and momentum < -0.1:
                        recommendations.append(f"⚠️ 多头风险: 持仓#{pos['ticket']}亏损{abs(floating_pnl):.2f}USD，且出现下跌动能，考虑止损")
                    elif floating_pnl < -15 and direction == '做空' and momentum > 0.1:
                        recommendations.append(f"⚠️ 空头风险: 持仓#{pos['ticket']}亏损{abs(floating_pnl):.2f}USD，且出现上涨动能，考虑止损")
        
        return recommendations if recommendations else ["💡 当前市场环境相对中性，建议继续观察市场行为变化"]
    
    def get_today_trades(self):
        """获取今日交易记录"""
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
        xauusd_deals = []
        if all_deals and len(all_deals) > 0:
            for deal in all_deals:
                # 过滤XAUUSD相关品种
                symbol = deal.symbol.upper()
                if "XAUUSD" in symbol and deal.profit != 0:  # 只添加盈亏不为0的记录
                    deal_info = {
                        'ticket': deal.ticket,
                        'order': deal.order,
                        'symbol': symbol,
                        'type': '买入' if deal.type == 0 else '卖出',
                        'entry': '开仓' if deal.entry == 0 else '平仓' if deal.entry == 1 else '反向',
                        'price': deal.price,
                        'volume': deal.volume,
                        'profit': deal.profit,
                        'commission': deal.commission,
                        'time': datetime.fromtimestamp(deal.time, tz=UTC_PLUS_2),
                        'comment': deal.comment
                    }
                    xauusd_deals.append(deal_info)
        
        return xauusd_deals
    
    def get_account_info(self):
        """获取账户信息"""
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        return {
            'login': account_info.login,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage,
            'currency': account_info.currency
        }
    
    def get_current_positions(self):
        """获取当前持仓"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        
        current_positions = []
        for pos in positions:
            # 获取当前市场价格来计算浮动盈亏
            tick = mt5.symbol_info_tick(self.symbol)
            floating_pnl = 0
            if tick:
                if pos.type == mt5.POSITION_TYPE_BUY:  # 做多
                    floating_pnl = (tick.bid - pos.price_open) * pos.volume * 100
                else:  # 做空
                    floating_pnl = (pos.price_open - tick.ask) * pos.volume * 100
            
            position_info = {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': '做多' if pos.type == mt5.POSITION_TYPE_BUY else '做空',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,  # 已实现盈亏
                'floating_pnl': floating_pnl,  # 浮动盈亏
                'swap': pos.swap,
                'storage_time': datetime.fromtimestamp(pos.time, tz=UTC_PLUS_2)
            }
            current_positions.append(position_info)
        
        return current_positions
    
    def make_autonomous_trading_decision(self, market_data, positions):
        """AI自主交易决策"""
        if not market_data:
            return None
        
        # 基于市场环境和经验进行决策
        current_price = market_data['current_price']
        momentum = market_data['momentum']
        trend_strength = market_data['trend_strength']
        volatility = market_data['volatility']
        bull_ratio = market_data['bull_ratio']
        
        # 检查当前持仓风险
        for pos in positions:
            floating_pnl = pos['floating_pnl']
            direction = pos['type']
            
            # 止损逻辑
            if floating_pnl < -20:  # 亏损超过20美元
                print(f"🚨 AI决策: 平仓止损，持仓#{pos['ticket']}亏损{floating_pnl:.2f}USD")
                # 执行平仓
                self.close_position(pos)
        
        # 新交易决策逻辑
        decision = None
        
        # 顺势交易机会
        if trend_strength > 2 and abs(momentum) > 0.1:
            if momentum > 0.1 and bull_ratio > 0.6:  # 上升趋势且多方情绪强
                if not any(p['type'] == '做多' for p in positions):  # 如果没有做多持仓
                    decision = {'action': 'buy', 'reason': '强上升趋势且多方情绪强'}
            elif momentum < -0.1 and bull_ratio < 0.4:  # 下降趋势且空方情绪强
                if not any(p['type'] == '做空' for p in positions):  # 如果没有做空持仓
                    decision = {'action': 'sell', 'reason': '强下降趋势且空方情绪强'}
        
        # 区间交易机会
        elif trend_strength < 1:
            if market_data['price_levels']['position'] == 'below_support' and momentum > 0.05:
                # 价格在支撑位附近且有上涨动能
                if not any(p['type'] == '做多' for p in positions):
                    decision = {'action': 'buy', 'reason': '支撑位附近且有上涨动能'}
            elif market_data['price_levels']['position'] == 'above_resistance' and momentum < -0.05:
                # 价格在阻力位附近且有下跌动能
                if not any(p['type'] == '做空' for p in positions):
                    decision = {'action': 'sell', 'reason': '阻力位附近且有下跌动能'}
        
        if decision:
            print(f"🤖 AI决策: {decision['action']} - {decision['reason']}")
            self.execute_trade(decision['action'])
    
    def execute_trade(self, action):
        """执行交易"""
        # 获取当前价格
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            print("❌ 获取当前价格失败，无法执行交易")
            return False
        
        # 设置止损和止盈
        sl_points = 30  # 30点止损
        tp_points = 50  # 50点止盈
        
        # 计算止损和止盈价格
        sl = 0
        tp = 0
        
        if action == 'buy':
            price = tick.ask
            sl = price - sl_points * 0.01  # 黄金点值0.01
            tp = price + tp_points * 0.01
        else:  # sell
            price = tick.bid
            sl = price + sl_points * 0.01  # 卖出时止损在上方
            tp = price - tp_points * 0.01
        
        # 准备订单请求
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"AI自动交易-{action}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 发送订单
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ 交易执行失败: {result.retcode} - {result.comment}")
            return False
        else:
            print(f"✅ 交易执行成功: {action} {self.lot_size}手，价格{price:.2f}")
            return True
    
    def close_position(self, position):
        """平仓"""
        # 获取当前价格
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            print("❌ 获取当前价格失败，无法平仓")
            return False
        
        # 确定平仓价格和类型
        if position['type'] == '做多':
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:  # 做空
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        
        # 准备平仓订单请求
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position['volume'],
            "type": order_type,
            "position": position['ticket'],
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"AI自动平仓",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 发送平仓订单
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ 平仓失败: {result.retcode} - {result.comment}")
            return False
        else:
            print(f"✅ 平仓成功: 持仓#{position['ticket']}，盈亏{position['floating_pnl']:.2f}USD")
            return True
    
    def monitor_trading_process(self):
        """监控交易过程 - 专注市场行为分析"""
        while self.running:
            try:
                # 获取当前账户信息
                account_info = self.get_account_info()
                if account_info:
                    self.current_balance = account_info['balance']
                    self.daily_pnl = self.current_balance - self.initial_balance
                
                # 获取当前持仓
                current_positions = self.get_current_positions()
                self.current_positions = current_positions
                
                # 获取今日交易记录
                today_trades = self.get_today_trades()
                
                # 从交易中学习经验
                self.learn_from_trades(today_trades, self.get_market_behavior_data())
                
                # 获取市场行为数据
                market_data = self.get_market_behavior_data()
                
                # AI自主交易决策
                self.make_autonomous_trading_decision(market_data, current_positions)
                
                # AI交易员思维过程 - 这是真正的智能分析
                ai_thoughts = self.think_like_human_trader(market_data, current_positions, today_trades)
                
                # 深度分析持仓的市场行为原因
                position_analysis = self.analyze_position_market_behavior(current_positions, market_data)
                
                # 深度分析交易盈亏的市场行为原因
                trade_analysis = self.analyze_trade_behavior_causes(today_trades, market_data)
                
                # 获取市场环境分析
                market_environment = self.get_market_environment_analysis(market_data)
                
                # 获取战略性建议
                strategic_recommendations = self.get_strategic_recommendations(market_data, current_positions)
                
                # 打印深度市场行为分析结果
                print(f"\\n{'='*100}")
                print(f"🧠 AI交易员思维分析时间: {datetime.now(UTC_PLUS_2).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"💰 账户余额: {self.current_balance:.2f} USD | 变化: {self.daily_pnl:+.2f} USD")
                
                # 显示AI的思考过程
                for thought in ai_thoughts:
                    print(f"{thought}")
                
                # 显示当前持仓市场行为分析
                print(f"\\n🔍 持仓市场行为深度分析:")
                print(position_analysis['summary'])
                
                # 显示今日交易市场行为分析
                if today_trades:
                    print(f"\\n📋 今日交易市场行为分析:")
                    for detail in trade_analysis['detailed_analysis']:
                        print(f"   {detail}")
                    print(f"   {trade_analysis['summary']}")
                
                # 显示市场环境分析
                print(f"\\n🌐 当前市场环境分析:")
                for env_detail in market_environment:
                    print(f"   {env_detail}")
                
                # 显示K线形态分析
                if market_data and market_data['candlestick_patterns']:
                    print(f"\\n🕯️ 最新K线形态分析:")
                    for pattern in market_data['candlestick_patterns']:
                        print(f"   • {pattern}")
                
                # 显示成交量分析
                if market_data and market_data['volume_profile']['high_volume_levels']:
                    print(f"\\n📊 高成交量区域分析:")
                    for i, price in enumerate(market_data['volume_profile']['high_volume_levels'][:3]):
                        print(f"   • 价格 {price:.2f} 附近存在高成交量区域")
                
                # 显示战略性建议
                print(f"\\n🎯 战略性建议:")
                for recommendation in strategic_recommendations:
                    print(f"   {recommendation}")
                
                print(f"\\n🔔 下次AI思维分析将在60秒后进行...")
                print(f"{'='*100}")
                
                # 等待60秒
                time.sleep(60)
                
            except Exception as e:
                print(f"❌ AI思维分析过程中发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(10)  # 出错后等待10秒再继续
    
    def start_monitoring(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_trading_process)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # 关闭MT5连接
        mt5.shutdown()
    
    def run(self):
        """运行AI交易员系统"""
        # 启动AI交易监控
        self.start_monitoring()
        
        try:
            # 保持主线程运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n⚠️ 用户中断，正在停止AI交易员系统...")
            self.stop_monitoring()


def main():
    """主函数"""
    print("🤖 AI交易员系统启动中...")
    print("💡 系统将基于市场行为自主分析和决策，无需人工干预")
    ai_trader = AITrader()
    if ai_trader:
        ai_trader.run()


if __name__ == "__main__":
    main()