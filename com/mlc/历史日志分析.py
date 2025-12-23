import re
from datetime import datetime
from collections import defaultdict

def analyze_trading_logs():
    """
    分析两个日志文件，提取每个M15周期的预测概率、持仓方向和持仓收益信息
    """
    
    # 定义正则表达式模式
    prediction_pattern = r'预测概率 - 上涨: ([0-9.]+), 下跌: ([0-9.]+)'
    decision_pattern = r'决策: (做多|做空|观望)'
    position_pattern = r'当前持仓: (做多|做空), 持仓收益: ([\-0-9.]+)美元'
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    
    # 存储分析结果
    log_data = {
        'trading_log': {
            'predictions': [],
            'decisions': [],
            'positions': []
        },
        'xauusd_trading_log': {
            'predictions': [],
            'decisions': [],
            'positions': []
        }
    }
    
    # 分析 trading_log.txt
    try:
        with open('D:/newProject/Trader/com/mlc/test/trading_log.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for line in lines:
                # 提取时间戳
                timestamp_match = re.search(timestamp_pattern, line)
                timestamp = timestamp_match.group(1) if timestamp_match else None
                
                # 提取预测概率
                pred_match = re.search(prediction_pattern, line)
                if pred_match and timestamp:
                    up_prob = float(pred_match.group(1))
                    down_prob = float(pred_match.group(2))
                    log_data['trading_log']['predictions'].append({
                        'timestamp': timestamp,
                        'up_probability': up_prob,
                        'down_probability': down_prob
                    })
                
                # 提取决策
                decision_match = re.search(decision_pattern, line)
                if decision_match and timestamp:
                    decision = decision_match.group(1)
                    log_data['trading_log']['decisions'].append({
                        'timestamp': timestamp,
                        'decision': decision
                    })
                
                # 提取持仓信息
                pos_match = re.search(position_pattern, line)
                if pos_match and timestamp:
                    direction = pos_match.group(1)
                    profit = float(pos_match.group(2))
                    log_data['trading_log']['positions'].append({
                        'timestamp': timestamp,
                        'direction': direction,
                        'profit': profit
                    })
    except FileNotFoundError:
        print("警告: 未找到 trading_log.txt 文件")
    
    # 分析 xauusd_trading_log.txt
    try:
        with open('D:/newProject/Trader/com/mlc/xauusd/xauusd_trading_log.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for line in lines:
                # 提取时间戳
                timestamp_match = re.search(timestamp_pattern, line)
                timestamp = timestamp_match.group(1) if timestamp_match else None
                
                # 提取预测概率
                pred_match = re.search(prediction_pattern, line)
                if pred_match and timestamp:
                    up_prob = float(pred_match.group(1))
                    down_prob = float(pred_match.group(2))
                    log_data['xauusd_trading_log']['predictions'].append({
                        'timestamp': timestamp,
                        'up_probability': up_prob,
                        'down_probability': down_prob
                    })
                
                # 提取决策
                decision_match = re.search(decision_pattern, line)
                if decision_match and timestamp:
                    decision = decision_match.group(1)
                    log_data['xauusd_trading_log']['decisions'].append({
                        'timestamp': timestamp,
                        'decision': decision
                    })
                
                # 提取持仓信息
                pos_match = re.search(position_pattern, line)
                if pos_match and timestamp:
                    direction = pos_match.group(1)
                    profit = float(pos_match.group(2))
                    log_data['xauusd_trading_log']['positions'].append({
                        'timestamp': timestamp,
                        'direction': direction,
                        'profit': profit
                    })
    except FileNotFoundError:
        print("警告: 未找到 xauusd_trading_log.txt 文件")
    
    # 打印分析结果
    print("="*80)
    print("TRADING_LOG.TXT 分析结果:")
    print("="*80)
    
    # 整合显示 trading_log 的信息
    if log_data['trading_log']['predictions']:
        print("\n整合信息 (按时间顺序):")
        # 创建时间戳索引
        pred_dict = {pred['timestamp']: pred for pred in log_data['trading_log']['predictions']}
        decision_dict = {dec['timestamp']: dec for dec in log_data['trading_log']['decisions']}
        pos_dict = {pos['timestamp']: pos for pos in log_data['trading_log']['positions']}
        
        # 获取所有时间戳并排序
        all_timestamps = sorted(set(pred_dict.keys()) | set(decision_dict.keys()) | set(pos_dict.keys()))
        
        for timestamp in all_timestamps:
            pred = pred_dict.get(timestamp, {})
            decision = decision_dict.get(timestamp, {})
            pos = pos_dict.get(timestamp, {})
            
            # 获取持仓方向和收益
            direction = pos.get('direction', '无持仓')
            profit = pos.get('profit', 0.0)
            
            # 打印整合信息
            up_prob = pred.get('up_probability', 'N/A')
            down_prob = pred.get('down_probability', 'N/A')
            decision_text = decision.get('decision', 'N/A')
            
            print(f"  {timestamp}, 上涨概率: {up_prob}, 下跌概率: {down_prob}, 决策: {decision_text}, 持仓方向: {direction}, 持仓收益: {profit:.2f}美元")
    
    print("\n" + "="*80)
    print("XAUUSD_TRADING_LOG.TXT 分析结果:")
    print("="*80)
    
    # 整合显示 xauusd_trading_log 的信息
    if log_data['xauusd_trading_log']['predictions']:
        print("\n整合信息 (按时间顺序):")
        # 创建时间戳索引
        pred_dict = {pred['timestamp']: pred for pred in log_data['xauusd_trading_log']['predictions']}
        decision_dict = {dec['timestamp']: dec for dec in log_data['xauusd_trading_log']['decisions']}
        pos_dict = {pos['timestamp']: pos for pos in log_data['xauusd_trading_log']['positions']}
        
        # 获取所有时间戳并排序
        all_timestamps = sorted(set(pred_dict.keys()) | set(decision_dict.keys()) | set(pos_dict.keys()))
        
        for timestamp in all_timestamps:
            pred = pred_dict.get(timestamp, {})
            decision = decision_dict.get(timestamp, {})
            pos = pos_dict.get(timestamp, {})
            
            # 获取持仓方向和收益
            direction = pos.get('direction', '无持仓')
            profit = pos.get('profit', 0.0)
            
            # 打印整合信息
            up_prob = pred.get('up_probability', 'N/A')
            down_prob = pred.get('down_probability', 'N/A')
            decision_text = decision.get('decision', 'N/A')
            
            print(f"  {timestamp}, 上涨概率: {up_prob}, 下跌概率: {down_prob}, 决策: {decision_text}, 持仓方向: {direction}, 持仓收益: {profit:.2f}美元")
    
    return log_data

def get_summary_by_log_file(log_file_path):
    """
    获取单个日志文件的摘要信息
    
    参数:
        log_file_path (str): 日志文件路径
    
    返回:
        dict: 包含预测、决策和持仓信息的字典
    """
    prediction_pattern = r'预测概率 - 上涨: ([0-9.]+), 下跌: ([0-9.]+)'
    decision_pattern = r'决策: (做多|做空|观望)'
    position_pattern = r'当前持仓: (做多|做空), 持仓收益: ([\-0-9.]+)美元'
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    
    results = {
        'predictions': [],
        'decisions': [],
        'positions': []
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for line in lines:
                # 提取时间戳
                timestamp_match = re.search(timestamp_pattern, line)
                timestamp = timestamp_match.group(1) if timestamp_match else None
                
                # 提取预测概率
                pred_match = re.search(prediction_pattern, line)
                if pred_match and timestamp:
                    up_prob = float(pred_match.group(1))
                    down_prob = float(pred_match.group(2))
                    results['predictions'].append({
                        'timestamp': timestamp,
                        'up_probability': up_prob,
                        'down_probability': down_prob
                    })
                
                # 提取决策
                decision_match = re.search(decision_pattern, line)
                if decision_match and timestamp:
                    decision = decision_match.group(1)
                    results['decisions'].append({
                        'timestamp': timestamp,
                        'decision': decision
                    })
                
                # 提取持仓信息
                pos_match = re.search(position_pattern, line)
                if pos_match and timestamp:
                    direction = pos_match.group(1)
                    profit = float(pos_match.group(2))
                    results['positions'].append({
                        'timestamp': timestamp,
                        'direction': direction,
                        'profit': profit
                    })
    except FileNotFoundError:
        print(f"警告: 未找到 {log_file_path} 文件")
    
    return results

if __name__ == "__main__":
    # 运行分析
    log_analysis = analyze_trading_logs()
    
    # 也可以单独分析某个文件
    print("\n" + "="*80)
    print("单独分析示例:")
    print("="*80)
    
    # 示例：分析单个文件
    test_results = get_summary_by_log_file('D:/newProject/Trader/com/mlc/test/trading_log.txt')
    print(f"trading_log.txt 包含 {len(test_results['predictions'])} 条预测记录, "
          f"{len(test_results['decisions'])} 条决策记录, "
          f"{len(test_results['positions'])} 条持仓记录")