import MetaTrader5 as mt5
import time
from datetime import datetime
import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入配置加载器
from com.mlc.utils.config_loader import config

# 导入时间处理工具
try:
    from utils.time_utils import TimeUtils
except ImportError:
    TimeUtils = None

def initialize_mt5():
    """初始化MT5连接"""
    if not mt5.initialize():
        print("MT5初始化失败")
        return False
    return True

def get_account_info():
    """获取账户信息"""
    account_info = mt5.account_info()
    if account_info is None:
        return None
    
    return {
        'balance': account_info.balance,
        'equity': account_info.equity,
        'profit': account_info.profit,
        'margin': account_info.margin,
        'free_margin': account_info.margin_free,
        'margin_level': account_info.margin_level
    }

def get_positions():
    """获取所有持仓信息"""
    positions = mt5.positions_get()
    if positions is None:
        return []
    
    position_list = []
    for position in positions:
        # 获取交易品种信息
        symbol_info = mt5.symbol_info(position.symbol)
        point = symbol_info.point if symbol_info is not None else 0.00001
        
        # 安全获取属性，避免某些版本没有的属性
        position_info = {
            'ticket': position.ticket,
            'symbol': position.symbol,
            'type': '多' if position.type == 0 else '空',
            'volume': position.volume,
            'price_open': position.price_open,
            'sl': position.sl,
            'tp': position.tp,
            'profit': position.profit,
            'swap': getattr(position, 'swap', 0),
            'commission': getattr(position, 'commission', 0),
            'time': datetime.fromtimestamp(position.time).strftime('%Y-%m-%d %H:%M:%S')
        }
        position_list.append(position_info)
    
    return position_list

def display_account_info(account_info):
    """显示账户信息"""
    if account_info is None:
        print("无法获取账户信息")
        return
    
    print(f"账户余额: {account_info['balance']:.2f} | 账户净值: {account_info['equity']:.2f} | 浮动盈亏: {account_info['profit']:.2f}")

def display_positions(positions):
    """显示持仓信息，一行展示一条持仓"""
    if not positions:
        print("当前无持仓")
        return
    
    print(f"当前持仓数量: {len(positions)}")
    for pos in positions:
        print(f"[{pos['symbol']}] {pos['type']} 手数:{pos['volume']} 开仓价:{pos['price_open']:.5f} 止损:{pos['sl']:.5f} 止盈:{pos['tp']:.5f} 盈亏:{pos['profit']:.2f}")

def main():
    """主函数"""
    # 初始化MT5
    if not initialize_mt5():
        return
    
    print("MT5账户持仓监控程序启动")
    
    try:
        while True:
            # 获取当前时间
            now = datetime.now()
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}]")
            
            # 获取并显示账户信息
            account_info = get_account_info()
            display_account_info(account_info)
            
            # 获取并显示持仓信息
            positions = get_positions()
            display_positions(positions)
            
            # 等待60秒
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
    finally:
        # 关闭MT5连接
        mt5.shutdown()
        print("MT5连接已关闭")

if __name__ == "__main__":
    main()