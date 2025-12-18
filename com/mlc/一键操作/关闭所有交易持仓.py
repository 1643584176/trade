import MetaTrader5 as mt5
import time

def connect_mt5():
    """连接MT5终端，失败则退出"""
    if not mt5.initialize():
        print(f"MT5连接失败！错误代码：{mt5.last_error()}")
        quit()
    print("MT5连接成功")

def close_all_positions():
    """关闭所有持仓订单（多仓/空仓通用）"""
    # 1. 获取所有当前持仓
    positions = mt5.positions_get()
    if not positions:
        print("当前无任何持仓")
        return True

    print(f"发现 {len(positions)} 个持仓，开始批量平仓...")

    # 2. 遍历每个持仓，逐一平仓
    for position in positions:
        symbol = position.symbol  # 持仓品种（如EURUSD）
        volume = position.volume  # 持仓手数
        position_type = position.type  # 持仓类型：POSITION_TYPE_BUY（多仓）/POSITION_TYPE_SELL（空仓）
        position_ticket = position.ticket  # 持仓订单号（必填，用于精准平仓）

        # 3. 构建平仓请求（反向操作：多仓卖平，空仓买平）
        close_type = mt5.ORDER_TYPE_SELL if position_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        # 获取当前平仓价格（多仓平用卖价bid，空仓平用买价ask）
        current_price = mt5.symbol_info_tick(symbol).bid if position_type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

        # 平仓请求参数
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,  # 执行交易
            "symbol": symbol,
            "volume": volume,  # 平仓手数 = 持仓手数
            "type": close_type,  # 平仓类型（反向）
            "position": position_ticket,  # 绑定要平仓的订单号（关键）
            "price": current_price,  # 平仓价格
            "deviation": 3,  # 允许滑点（3个点，根据行情可调整）
            "comment": "批量关闭所有持仓",
            "type_filling": mt5.ORDER_FILLING_FOK,  # 成交方式：立即成交（否则取消）
        }

        # 4. 发送平仓请求
        result = mt5.order_send(close_request)

        # 5. 打印平仓结果
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ 平仓成功！品种：{symbol}，订单号：{position_ticket}，手数：{volume}，平仓价格：{current_price:.5f}")
        else:
            print(f"❌ 平仓失败！品种：{symbol}，订单号：{position_ticket}，错误代码：{result.retcode}，原因：{result.comment}")

    # 6. 平仓后验证是否所有持仓已关闭
    remaining_positions = mt5.positions_get()
    if not remaining_positions:
        print("\n🎉 所有持仓已成功关闭")
        return True
    else:
        print(f"\n⚠️  仍有 {len(remaining_positions)} 个持仓未关闭，清单：{[p.symbol for p in remaining_positions]}")
        return False

if __name__ == "__main__":
    # 步骤1：连接MT5
    connect_mt5()

    # 步骤2：关闭所有持仓
    close_all_positions()

    # 步骤3：断开MT5连接（可选，若后续还要操作可注释）
    mt5.shutdown()
    print("MT5连接已断开")