import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import time


def get_ftmo_xauusd_orders(start_date=None, end_date=None):
    """
    查询FTMO账户中XAUUSD的历史订单
    :param start_date: 起始日期（如"2026-01-10"），默认查询当天
    :param end_date: 结束日期（如"2026-01-14"），默认查询当天
    :return: XAUUSD订单列表
    """

    if not mt5.initialize():
        print(f"MT5初始化失败，错误：{mt5.last_error()}")
        return None


    # 2. 处理时间范围（默认当天，UTC时区）
    if not start_date:
        start_utc = datetime.now(timezone.utc).date()
    else:
        start_utc = datetime.strptime(start_date, "%Y-%m-%d").date()

    if not end_date:
        end_utc = datetime.now(timezone.utc).date()
    else:
        end_utc = datetime.strptime(end_date, "%Y-%m-%d").date()

    # 转成UTC时间的datetime对象（当天0点）
    start_time_utc = datetime.combine(start_utc, datetime.min.time(), timezone.utc)
    end_time_utc = datetime.combine(end_utc, datetime.max.time(), timezone.utc)

    # 转成MT5需要的秒级时间戳（旧版兼容）
    start_ts = int(time.mktime(start_time_utc.timetuple()))
    end_ts = int(time.mktime(end_time_utc.timetuple()))

    # 获取账户信息以获取当前余额
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ 无法获取账户信息")
        mt5.shutdown()
        return None

    current_balance = account_info.balance
    print(f"💰 当前账户余额: {current_balance:.2f} USD")

    # 3. 查询指定时间范围的所有订单
    try:
        all_orders = mt5.history_orders_get(start_ts, end_ts)
    except Exception as e:
        print(f"❌ 查询订单异常：{str(e)}")
        mt5.shutdown()
        return None

    # 4. 过滤XAUUSD品种（兼容FTMO的不同后缀：XAUUSD、XAUUSDm、XAUUSD.pro等）
    xauusd_orders = []
    if all_orders and len(all_orders) > 0:
        # 订单字段映射（兼容旧版MT5）
        order_type_map = {0: "买入(市价)", 1: "卖出(市价)", 2: "买入限价",
                          3: "卖出限价", 4: "买入止损", 5: "卖出止损"}
        order_state_map = {0: "待成交", 1: "部分成交", 2: "已成交",
                           3: "已撤销", 4: "已取消", 5: "已过期", 6: "已拒绝"}

        for order in all_orders:
            # 过滤XAUUSD相关品种（忽略大小写、后缀）
            symbol = order.symbol.upper()
            if "XAUUSD" in symbol:
                # 兼容旧版字段
                order_time_ts = getattr(order, 'time', getattr(order, 'time_setup', 0))
                order_time_utc = datetime.fromtimestamp(order_time_ts, timezone.utc) if order_time_ts != 0 else "未知时间"
                order_volume = getattr(order, 'volume', getattr(order, 'volume_initial', 0.0))
                order_price = getattr(order, 'price_open', getattr(order, 'price_order', 0.0))

                # 整理订单信息
                order_info = {
                    "订单号": order.ticket,
                    "品种": symbol,
                    "订单类型": order_type_map.get(order.type, f"未知({order.type})"),
                    "订单状态": order_state_map.get(order.state, f"未知({order.state})"),
                    "开仓价格": order_price,
                    "手数": order_volume,
                    "订单时间(UTC)": order_time_utc.strftime("%Y-%m-%d %H:%M:%S") if isinstance(order_time_utc,
                                                                                                datetime) else order_time_utc,
                    "注释": order.comment if order.comment else "无"
                }
                xauusd_orders.append(order_info)
    else:
        print("ℹ️ 该时间范围无任何订单")

    # 5. 查询交易成交记录（deals）以获取盈亏信息
    try:
        all_deals = mt5.history_deals_get(start_ts, end_ts)
    except Exception as e:
        print(f"❌ 查询交易记录异常：{str(e)}")
        mt5.shutdown()
        return None

    # 6. 过滤XAUUSD的交易记录，只包含盈亏不为0的记录
    xauusd_deals = []
    daily_profit_loss = 0  # 今日盈亏
    daily_losses = []  # 今日亏损记录（绝对值）

    if all_deals and len(all_deals) > 0:
        # 交易类型映射
        deal_type_map = {0: "买入", 1: "卖出", 2: "余额", 3: "信贷"}
        deal_entry_map = {0: "开仓", 1: "平仓", 2: "反向"}  # entry表示是开仓还是平仓

        for deal in all_deals:
            # 过滤XAUUSD相关品种
            symbol = deal.symbol.upper()
            if "XAUUSD" in symbol and deal.profit != 0:  # 只添加盈亏不为0的记录
                # 兼容旧版字段
                deal_time_ts = getattr(deal, 'time', 0)
                deal_time_utc = datetime.fromtimestamp(deal_time_ts, timezone.utc) if deal_time_ts != 0 else "未知时间"

                # 整理交易信息
                deal_info = {
                    "交易号": deal.ticket,
                    "订单号": deal.order,
                    "品种": symbol,
                    "交易类型": deal_type_map.get(deal.type, f"未知({deal.type})"),
                    "交易方向": deal_entry_map.get(deal.entry, f"未知({deal.entry})"),
                    "成交价格": deal.price,
                    "手数": deal.volume,
                    "盈亏": deal.profit,
                    "佣金": deal.commission,
                    "成交时间(UTC)": deal_time_utc.strftime("%Y-%m-%d %H:%M:%S") if isinstance(deal_time_utc,
                                                                                               datetime) else deal_time_utc,
                    "注释": deal.comment if deal.comment else "无"
                }
                xauusd_deals.append(deal_info)

                # 累加今日盈亏
                daily_profit_loss += deal.profit

                # 记录亏损（绝对值）
                if deal.profit < 0:
                    daily_losses.append(abs(deal.profit))
    else:
        print("ℹ️ 该时间范围无任何交易记录")

    # 7. 计算统计信息
    total_profit = sum(deal['盈亏'] for deal in xauusd_deals if deal['盈亏'] is not None)
    total_commission = sum(deal['佣金'] for deal in xauusd_deals if deal['佣金'] is not None)

    # 计算今日初始余额（当前余额 - 今日盈亏）
    daily_start_balance = current_balance - daily_profit_loss

    # 计算今日最大亏损
    max_daily_loss = max(daily_losses) if daily_losses else 0

    # 8. 断开MT5连接
    mt5.shutdown()


    if xauusd_deals:
        print(f"📊 今日盈亏: {daily_profit_loss:.2f} USD")
        print(f"📊 今日初始余额: {daily_start_balance:.2f} USD")
        print(f"📊 当前余额: {current_balance:.2f} USD")
        print(f"📊 今日最大亏损: {max_daily_loss:.2f} USD")
        print(f"💸 总佣金: {total_commission:.2f} USD")
        print(f"💰 净收益: {total_profit:.2f} USD")
    else:
        print(f"ℹ️ 该时间范围无XAUUSD交易记录（盈亏不为0）")
        print(f"📊 今日初始余额: {current_balance:.2f} USD")
        print(f"📊 当前余额: {current_balance:.2f} USD")

    return xauusd_orders, xauusd_deals


def print_deals_details(deals_list):
    """打印交易记录详情"""
    if deals_list:
        print("\n" + "=" * 120)
        print("💼 FTMO XAUUSD交易记录详情（仅显示盈亏不为0的记录）：")
        print("=" * 120)
        print(
            f"{'交易号':<10} {'订单号':<10} {'品种':<8} {'类型':<6} {'方向':<6} {'价格':<10} {'手数':<6} {'盈亏':<10} {'佣金':<10} {'时间':<20} {'注释'}")
        print("-" * 120)
        for idx, deal in enumerate(deals_list, 1):
            profit_str = f"{deal['盈亏']:.2f}"
            if deal['盈亏'] > 0:
                profit_str += " 💚"
            elif deal['盈亏'] < 0:
                profit_str += " ❤️"

            print(f"{deal['交易号']:<10} {deal['订单号']:<10} {deal['品种']:<8} "
                  f"{deal['交易类型']:<6} {deal['交易方向']:<6} {deal['成交价格']:<10.2f} "
                  f"{deal['手数']:<6.1f} {profit_str:<12} {deal['佣金']:<10.2f} "
                  f"{deal['成交时间(UTC)']:<20} {deal['注释']}")
        print("=" * 120)


# 主函数调用
if __name__ == "__main__":
    # 用法1：查询当天XAUUSD订单和交易记录
    orders, deals = get_ftmo_xauusd_orders()
    #
    # # 用法2：查询指定日期范围（如2026-01-10到2026-01-14）
    # orders, deals = get_ftmo_xauusd_orders(start_date="2026-01-14", end_date="2026-01-14")

    # # 打印订单详情
    # if orders:
    #     print("\n" + "=" * 80)
    #     print("📜 FTMO XAUUSD历史订单详情：")
    #     print("=" * 80)
    #     for idx, order in enumerate(orders, 1):
    #         print(f"\n【订单 {idx}】")
    #         for key, value in order.items():
    #             print(f"  {key}: {value}")
    #     print("\n" + "=" * 80)
