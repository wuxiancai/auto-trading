#!/usr/bin/env python3
"""
BTCUSDT 15m 布林带循环策略回测 + 参数优化
- 目标：避免窄幅震荡段（UP/DN 太近）参与交易，在宽度扩张且触及上下轨的区域执行：
  - 下轨+RSI低 -> 开多；到上轨+RSI高 -> 平多并反手开空；反之亦然。
- 数据：>= 2 个月（分页抓取 Binance Futures 15m K线）。
- 费用：支持 maker/taker；默认 taker 0.05%，maker 0.027%。
- 风控：ATR 动态止损，随中轨移动（跟踪）。
- 优化：网格搜索自动调参，筛选胜率>=60% 或 总回报>=20% 的参数集，保存到 config.json。
此示例仅用于研究与演示，不构成投资建议。
"""

import time
import math
import json
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import argparse
import re

BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# --------- 数据抓取 ---------

def _get_with_retry(url: str, params: Dict, max_retry: int = 5, timeout: int = 15):
    backoff = 0.5
    for attempt in range(max_retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt == max_retry - 1:
                raise
            time.sleep(backoff)
            backoff *= 2


def fetch_klines_paged(symbol: str = "BTCUSDT", interval: str = "15m", days: int = 65) -> pd.DataFrame:
    """分页抓取最近 `days` 天的期货K线。Binance单次最多返回1500根，需要分页。
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    klines: List[List] = []
    cur_start = start_ms
    while True:
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": 1500, "startTime": cur_start, "endTime": end_ms}
        r = _get_with_retry(url, params=params, timeout=15)
        data = r.json()
        if not data:
            break
        klines.extend(data)
        last_close = data[-1][6]  # close_time
        cur_start = last_close + 1  # 下一页
        if cur_start >= end_ms:
            break
        time.sleep(0.05)

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","num_trades",
            "taker_buy_base","taker_buy_quote","ignore",
        ],
    )
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df

# --------- 指标 ---------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def compute_indicators(df: pd.DataFrame, bb_window: int = 20, bb_mult: float = 2.0) -> pd.DataFrame:
    close = df["close"]
    mid = close.rolling(bb_window).mean()
    std = close.rolling(bb_window).std(ddof=0)
    upper = mid + bb_mult * std
    lower = mid - bb_mult * std
    width = (upper - lower) / (mid + 1e-12)
    width_slope = width.diff()

    df = df.copy()
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = width
    df["bb_width_slope"] = width_slope
    df["rsi"] = rsi(close, 14)
    df["atr"] = atr(df["high"], df["low"], close, 14)
    df["sma50"] = close.rolling(50).mean()
    df["sma_slope"] = df["sma50"].diff() / (df["sma50"] + 1e-12)
    df.attrs["bb_window"] = bb_window
    df.attrs["bb_mult"] = bb_mult
    return df

# --------- 参数与费用 ---------

@dataclass
class Params:
    rsi_low: int = 30
    rsi_high: int = 70
    bb_width_min: float = 0.05
    width_slope_min: float = 0.0
    stop_atr: float = 1.5
    fee_mode: str = "maker"
    taker_fee: float = 0.00005
    maker_fee: float = 0.00001
    leverage: float = 3.0
    bb_window: int = 20
    bb_mult: float = 2.0
    ma_slope_min: float = 0.0001

    def fee(self) -> float:
        return self.maker_fee if self.fee_mode == "maker" else self.taker_fee

# --------- 回测 ---------

def backtest_cycle_strategy(df_raw: pd.DataFrame, params: Params) -> Dict:
    df = compute_indicators(df_raw, bb_window=params.bb_window, bb_mult=params.bb_mult)

    position = 0
    entry_price = None
    stop_price = None

    trades: List[Dict] = []
    equity = 1.0
    equity_curve = [equity]
    equity_points: List[Dict] = []

    start = max(
        df["bb_mid"].first_valid_index(),
        df["rsi"].first_valid_index(),
        df["atr"].first_valid_index(),
    )
    if start is None:
        raise ValueError("指标尚未形成，数据量不足")

    f = params.fee()
    fee_fill = f * params.leverage

    # 记录初始权益点
    equity_points.append({"time": df.iloc[start]["close_time"], "equity": equity})

    for i in range(start, len(df)):
        row = df.iloc[i]
        close = row["close"]
        upper = row["bb_upper"]
        lower = row["bb_lower"]
        mid = row["bb_mid"]
        width = row["bb_width"]
        width_slope = row["bb_width_slope"]
        rsi_v = row["rsi"]
        atr_v = row["atr"]
        ts = row["close_time"]
        sma_slope = row["sma_slope"]

        tradable = (width >= params.bb_width_min) and (width_slope >= params.width_slope_min)
        if not np.isfinite([upper, lower, mid, width, rsi_v, atr_v, sma_slope]).all():
            continue

        if position == 0:
            long_entry = (
                tradable and (sma_slope >= params.ma_slope_min) and (
                    (close <= lower) or ((close <= mid) and (rsi_v <= params.rsi_low))
                )
            )
            short_entry = (
                tradable and (sma_slope <= -params.ma_slope_min) and (
                    (close >= upper) or ((close >= mid) and (rsi_v >= params.rsi_high))
                )
            )
            if long_entry:
                position = 1
                entry_price = close
                stop_price = close - params.stop_atr * atr_v
                trades.append({"time": ts, "action": "long_entry", "price": close})
                equity *= (1 - fee_fill)
                equity_points.append({"time": ts, "equity": equity})
            elif short_entry:
                position = -1
                entry_price = close
                stop_price = close + params.stop_atr * atr_v
                trades.append({"time": ts, "action": "short_entry", "price": close})
                equity *= (1 - fee_fill)
                equity_points.append({"time": ts, "equity": equity})

        elif position == 1:
            trail = mid - params.stop_atr * atr_v
            stop_price = max(stop_price, trail) if stop_price is not None else trail

            if close <= stop_price:
                pnl = (close / entry_price - 1.0) * params.leverage
                equity *= (1.0 + pnl) * (1 - fee_fill)
                trades.append({"time": ts, "action": "long_stop", "price": close, "pnl": float(pnl)})
                position, entry_price, stop_price = 0, None, None
                equity_curve.append(equity)
                equity_points.append({"time": ts, "equity": equity})
            elif tradable and ((close >= upper) or ((close >= mid) and (rsi_v >= params.rsi_high))):
                pnl = (close / entry_price - 1.0) * params.leverage
                equity *= (1.0 + pnl) * (1 - fee_fill)
                trades.append({"time": ts, "action": "long_exit_flip", "price": close, "pnl": float(pnl)})
                position = -1
                entry_price = close
                stop_price = close + params.stop_atr * atr_v
                equity *= (1 - fee_fill)
                equity_curve.append(equity)
                equity_points.append({"time": ts, "equity": equity})

        else:  # position == -1
            trail = mid + params.stop_atr * atr_v
            stop_price = min(stop_price, trail) if stop_price is not None else trail

            if close >= stop_price:
                pnl = (entry_price / close - 1.0) * params.leverage
                equity *= (1.0 + pnl) * (1 - fee_fill)
                trades.append({"time": ts, "action": "short_stop", "price": close, "pnl": float(pnl)})
                position, entry_price, stop_price = 0, None, None
                equity_curve.append(equity)
                equity_points.append({"time": ts, "equity": equity})
            elif tradable and ((close <= lower) or ((close <= mid) and (rsi_v <= params.rsi_low))):
                pnl = (entry_price / close - 1.0) * params.leverage
                equity *= (1.0 + pnl) * (1 - fee_fill)
                trades.append({"time": ts, "action": "short_exit_flip", "price": close, "pnl": float(pnl)})
                position = 1
                entry_price = close
                stop_price = close - params.stop_atr * atr_v
                equity *= (1 - fee_fill)
                equity_curve.append(equity)
                equity_points.append({"time": ts, "equity": equity})

    exit_trades = [t for t in trades if "pnl" in t]
    wins = sum(1 for t in exit_trades if t["pnl"] > 0)
    total_return = equity - 1.0
    max_dd = 0.0
    peak = -1e9
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        "trades": trades,
        "exit_count": len(exit_trades),
        "win_rate": (wins / len(exit_trades)) if exit_trades else 0.0,
        "final_equity": equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "equity_points": equity_points,
    }

# --------- 参数优化与配置输出 ---------

def optimize_params(df: pd.DataFrame, interval: str, fee_mode_filter: str = "both", max_dd_limit: Optional[float] = None, grid: Optional[Dict[str, List]] = None) -> Tuple[Params, Dict, List[Dict]]:
    # 网格来源：优先使用配置中的自定义网格，其次使用内置默认
    rsi_low_list = (grid or {}).get("rsi_low", [30, 35])
    rsi_high_list = (grid or {}).get("rsi_high", [65, 70])
    bb_width_min_list = (grid or {}).get("bb_width_min", [0.0, 0.02, 0.03])
    width_slope_min_list = (grid or {}).get("width_slope_min", [0.0, 0.0001, 0.0002])
    stop_atr_list = (grid or {}).get("stop_atr", [1.2, 1.5])
    bb_window_list = (grid or {}).get("bb_window", [20])
    bb_mult_list = (grid or {}).get("bb_mult", [1.5, 2.0])
    leverage_list = (grid or {}).get("leverage", [5.0, 8.0, 10.0])
    ma_slope_min_list = (grid or {}).get("ma_slope_min", [0.0, 0.00005, 0.0001])

    if fee_mode_filter == "both":
        fee_modes = ["maker", "taker"]
    else:
        fee_modes = [fee_mode_filter]

    meets: List[Dict] = []
    best: Optional[Dict] = None
    best_params: Optional[Params] = None
    best_constrained: Optional[Dict] = None
    best_params_constrained: Optional[Params] = None

    for fee_mode in fee_modes:
        for lev in leverage_list:
            for rsi_low in rsi_low_list:
                for rsi_high in rsi_high_list:
                    for bbm in bb_width_min_list:
                        for wsm in width_slope_min_list:
                            for sa in stop_atr_list:
                                for bw in bb_window_list:
                                    for bm in bb_mult_list:
                                        for ms in ma_slope_min_list:
                                            p = Params(
                                                rsi_low=rsi_low,
                                                rsi_high=rsi_high,
                                                bb_width_min=bbm,
                                                width_slope_min=wsm,
                                                stop_atr=sa,
                                                fee_mode=fee_mode,
                                                leverage=lev,
                                                bb_window=bw,
                                                bb_mult=bm,
                                                ma_slope_min=ms,
                                            )
                                            res = backtest_cycle_strategy(df, p)
                                            record = {
                                                "interval": interval,
                                                "params": asdict(p),
                                                "metrics": {
                                                    "win_rate": res["win_rate"],
                                                    "total_return": res["total_return"],
                                                    "max_drawdown": res["max_drawdown"],
                                                    "exit_count": res["exit_count"],
                                                    "final_equity": res["final_equity"],
                                                },
                                            }

                                            if record["metrics"]["total_return"] >= 0.20:
                                                meets.append(record)

                                            if (best is None) or (
                                                record["metrics"]["total_return"] > best["metrics"]["total_return"]
                                                or (
                                                    math.isclose(
                                                        record["metrics"]["total_return"],
                                                        best["metrics"]["total_return"],
                                                        rel_tol=1e-9,
                                                    )
                                                    and record["metrics"]["win_rate"] > best["metrics"]["win_rate"]
                                                )
                                            ):
                                                best = record
                                                best_params = p

                                            # 受最大回撤限制的最优（若设置）
                                            if (max_dd_limit is not None) and (
                                                record["metrics"]["max_drawdown"] <= max_dd_limit
                                            ):
                                                if (best_constrained is None) or (
                                                    record["metrics"]["total_return"]
                                                    > best_constrained["metrics"]["total_return"]
                                                ):
                                                    best_constrained = record
                                                    best_params_constrained = p

    if (max_dd_limit is not None) and (best_constrained is not None):
        return best_params_constrained, best_constrained, meets
    return best_params, best, meets


def _strip_jsonc_comments(text: str) -> str:
    import re
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\n)\s*//.*?(?=\n|$)", r"\\1", text)
    return text


def load_config_defaults(path_jsonc: str = "config.jsonc", path_json: str = "config.json") -> Dict:
    # 优先读取支持注释的 JSONC；失败则回退到纯 JSON
    for p, is_jsonc in [(path_jsonc, True), (path_json, False)]:
        try:
            with open(p, "r") as f:
                raw = f.read()
            if is_jsonc:
                raw = _strip_jsonc_comments(raw)
                cfg = json.loads(raw)
            else:
                cfg = json.loads(raw)
            return cfg.get("defaults", {})
        except Exception:
            continue
    return {}


def write_config(meets: List[Dict], best: Dict, path: str = "config.jsonc", portfolio: Dict = None):
    # 保留此函数仅用于“初始化模版”用途；如果文件已存在，则不修改，避免覆盖人工配置
    try:
        with open(path, "r") as _:
            return  # 配置已存在，尊重用户手动维护，不再写入
    except FileNotFoundError:
        pass

    # 初次生成精简版 JSONC（带注释），便于首次使用
    defaults = {
        "symbol": "BTCUSDT",
        "portfolio_spec": "30m:0.7,5m:0.3",
        "fee_mode": "maker",
        "max_dd_limit": 0.10,
        "concurrent": "both",
        "init_total": 1000.0,
        "opt_grid": {
            "rsi_low": [30, 35],
            "rsi_high": [65, 70],
            "bb_width_min": [0.0, 0.02, 0.03],
            "width_slope_min": [0.0, 0.0001, 0.0002],
            "stop_atr": [1.2, 1.5],
            "bb_window": [20],
            "bb_mult": [1.5, 2.0],
            "leverage": [5.0, 8.0, 10.0],
            "ma_slope_min": [0.0, 0.00005, 0.0001]
        }
    }

    header = (
        "// 用法示例:\n"
        "// - 直接使用默认组合: python3 bb_cycle_backtest.py --portfolio \"30m:0.7,5m:0.3\" --fee_mode maker --max_dd_limit 0.10\n"
        "// - 从配置读取默认: 直接运行 python3 bb_cycle_backtest.py（脚本会读取 defaults）\n"
        "// - 单周期回测: python3 bb_cycle_backtest.py --interval 30m --fee_mode maker --max_dd_limit 0.10\n"
        "// - 切换持仓并运行: python3 bb_cycle_backtest.py --concurrent 5m_only 或 30m_only\n\n"
        "// 字段说明: 同之前注释\n"
    )

    core = {"defaults": defaults}
    jsonc_body = json.dumps(core, indent=2)
    content = header + jsonc_body + "\n"
    with open(path, "w") as f:
        f.write(content)


def write_results(meets: List[Dict], best: Dict, portfolio: Dict = None, path: str = "results.json"):
    # 写入完整结果（含 meets、equity_curve 等）到独立的结果文件；不会修改 config.jsonc
    results = {
        "criteria": {"total_return_min": 0.20},
        "best_overall": best,
        "meets": meets,
        "portfolio": portfolio,
        "generated_at": int(time.time())
    }
    with open(path, "w") as rf:
        json.dump(results, rf, indent=2)


def run_portfolio(symbol: str, spec: str, fee_mode_filter: str = "maker", max_dd_limit: Optional[float] = None, concurrent: str = "both", grid: Optional[Dict[str, List]] = None) -> Dict:
    # 解析权重
    pairs = [s.strip() for s in spec.split(",") if s.strip()]
    items: List[Tuple[str, float]] = []
    for p in pairs:
        itv, w = p.split(":")
        items.append((itv.strip(), float(w)))
    if not items:
        raise ValueError("组合 spec 为空")

    # 按 concurrent 过滤周期
    if concurrent == "5m_only":
        items = [(itv, w) for itv, w in items if itv.lower() == "5m"]
    elif concurrent == "30m_only":
        items = [(itv, w) for itv, w in items if itv.lower() == "30m"]

    # 权重归一；若只剩一个周期则权重置为 1.0
    if len(items) == 1:
        items = [(items[0][0], 1.0)]
    total_w = sum(w for _, w in items)
    items = [(itv, w / total_w) for itv, w in items]
    if len(items) == 0:
        raise ValueError("concurrent 与 spec 不匹配，未找到有效周期")
    if len(items) > 2:
        items = items[:2]  # 当前聚合器支持两个周期

    sub = []
    points = {}
    meets_all: List[Dict] = []
    best_overall: Optional[Dict] = None

    for itv, w in items:
        df = fetch_klines_paged(symbol, interval=itv, days=65)
        best_params, best, meets = optimize_params(df, itv, fee_mode_filter, max_dd_limit, grid)
        res = backtest_cycle_strategy(df, best_params)
        sub.append({
            "interval": itv,
            "weight": w,
            "params": best["params"],
            "metrics": {
                "win_rate": res["win_rate"],
                "total_return": res["total_return"],
                "max_drawdown": res["max_drawdown"],
                "exit_count": res["exit_count"],
            },
        })
        points[itv] = res.get("equity_points", [])
        meets_all.extend(meets)
        if (best_overall is None) or (best["metrics"]["total_return"] > best_overall["metrics"]["total_return"]):
            best_overall = best

    # 组合汇总
    if len(items) == 1:
        itv_a, w_a = items[0]
        eq_curve = [p["equity"] for p in points[itv_a]]
        final_equity = eq_curve[-1] if eq_curve else 1.0
        total_return = final_equity - 1.0
        peak = -1e9
        max_dd = 0.0
        for v in eq_curve:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        agg = {
            "equity_curve": eq_curve[:300],
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": max_dd,
        }
    else:
        (itv_a, w_a), (itv_b, w_b) = items
        agg = aggregate_portfolio(points[itv_a], points[itv_b], w_a, w_b, init_total=1000.0)

    portfolio = {
        "spec": spec,
        "concurrent": concurrent,
        "sub_strategies": sub,
        "combined": {
            "final_equity": agg["final_equity"],
            "total_return": agg["total_return"],
            "max_drawdown": agg["max_drawdown"],
            "equity_curve": agg["equity_curve"][:300],
        },
    }

    write_config(meets_all, best_overall, path="config.jsonc", portfolio=portfolio)
    write_results(meets_all, best_overall, portfolio=portfolio, path="results.json")
    return portfolio


def aggregate_portfolio(points_a: List[Dict], points_b: List[Dict], weight_a: float, weight_b: float, init_total: float = 1000.0) -> Dict:
    w_a_usdt = init_total * weight_a
    w_b_usdt = init_total * weight_b
    i, j = 0, 0
    eq_a, eq_b = 1.0, 1.0
    curve_total: List[float] = []
    times_total: List[str] = []

    # 初始点
    if (points_a and len(points_a) > 0) or (points_b and len(points_b) > 0):
        if points_a and points_b:
            t0 = points_a[0]["time"] if points_a[0]["time"] <= points_b[0]["time"] else points_b[0]["time"]
        elif points_a:
            t0 = points_a[0]["time"]
        else:
            t0 = points_b[0]["time"]
        total_usdt = w_a_usdt * eq_a + w_b_usdt * eq_b
        curve_total.append(total_usdt / init_total)
        times_total.append(str(t0))

    while i < len(points_a) or j < len(points_b):
        t_a = points_a[i]["time"] if i < len(points_a) else None
        t_b = points_b[j]["time"] if j < len(points_b) else None
        if t_b is None or (t_a is not None and t_a <= t_b):
            eq_a = points_a[i]["equity"]
            t = t_a
            i += 1
        else:
            eq_b = points_b[j]["equity"]
            t = t_b
            j += 1
        total_usdt = w_a_usdt * eq_a + w_b_usdt * eq_b
        curve_total.append(total_usdt / init_total)
        times_total.append(str(t))

    peak = -1e9
    max_dd = 0.0
    for v in curve_total:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    final_equity = (w_a_usdt * eq_a + w_b_usdt * eq_b) / init_total
    total_return = final_equity - 1.0
    return {
        "equity_curve": curve_total,
        "times": times_total,
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--days", type=int, default=65)
    parser.add_argument("--fee_mode", type=str, default="both", choices=["maker","taker","both"])
    parser.add_argument("--portfolio", type=str, default="", help="组合模式，例如 '30m:0.7,5m:0.3' (权重将自动归一)")
    parser.add_argument("--max_dd_limit", type=float, default=None, help="最大回撤限制（如 0.10 表示≤10%），用于挑选受限最优方案")
    parser.add_argument("--concurrent", type=str, default="both", choices=["both","5m_only","30m_only"], help="切换 5m/30m/同时持仓")

    args = parser.parse_args()

    # 从配置读取默认值（当命令行仍是其内置默认时）
    defaults = load_config_defaults()
    explicit_interval = args.interval != "15m"
    # 仅当用户未显式指定 interval 时，采用配置中的组合默认
    if args.portfolio == "" and not explicit_interval and defaults.get("portfolio_spec"):
        args.portfolio = defaults["portfolio_spec"]
    if args.fee_mode == "both" and defaults.get("fee_mode"):
        args.fee_mode = defaults["fee_mode"]
    if args.max_dd_limit is None and defaults.get("max_dd_limit") is not None:
        args.max_dd_limit = defaults["max_dd_limit"]
    if args.concurrent == "both" and defaults.get("concurrent"):
        args.concurrent = defaults["concurrent"]
    opt_grid = defaults.get("opt_grid")

    if args.portfolio:
        result = run_portfolio(args.symbol, args.portfolio, args.fee_mode, args.max_dd_limit, args.concurrent, opt_grid)
        subs = result["sub_strategies"]
        comb = result["combined"]
        if len(subs) == 1:
            # 由 defaults.concurrent 过滤只保留一个子策略时，按“单周期”格式输出
            s = subs[0]
            print("=== 单周期结果（来自 defaults.concurrent） ===")
            print(f"{s['interval']} | 胜率: {s['metrics']['win_rate']*100:.2f}% | 总回报: {s['metrics']['total_return']*100:.2f}% | 最大回撤: {s['metrics']['max_drawdown']*100:.2f}% | 期末权益: {comb['final_equity']*100:.2f}% | 交易数: {s['metrics']['exit_count']}")
        else:
            print("=== 组合结果 ===")
            for s in subs:
                print(f"{s['interval']} | 权重: {s['weight']:.2f} | 胜率: {s['metrics']['win_rate']*100:.2f}% | 总回报: {s['metrics']['total_return']*100:.2f}% | 最大回撤: {s['metrics']['max_drawdown']*100:.2f}% | 交易数: {s['metrics']['exit_count']}")
            print(f"组合 | 总回报: {comb['total_return']*100:.2f}% | 最大回撤: {comb['max_drawdown']*100:.2f}% | 期末权益: {comb['final_equity']*100:.2f}%")
    else:
        df = fetch_klines_paged(args.symbol, interval=args.interval, days=args.days)
        best_params, best, meets = optimize_params(df, args.interval, args.fee_mode, args.max_dd_limit, opt_grid)
        res = backtest_cycle_strategy(df, best_params)
        write_config(meets, best, path="config.jsonc", portfolio=None)
        write_results(meets, best, portfolio=None, path="results.json")
        print("=== 单周期结果 ===")
        print(f"{args.interval} | 胜率: {res['win_rate']*100:.2f}% | 总回报: {res['total_return']*100:.2f}% | 最大回撤: {res['max_drawdown']*100:.2f}% | 期末权益: {res['final_equity']*100:.2f}% | 交易数: {res['exit_count']}")


if __name__ == "__main__":
    main()