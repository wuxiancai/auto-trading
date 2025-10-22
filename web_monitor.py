#!/usr/bin/env python3
"""
Simple Web Monitor (Flask) for Live Paper Trading
- Port: 5000
- Reads paper_results.json produced by live_paper_trading.py
- Fetches latest price and recent klines from Binance Futures REST
- Displays dashboard panels similar to the provided screenshot
"""
import json
import time
from typing import Dict, Any

import requests
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
RESULT_PATH = "paper_results.json"

HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>实盘监控面板</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #f5f7fb; }
    header { background: #1f3a63; color: #fff; padding: 12px 16px; }
    .wrap { padding: 16px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; grid-gap: 12px; }
    .card { background: #fff; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); padding: 12px; }
    .title { font-weight: 600; margin-bottom: 8px; color: #333; }
    .row { display: flex; justify-content: space-between; margin: 6px 0; }
    .muted { color: #666; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; }
    th { background: #fafafa; }
    .pos-long { color: #0a8f08; font-weight: 600; }
    .pos-short { color: #d83a2e; font-weight: 600; }
    .metric { font-weight: 600; }
  </style>
</head>
<body>
<header>实盘监控面板 · BTCUSDT</header>
<div class="wrap">
  <div class="grid">
    <div class="card">
      <div class="title">系统参数</div>
      <div id="sys"></div>
    </div>
    <div class="card">
      <div class="title">实时价格与均线</div>
      <div id="ind"></div>
    </div>
    <div class="card">
      <div class="title">当前持仓与账户</div>
      <div id="pos"></div>
    </div>
    <div class="card">
      <div class="title">最近交易</div>
      <table id="trades"></table>
    </div>
    <div class="card" style="grid-column: span 2;">
      <div class="title">最近K线</div>
      <table id="klines"></table>
    </div>
  </div>
</div>
<script>
async function loadState() {
  const r = await fetch('/api/state');
  const s = await r.json();
  // 系统参数
  document.getElementById('sys').innerHTML = `
    <div class='row'><span class='muted'>交易品种</span><span class='metric'>${s.symbol}</span></div>
    <div class='row'><span class='muted'>周期</span><span>${s.interval}</span></div>
    <div class='row'><span class='muted'>杠杆</span><span>${s.leverage}x</span></div>
    <div class='row'><span class='muted'>费用模式</span><span>${s.fee_mode}</span></div>
    <div class='row'><span class='muted'>初始资金</span><span>${s.init_total.toFixed(2)} USDT</span></div>
  `;
  // 指标
  document.getElementById('ind').innerHTML = `
    <div class='row'><span class='muted'>现价</span><span class='metric'>${s.price.toFixed(2)}</span></div>
    <div class='row'><span class='muted'>涨跌</span><span>${(s.change_pct*100).toFixed(2)}%</span></div>
    <div class='row'><span class='muted'>EMA(10)</span><span>${s.ema10.toFixed(2)}</span></div>
    <div class='row'><span class='muted'>MA(30)</span><span>${s.ma30.toFixed(2)}</span></div>
  `;
  // 持仓与账户
  const posClass = (s.position === 1) ? 'pos-long' : (s.position === -1 ? 'pos-short' : '');
  const posText = (s.position === 1) ? '做多' : (s.position === -1 ? '做空' : '空仓');
  document.getElementById('pos').innerHTML = `
    <div class='row'><span class='muted'>方向</span><span class='${posClass}'>${posText}</span></div>
    <div class='row'><span class='muted'>开仓价</span><span>${s.entry_price ? s.entry_price.toFixed(2) : '-'}</span></div>
    <div class='row'><span class='muted'>未实现盈亏</span><span>${(s.unrealized_pnl*100).toFixed(2)}%</span></div>
    <div class='row'><span class='muted'>账户权益</span><span>${s.final_usdt.toFixed(2)} USDT</span></div>
    <div class='row'><span class='muted'>最大回撤</span><span>${(s.max_dd*100).toFixed(2)}%</span></div>
    <div class='row'><span class='muted'>交易次数</span><span>${s.trade_count}</span></div>
  `;
  // 交易
  const th = `<tr><th>时间</th><th>方向/动作</th><th>价格</th><th>PnL</th></tr>`;
  const rows = s.trades.slice(-20).reverse().map(t => `
    <tr>
      <td>${t.time}</td>
      <td>${t.action}</td>
      <td>${t.price ? Number(t.price).toFixed(2) : '-'}</td>
      <td>${t.pnl !== undefined ? (Number(t.pnl)*100).toFixed(2)+'%' : '-'}</td>
    </tr>`).join('');
  document.getElementById('trades').innerHTML = th + rows;
  // K线
  const kh = `<tr><th>收盘时间</th><th>开</th><th>高</th><th>低</th><th>收</th><th>量</th></tr>`;
  const krows = s.klines.map(k => `
    <tr>
      <td>${k.close_time}</td>
      <td>${k.open.toFixed(2)}</td>
      <td>${k.high.toFixed(2)}</td>
      <td>${k.low.toFixed(2)}</td>
      <td>${k.close.toFixed(2)}</td>
      <td>${k.volume.toFixed(2)}</td>
    </tr>
  `).join('');
  document.getElementById('klines').innerHTML = kh + krows;
}

loadState();
setInterval(loadState, 5000);
</script>
</body>
</html>
"""

app = Flask(__name__)


def _load_results() -> Dict[str, Any]:
    try:
        with open(RESULT_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _get_price(symbol: str) -> float:
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/ticker/price"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


def _get_klines(symbol: str, interval: str = "5m", limit: int = 50):
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore",
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/state")
def api_state():
    res = _load_results()
    symbol = res.get("symbol", "BTCUSDT")
    interval = res.get("interval", "5m")
    init_total = float(res.get("init_total", 1000.0))
    fee_mode = res.get("fee_mode", "maker")
    leverage = float(res.get("leverage", 10.0))
    position = int(res.get("position", 0))
    entry_price = res.get("entry_price")
    final_usdt = float(res.get("final_usdt", init_total))
    max_dd = float(res.get("max_drawdown", 0.0))
    trade_count = int(res.get("trade_count", 0))

    price = _get_price(symbol)
    # 默认返回最新 5 根 K 线；支持通过 /api/state?limit=10 调整
    kl_limit = int(request.args.get("limit", 5))
    kl = _get_klines(symbol, interval, limit=max(kl_limit, 5))
    # EMA10 / MA30
    ema10 = kl["close"].ewm(span=10, adjust=False).mean().iloc[-1]
    ma30 = kl["close"].rolling(30).mean().iloc[-1]
    # 涨跌：最后一根 vs 前一根收盘
    change_pct = (kl["close"].iloc[-1] / kl["close"].iloc[-2] - 1.0)

    # 未实现盈亏（百分比），基于入场价与当前价
    unrealized_pnl = 0.0
    if position == 1 and entry_price:
        unrealized_pnl = (price / float(entry_price) - 1.0) * leverage
    elif position == -1 and entry_price:
        unrealized_pnl = (float(entry_price) / price - 1.0) * leverage

    # 最近 N 根表格：按 close_time 倒序（最新在前），默认 5 根
    ktab = kl.tail(kl_limit).copy().sort_values("close_time", ascending=False)
    ktab_rows = [{
        "close_time": str(ct),
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "volume": float(v),
    } for ct, o, h, l, c, v in zip(
        ktab["close_time"], ktab["open"], ktab["high"], ktab["low"], ktab["close"], ktab["volume"]
    )]

    return jsonify({
        "symbol": symbol,
        "interval": interval,
        "init_total": init_total,
        "fee_mode": fee_mode,
        "leverage": leverage,
        "position": position,
        "entry_price": entry_price,
        "final_usdt": final_usdt,
        "max_dd": max_dd,
        "trade_count": trade_count,
        "price": price,
        "ema10": float(ema10) if pd.notna(ema10) else price,
        "ma30": float(ma30) if pd.notna(ma30) else price,
        "change_pct": float(change_pct),
        "unrealized_pnl": float(unrealized_pnl),
        "trades": res.get("trades", [])[-50:],
        "klines": ktab_rows,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002)