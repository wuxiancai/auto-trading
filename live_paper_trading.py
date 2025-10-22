#!/usr/bin/env python3
"""
Live Paper Trading for BTCUSDT Perpetual using Binance Futures WebSocket (kline)
- Default capital: 1000 USDT
- Default leverage: 10x
- Signals: Reuse Bollinger-band cycle strategy (same rules as backtest)
- Interval: configurable (default 5m) — decisions are made at each candle close

This module subscribes to Binance USDT-M futures kline stream and simulates
trades in real-time. It prints trades, keeps running equity, and writes results
to paper_results.json on each update. Config files are not modified.

Note: This is for research/demo only and does NOT constitute investment advice.
"""
import json
import time
import argparse
import threading
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import websocket  # pip install websocket-client

# Reuse indicator computation and data fetch from backtest module
from bb_cycle_backtest import compute_indicators, Params, fetch_klines_paged

WS_BASE = "wss://fstream.binance.com/stream?streams={stream}"


def _ts_ms() -> int:
    return int(time.time() * 1000)


class LivePaperTrader:
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "5m", init_total: float = 1000.0, leverage: float = 10.0, fee_mode: str = "maker"):
        self.symbol = symbol.upper()
        self.interval = interval
        self.init_total = float(init_total)
        self.params = Params(
            fee_mode=fee_mode,
            leverage=float(leverage),
        )
        # Keep other Params at defaults; you may tune thresholds later if needed
        self.fee_fill = self.params.fee() * self.params.leverage

        self.df: pd.DataFrame = pd.DataFrame(columns=[
            "open_time","open","high","low","close","volume","close_time"
        ])

        # Trading state
        self.position = 0  # 0/1/-1
        self.entry_price: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.equity = 1.0  # normalized; absolute USDT = equity * init_total
        self.trades: List[Dict] = []
        self.equity_points: List[Dict] = []

        # Control
        self.ws_app: Optional[websocket.WebSocketApp] = None
        self.stop_flag = threading.Event()
        self._prefetched = False

    def prefetch_history(self, days: int = 10):
        """Load some recent history so indicators form immediately."""
        try:
            df = fetch_klines_paged(self.symbol, self.interval, days)
            self.df = df[["open_time","open","high","low","close","volume","close_time"]].copy()
            self._prefetched = True
        except Exception as e:
            print(f"[Prefetch] failed: {e}")
            self._prefetched = False

    def _append_kline(self, k: Dict):
        """Append a finalized kline from WS to the dataframe."""
        # kline payload per Binance futures:
        # k = {
        #   't': openTime, 'T': closeTime, 'o': '...', 'h': '...', 'l': '...', 'c': '...', 'v': '...', 'x': isFinal,
        # }
        row = {
            "open_time": pd.to_datetime(int(k["t"]), unit="ms"),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k.get("v", 0.0)),
            "close_time": pd.to_datetime(int(k["T"]), unit="ms"),
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    def _eval_latest(self):
        """Evaluate strategy on the latest closed candle."""
        df = compute_indicators(self.df, bb_window=self.params.bb_window, bb_mult=self.params.bb_mult)
        # Find first valid index so indicators are formed
        start_idx = max(
            df["bb_mid"].first_valid_index(),
            df["rsi"].first_valid_index(),
            df["atr"].first_valid_index(),
        )
        if start_idx is None:
            return
        # For live, only evaluate on the latest bar
        i = len(df) - 1
        if i < start_idx:
            return
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

        if not np.isfinite([upper, lower, mid, width, rsi_v, atr_v, sma_slope]).all():
            return

        tradable = (width >= self.params.bb_width_min) and (width_slope >= self.params.width_slope_min)

        def _record_equity():
            self.equity_points.append({"time": ts, "equity": self.equity})

        if self.position == 0:
            long_entry = (
                tradable and (sma_slope >= self.params.ma_slope_min) and (
                    (close <= lower) or ((close <= mid) and (rsi_v <= self.params.rsi_low))
                )
            )
            short_entry = (
                tradable and (sma_slope <= -self.params.ma_slope_min) and (
                    (close >= upper) or ((close >= mid) and (rsi_v >= self.params.rsi_high))
                )
            )
            if long_entry:
                self.position = 1
                self.entry_price = close
                self.stop_price = close - self.params.stop_atr * atr_v
                self.equity *= (1 - self.fee_fill)
                self.trades.append({"time": str(ts), "action": "long_entry", "price": float(close)})
                print(f"[LONG] entry @ {close:.2f} | equity {self.equity*self.init_total:.2f} USDT")
                _record_equity()
            elif short_entry:
                self.position = -1
                self.entry_price = close
                self.stop_price = close + self.params.stop_atr * atr_v
                self.equity *= (1 - self.fee_fill)
                self.trades.append({"time": str(ts), "action": "short_entry", "price": float(close)})
                print(f"[SHORT] entry @ {close:.2f} | equity {self.equity*self.init_total:.2f} USDT")
                _record_equity()

        elif self.position == 1:
            trail = mid - self.params.stop_atr * atr_v
            self.stop_price = max(self.stop_price, trail) if self.stop_price is not None else trail

            if close <= self.stop_price:
                pnl = (close / self.entry_price - 1.0) * self.params.leverage
                self.equity *= (1.0 + pnl) * (1 - self.fee_fill)
                self.trades.append({"time": str(ts), "action": "long_stop", "price": float(close), "pnl": float(pnl)})
                print(f"[LONG] stop @ {close:.2f} | pnl {pnl*100:.2f}% | equity {self.equity*self.init_total:.2f} USDT")
                self.position, self.entry_price, self.stop_price = 0, None, None
                _record_equity()
            elif tradable and ((close >= upper) or ((close >= mid) and (rsi_v >= self.params.rsi_high))):
                pnl = (close / self.entry_price - 1.0) * self.params.leverage
                self.equity *= (1.0 + pnl) * (1 - self.fee_fill)
                self.trades.append({"time": str(ts), "action": "long_exit_flip", "price": float(close), "pnl": float(pnl)})
                print(f"[LONG->SHORT] exit/flip @ {close:.2f} | pnl {pnl*100:.2f}% | equity {self.equity*self.init_total:.2f} USDT")
                self.position = -1
                self.entry_price = close
                self.stop_price = close + self.params.stop_atr * atr_v
                self.equity *= (1 - self.fee_fill)
                _record_equity()

        else:  # self.position == -1
            trail = mid + self.params.stop_atr * atr_v
            self.stop_price = min(self.stop_price, trail) if self.stop_price is not None else trail

            if close >= self.stop_price:
                pnl = (self.entry_price / close - 1.0) * self.params.leverage
                self.equity *= (1.0 + pnl) * (1 - self.fee_fill)
                self.trades.append({"time": str(ts), "action": "short_stop", "price": float(close), "pnl": float(pnl)})
                print(f"[SHORT] stop @ {close:.2f} | pnl {pnl*100:.2f}% | equity {self.equity*self.init_total:.2f} USDT")
                self.position, self.entry_price, self.stop_price = 0, None, None
                _record_equity()
            elif tradable and ((close <= lower) or ((close <= mid) and (rsi_v <= self.params.rsi_low))):
                pnl = (self.entry_price / close - 1.0) * self.params.leverage
                self.equity *= (1.0 + pnl) * (1 - self.fee_fill)
                self.trades.append({"time": str(ts), "action": "short_exit_flip", "price": float(close), "pnl": float(pnl)})
                print(f"[SHORT->LONG] exit/flip @ {close:.2f} | pnl {pnl*100:.2f}% | equity {self.equity*self.init_total:.2f} USDT")
                self.position = 1
                self.entry_price = close
                self.stop_price = close - self.params.stop_atr * atr_v
                self.equity *= (1 - self.fee_fill)
                _record_equity()

    def save_results(self, path: str = "paper_results.json"):
        # Compute basic equity stats
        eq_curve = [p["equity"] for p in self.equity_points]
        final_equity = eq_curve[-1] if eq_curve else self.equity
        # max drawdown
        peak, max_dd = -1e9, 0.0
        for v in eq_curve:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        out = {
            "symbol": self.symbol,
            "interval": self.interval,
            "init_total": self.init_total,
            "params": self.params.__dict__,
            "trades": self.trades,
            "equity_points": self.equity_points[-300:],
            "final_equity": final_equity,
            "final_usdt": final_equity * self.init_total,
            "max_drawdown": max_dd,
            "generated_at": int(time.time()),
            # extra fields for Web monitor
            "position": self.position,
            "entry_price": float(self.entry_price) if self.entry_price is not None else None,
            "equity_norm": self.equity,
            "fee_mode": self.params.fee_mode,
            "leverage": self.params.leverage,
            "trade_count": len(self.trades),
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    # --------- WebSocket handling ---------
    def _on_message(self, _ws, message: str):
        try:
            payload = json.loads(message)
            data = payload.get("data") or payload
            k = (data or {}).get("k")
            if not k:
                return
            # Only act on finalized candles
            if bool(k.get("x", False)):
                self._append_kline(k)
                self._eval_latest()
                self.save_results()
        except Exception as e:
            print(f"[WS message] error: {e}")

    def _on_open(self, _ws):
        print("[WS] connected")

    def _on_close(self, _ws, code, reason):
        print(f"[WS] closed: {code} {reason}")

    def _on_error(self, _ws, error):
        print(f"[WS] error: {error}")

    def _build_stream(self) -> str:
        return f"{self.symbol.lower()}@kline_{self.interval}"

    def connect_and_run(self):
        # Prefetch history for indicators
        if not self._prefetched:
            self.prefetch_history(days=10)
        stream = self._build_stream()
        url = WS_BASE.format(stream=stream)
        while not self.stop_flag.is_set():
            try:
                self.ws_app = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[WS] run_forever error: {e}")
            if self.stop_flag.is_set():
                break
            print("[WS] reconnecting in 3s...")
            time.sleep(3)

    def stop(self):
        self.stop_flag.set()
        try:
            if self.ws_app:
                self.ws_app.close()
        except Exception:
            pass
        self.save_results()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="5m")
    parser.add_argument("--init_total", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--fee_mode", type=str, default="maker", choices=["maker","taker"])
    args = parser.parse_args()

    trader = LivePaperTrader(
        symbol=args.symbol,
        interval=args.interval,
        init_total=args.init_total,
        leverage=args.leverage,
        fee_mode=args.fee_mode,
    )

    print(f"Starting live paper trading: {args.symbol} {args.interval} | init {args.init_total} USDT | lev {args.leverage}x | fee {args.fee_mode}")
    try:
        trader.connect_and_run()
    except KeyboardInterrupt:
        print("[MAIN] interrupted, stopping...")
        trader.stop()


if __name__ == "__main__":
    main()