import os
import html
import logging
import asyncio
from typing import Dict, Any, List, Set

import requests
import pandas as pd
import ccxt

from telegram import (
    Update,
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.error import BadRequest, TimedOut

TG_TOKEN = os.getenv("TG_TOKEN", "")


PROXY_HOST = os.getenv("PROXY_HOST", "")
PROXY_PORT = os.getenv("PROXY_PORT", "")
PROXY_USER = os.getenv("PROXY_USER", "")
PROXY_PASS = os.getenv("PROXY_PASS", "")

if PROXY_HOST and PROXY_PORT:
    if PROXY_USER and PROXY_PASS:
        PROXY_URL = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
    else:
        PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
else:
    PROXY_URL = None

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
LOG = logging.getLogger("binance-oi-bot")

EXCHANGE = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})

N_DAYS = 30
OI_DAYS = 30
OI_GROWTH_PCT = 50
PRICE_MAX_PCT_UP = 50
MIN_DAILY_VOL_USD = 5_000_000
MAX_MARKETS = 200
SUBSCRIBERS: Set[int] = set()
SEEN_HITS: Set[str] = set()
LAST_AUTOSCAN_STATUS_MSG: Dict[int, int] = {}
BINANCE_OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
SORT_MODE = "oi_contracts"
SORT_MODE_LABELS = {
    "oi_contracts": "by OI growth in contracts",
    "oi_usd": "by OI growth in USD",
    "price": "by price growth",
    "volume": "by average daily volume",
}

def chunk_html(text: str, limit: int = 3500) -> List[str]:

    parts: List[str] = []
    buf = ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > limit:
            parts.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        parts.append(buf)
    return parts


def fetch_ohlc(symbol: str, limit: int = 200) -> pd.DataFrame:

    try:
        ohlc = EXCHANGE.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
        return pd.DataFrame(
            ohlc,
            columns=["time", "open", "high", "low", "close", "volume"],
        )
    except Exception as e:
        LOG.warning(f"OHLC fetch error {symbol}: {e}")
        return pd.DataFrame()


def fetch_binance_oi_history(symbol: str, days: int) -> List[Dict[str, float]]:

    days = min(days, OI_DAYS)

    params = {
        "symbol": symbol.replace("/", ""),
        "period": "1d",
        "limit": min(days + 2, 500),
    }
    try:
        r = requests.get(BINANCE_OI_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        LOG.warning(f"OI request error for {symbol}: {e}")
        return []

    out: List[Dict[str, float]] = []
    for row in data:
        try:
            out.append({
                "ts": int(row["timestamp"]),
                "oi": float(row["sumOpenInterest"]),
            })
        except Exception:
            pass

    if not out:
        return []

    out.sort(key=lambda x: x["ts"])

    if len(out) > days:
        out = out[-days:]

    return out


def get_top_usdt_futures(limit: int = MAX_MARKETS) -> List[str]:
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        LOG.warning(f"Error get_top_usdt_futures: {e}")
        return []

    items: List[tuple[str, float]] = []
    for x in data:
        sym = x.get("symbol", "")
        if sym.endswith("USDT"):
            try:
                qv = float(x.get("quoteVolume", 0))
            except Exception:
                qv = 0.0
            items.append((sym, qv))

    items.sort(key=lambda x: x[1], reverse=True)
    return [f"{s.replace('USDT', '')}/USDT" for s, _ in items[:limit]]


def linkify(symbol: str) -> str:

    s = symbol.replace("/", "")
    binance_url = f"https://www.binance.com/en/futures/{s}"
    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{s}"
    return (
        f"<b>{html.escape(symbol)}</b> "
        f"(<a href='{binance_url}'>Binance Futures</a> | "
        f"<a href='{tv_url}'>TradingView</a>)"
    )


def format_symbol_block(
    h: Dict[str, Any],
    oi_days: int,
    include_header: bool = False,
) -> str:

    lnk = linkify(h["symbol"])

    header = ""
    if include_header:
        header = f"📊 <b>Analysis for {html.escape(h['symbol'])}</b>\n\n"

    line = (
        f"{header}"
        f"• {lnk}\n"
        f"  OI (contracts): <b>{h['oi_change_pct']:.1f}%</b> "
        f"({h['oi_first']:.0f} → {h['oi_last']:.0f}) over {oi_days} days\n"
        f"  Price: <b>{h['price_change_pct']:.1f}%</b> over {N_DAYS} days\n"
        f"  Volume: <b>{h['avg_daily_vol'] / 1_000_000:.2f}M</b> USD (avg daily)\n"
    )

    if h.get("oi_usd_change_pct") is not None:
        line += (
            f"  OI (≈ USD): <b>{h['oi_usd_change_pct']:.1f}%</b> "
            f"(using asset price)\n"
        )

    return line


def analyze_symbol(symbol: str) -> Dict[str, Any]:

    df = fetch_ohlc(symbol, limit=N_DAYS + 20)
    if df.empty:
        return {"hit": False, "reason": "candles"}

    last = df.tail(N_DAYS).copy()
    if len(last) < N_DAYS:
        return {"hit": False, "reason": "few_days"}

    usd_vol = (last["volume"] * last["close"]).astype("float64")
    avg_daily_vol = float(usd_vol.mean())
    if avg_daily_vol < MIN_DAILY_VOL_USD:
        return {"hit": False, "reason": "volume"}

    try:
        price_first = float(last["close"].iloc[0])
        price_last = float(last["close"].iloc[-1])
        if price_first <= 0:
            return {"hit": False, "reason": "bad_close"}
        price_change_pct = (price_last - price_first) / price_first * 100
    except Exception:
        return {"hit": False, "reason": "price_calc"}

    if price_change_pct > PRICE_MAX_PCT_UP:
        return {"hit": False, "reason": "price_up_too_much"}

    oi_days = min(N_DAYS, OI_DAYS)
    oi = fetch_binance_oi_history(symbol, oi_days)
    if len(oi) < 2:
        return {"hit": False, "reason": "oi_short"}

    oi_first = oi[0]["oi"]
    oi_last = oi[-1]["oi"]
    if oi_first <= 0:
        return {"hit": False, "reason": "oi_zero"}

    oi_change_pct = (oi_last - oi_first) / oi_first * 100
    if oi_change_pct < OI_GROWTH_PCT:
        return {"hit": False, "reason": "oi_growth"}

    oi_usd_change_pct = None
    try:
        times = last["time"].astype("int64").tolist()

        def nearest_price(ts_ms: int) -> float:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - ts_ms))
            return float(last["close"].iloc[idx])

        p0 = nearest_price(oi[0]["ts"])
        p1 = nearest_price(oi[-1]["ts"])

        oi_usd_first = oi_first * p0
        oi_usd_last = oi_last * p1

        if oi_usd_first > 0:
            oi_usd_change_pct = (oi_usd_last - oi_usd_first) / oi_usd_first * 100
    except Exception:
        pass

    return {
        "hit": True,
        "symbol": symbol,
        "avg_daily_vol": avg_daily_vol,
        "oi_first": oi_first,
        "oi_last": oi_last,
        "oi_change_pct": oi_change_pct,
        "price_first": price_first,
        "price_last": price_last,
        "price_change_pct": price_change_pct,
        "oi_usd_change_pct": oi_usd_change_pct,
    }

def analyze_symbol_raw(symbol: str) -> Dict[str, Any]:

    df = fetch_ohlc(symbol, limit=N_DAYS + 20)
    if df.empty:
        return {"ok": False, "reason": "candles", "symbol": symbol}

    last = df.tail(N_DAYS).copy()
    if len(last) < 2:
        return {"ok": False, "reason": "few_days", "symbol": symbol}

    usd_vol = (last["volume"] * last["close"]).astype("float64")
    avg_daily_vol = float(usd_vol.mean())

    try:
        price_first = float(last["close"].iloc[0])
        price_last = float(last["close"].iloc[-1])
        if price_first <= 0:
            return {"ok": False, "reason": "bad_close", "symbol": symbol}
        price_change_pct = (price_last - price_first) / price_first * 100
    except Exception as e:
        LOG.warning(f"[RAW] Price change calc error {symbol}: {e}")
        return {"ok": False, "reason": "price_calc", "symbol": symbol}

    oi_days = min(N_DAYS, OI_DAYS)
    oi = fetch_binance_oi_history(symbol, oi_days)
    if len(oi) < 2:
        return {"ok": False, "reason": "oi_short", "symbol": symbol}

    oi_first = oi[0]["oi"]
    oi_last = oi[-1]["oi"]
    if oi_first <= 0:
        return {"ok": False, "reason": "oi_zero", "symbol": symbol}

    oi_change_pct = (oi_last - oi_first) / oi_first * 100

    oi_usd_change_pct = None
    try:
        times = last["time"].astype("int64").tolist()

        def nearest_price(ts_ms: int) -> float:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - ts_ms))
            return float(last["close"].iloc[idx])

        price_for_oi_first = nearest_price(oi[0]["ts"])
        price_for_oi_last = nearest_price(oi[-1]["ts"])

        oi_usd_first = oi_first * price_for_oi_first
        oi_usd_last = oi_last * price_for_oi_last

        if oi_usd_first > 0:
            oi_usd_change_pct = (oi_usd_last - oi_usd_first) / oi_usd_first * 100
    except Exception as e:
        LOG.warning(f"[RAW] OI USD change calc error {symbol}: {e}")
        oi_usd_change_pct = None

    return {
        "ok": True,
        "symbol": symbol,
        "avg_daily_vol": avg_daily_vol,
        "oi_first": oi_first,
        "oi_last": oi_last,
        "oi_change_pct": oi_change_pct,
        "price_first": price_first,
        "price_last": price_last,
        "price_change_pct": price_change_pct,
        "oi_usd_change_pct": oi_usd_change_pct,
    }

async def collect_hits(progress_cb=None) -> List[Dict[str, Any]]:

    syms = get_top_usdt_futures(MAX_MARKETS)

    if progress_cb:
        await progress_cb(f"⏳ Scanning {len(syms)} futures…")

    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, analyze_symbol, s) for s in syms]

    hits: List[Dict[str, Any]] = []
    processed = 0

    for coro in asyncio.as_completed(tasks):
        res = await coro
        processed += 1

        if res.get("hit"):
            hits.append(res)

        if progress_cb and processed % 10 == 0:
            await progress_cb(
                f"⏳ Processed {processed}/{len(syms)} — hits: {len(hits)}"
            )

    return hits


def sort_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    def sort_key(h: Dict[str, Any]):
        if SORT_MODE == "oi_contracts":
            return (h["oi_change_pct"], h["avg_daily_vol"])
        elif SORT_MODE == "oi_usd":
            v = h.get("oi_usd_change_pct")
            if v is None:
                v = -1e12
            return (v, h["avg_daily_vol"])
        elif SORT_MODE == "price":
            return (h["price_change_pct"], h["avg_daily_vol"])
        elif SORT_MODE == "volume":
            return (h["avg_daily_vol"], h["oi_change_pct"])
        else:
            return (h["oi_change_pct"], h["avg_daily_vol"])

    return sorted(hits, key=sort_key, reverse=True)


def build_text_from_hits(hits: List[Dict[str, Any]]) -> str:

    oi_days = min(N_DAYS, OI_DAYS)
    sort_label = SORT_MODE_LABELS.get(SORT_MODE, "by OI growth in contracts")

    header = (
        "📊 <b>Binance OI Scanner</b>\n\n"
        f"Filters (global scan):\n"
        f"• OI (contracts) growth ≥ {OI_GROWTH_PCT}% over last {oi_days} days\n"
        f"• Price growth ≤ {PRICE_MAX_PCT_UP}% over last {N_DAYS} days\n"
        f"• Average daily volume ≥ {MIN_DAILY_VOL_USD / 1_000_000:.1f}M USD\n"
        f"• Sorting: <b>{html.escape(sort_label)}</b>\n\n"
    )

    if not hits:
        return header + "No matches found."

    hits_sorted = sort_hits(hits)

    out: List[str] = [header]
    for h in hits_sorted:
        out.append(format_symbol_block(h, oi_days))

    return "\n".join(out)


async def build_report(progress_cb):
    hits = await collect_hits(progress_cb)
    return build_text_from_hits(hits)

def keyboard(chat_id: int | None = None) -> ReplyKeyboardMarkup:

    if chat_id is not None and chat_id in SUBSCRIBERS:
        autoscan_label = "🟢 Autoscan"
    else:
        autoscan_label = "🔴 Autoscan"

    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("Scan")],
            [KeyboardButton(autoscan_label)],
            [KeyboardButton("Sort OI contracts"), KeyboardButton("Sort OI USD")],
            [KeyboardButton("Sort price"), KeyboardButton("Sort volume")],
        ],
        resize_keyboard=True,
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sort_label = SORT_MODE_LABELS.get(SORT_MODE, "by OI growth in contracts")
    chat_id = update.effective_chat.id

    await update.message.reply_text(
        "Bot is ready.\n"
        "• Buttons: global scan / autoscan (single ON/OFF toggle) / sorting.\n"
        "• Send a ticker (ETH, BTC, DOT, etc.) to get raw OI & price data.\n"
        f"Current sorting: {sort_label}.",
        reply_markup=keyboard(chat_id),
        disable_web_page_preview=True,
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "⏳ Running scan…", disable_web_page_preview=True
    )

    async def progress(t: str):
        try:
            await msg.edit_text(t, parse_mode="HTML", disable_web_page_preview=True)
        except BadRequest:
            pass
        except Exception:
            pass

    text = await build_report(progress)

    for chunk in chunk_html(text):
        try:
            await update.message.reply_html(chunk, disable_web_page_preview=True)
        except TimedOut:
            LOG.warning("Timed out while sending scan result chunk, retrying once...")
            try:
                await update.message.reply_html(chunk, disable_web_page_preview=True)
            except TimedOut:
                LOG.error(
                    "Timed out again while sending scan result chunk, "
                    "giving up on this chunk."
                )
                break
        except BadRequest as e:
            LOG.warning(f"BadRequest while sending scan result chunk: {e}")
            continue
        except Exception as e:
            LOG.error(f"Unexpected error while sending scan result chunk: {e}")
            continue


async def toggle_autoscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    if chat_id in SUBSCRIBERS:
        SUBSCRIBERS.discard(chat_id)
        state_text = "🔴 Autoscan: OFF"
    else:
        SUBSCRIBERS.add(chat_id)
        state_text = "🟢 Autoscan: ON (H1)"

    await update.message.reply_text(
        state_text,
        reply_markup=keyboard(chat_id),
        disable_web_page_preview=True,
    )


async def start_autoscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await toggle_autoscan(update, context)


async def stop_autoscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.discard(chat_id)
    await update.message.reply_text(
        "🔴 Autoscan: OFF",
        reply_markup=keyboard(chat_id),
        disable_web_page_preview=True,
    )


async def cmd_oireset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SEEN_HITS.clear()
    await update.message.reply_text(
        "♻️ OI signal memory cleared. New signals will be sent again.",
        disable_web_page_preview=True,
    )


async def _delete_prev_status_if_any(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
):

    msg_id = LAST_AUTOSCAN_STATUS_MSG.get(chat_id)
    if not msg_id:
        return
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
    except BadRequest as e:

        LOG.info(f"Cannot delete previous autoscan status msg in {chat_id}: {e}")
    except Exception as e:
        LOG.info(f"Error deleting previous autoscan status msg in {chat_id}: {e}")
    finally:

        LAST_AUTOSCAN_STATUS_MSG.pop(chat_id, None)


async def periodic_scan(context: ContextTypes.DEFAULT_TYPE):

    if not SUBSCRIBERS:
        return

    hits = await collect_hits(progress_cb=None)
    oi_days = min(N_DAYS, OI_DAYS)

    new_hits = [h for h in hits if h["symbol"] not in SEEN_HITS]

    if new_hits:
        for h in new_hits:
            SEEN_HITS.add(h["symbol"])

        new_hits_sorted = sort_hits(new_hits)

        lines: List[str] = []
        for h in new_hits_sorted:
            lines.append(format_symbol_block(h, oi_days))
        body = "\n".join(lines)

        text = (
            "🟢 Autoscan (H1) — NEW signals\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            + body
        )

        for chat_id in list(SUBSCRIBERS):

            await _delete_prev_status_if_any(context, chat_id)

            for chunk in chunk_html(text):
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                except TimedOut:
                    LOG.warning(
                        f"Timed out while sending autoscan chunk to {chat_id}, "
                        f"retrying once..."
                    )
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except TimedOut:
                        LOG.error(
                            f"Timed out again while sending autoscan chunk to {chat_id}, "
                            f"skipping this chunk."
                        )
                        break
                except BadRequest as e:
                    LOG.warning(
                        f"BadRequest while sending autoscan chunk to {chat_id}: {e}"
                    )
                    break
                except Exception as e:
                    LOG.error(
                        f"Unexpected error while sending autoscan chunk to {chat_id}: {e}"
                    )
                    break
        return
    status_text = (
        "🟢 Autoscan (H1) — running, no new signals\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )

    for chat_id in list(SUBSCRIBERS):
        await _delete_prev_status_if_any(context, chat_id)

        try:
            m = await context.bot.send_message(
                chat_id=chat_id,
                text=status_text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )

            LAST_AUTOSCAN_STATUS_MSG[chat_id] = m.message_id
        except TimedOut:
            LOG.warning(
                f"Timed out while sending autoscan status to {chat_id}, retrying once..."
            )
            try:
                m = await context.bot.send_message(
                    chat_id=chat_id,
                    text=status_text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                LAST_AUTOSCAN_STATUS_MSG[chat_id] = m.message_id
            except Exception as e:
                LOG.error(f"Failed to send autoscan status to {chat_id}: {e}")
        except Exception as e:
            LOG.error(f"Unexpected error while sending autoscan status to {chat_id}: {e}")


def normalize_symbol_text(user_text: str) -> str:

    t = user_text.strip().upper().replace(" ", "")

    if "/" in t:
        return t

    if t.endswith("USDT"):
        base = t[:-4]
        return f"{base}/USDT"

    return f"{t}/USDT"


async def handle_manual_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = update.message.text.strip()
    symbol = normalize_symbol_text(raw)

    await update.message.reply_text(
        f"⏳ Checking {symbol} (raw, no filters)…",
        disable_web_page_preview=True,
    )

    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, analyze_symbol_raw, symbol)

    if not res.get("ok"):
        reason = res.get("reason", "unknown")
        reason_map = {
            "candles": "no candle data available.",
            "few_days": "not enough days of data.",
            "bad_close": "invalid close price data.",
            "price_calc": "error while calculating price change.",
            "oi_short": "not enough OI history.",
            "oi_zero": "zero initial OI value.",
        }
        text = reason_map.get(
            reason,
            f"failed to compute metrics (reason={reason}).",
        )
        await update.message.reply_text(
            f"{symbol}: {text}",
            disable_web_page_preview=True,
        )
        return

    oi_days = min(N_DAYS, OI_DAYS)
    block = format_symbol_block(res, oi_days, include_header=True)
    await update.message.reply_html(block, disable_web_page_preview=True)


async def handle_sort_button(update: Update, txt: str):

    global SORT_MODE

    txt = txt.lower()
    if txt == "sort oi contracts":
        SORT_MODE = "oi_contracts"
    elif txt == "sort oi usd":
        SORT_MODE = "oi_usd"
    elif txt == "sort price":
        SORT_MODE = "price"
    elif txt == "sort volume":
        SORT_MODE = "volume"
    else:
        return

    sort_label = SORT_MODE_LABELS.get(SORT_MODE, "by OI growth in contracts")
    await update.message.reply_text(
        f"Sorting mode changed to: {sort_label}.",
        disable_web_page_preview=True,
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    txt_raw = update.message.text or ""
    txt = txt_raw.strip().lower()

    if txt == "scan":
        return await cmd_scan(update, context)

    if "autoscan" in txt:
        return await toggle_autoscan(update, context)

    if txt.startswith("sort "):
        return await handle_sort_button(update, txt)

    # Otherwise treat it as a ticker
    return await handle_manual_symbol(update, context)

def main():
    from datetime import datetime, timedelta

    if not TG_TOKEN:
        LOG.error(
            "TG_TOKEN environment variable is not set. "
            "Please export TG_TOKEN before running the bot."
        )
        raise SystemExit(1)

    def seconds_to_next_hour() -> int:
        now = datetime.utcnow()
        next_hour = (now + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0
        )
        return int((next_hour - now).total_seconds())

    builder = Application.builder().token(TG_TOKEN)

    if PROXY_URL:
        builder.proxy(PROXY_URL)

    app = builder.build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("autostart", start_autoscan))
    app.add_handler(CommandHandler("stop", stop_autoscan))
    app.add_handler(CommandHandler("stopscan", stop_autoscan))
    app.add_handler(CommandHandler("autoscan", start_autoscan))
    app.add_handler(CommandHandler("oireset", cmd_oireset))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    app.job_queue.run_repeating(
        periodic_scan,
        interval=3600,
        first=seconds_to_next_hour(),
    )

    LOG.info("BOT STARTED")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
