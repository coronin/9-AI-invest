"""
AI投研团队 · 金融数据获取模块
"""
import yfinance as yf
from typing import Optional, Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_stock_data(code: str, start: str, end: str) -> pd.DataFrame:
    """
    获取股票/指数数据
    code: 股票代码，如 "600519.SS"（茅台）、"000001.SZ"（平安）
          或美股如 "AAPL", "MSFT"
    start: 开始日期 "YYYY-MM-DD"
    end: 结束日期 "YYYY-MM-DD"
    返回: OHLCV DataFrame
    """
    try:
        ticker = yf.Ticker(code)
        df = ticker.history(start=start, end=end)
        if df.empty:
            logger.warning(f"[data_fetcher] No data for {code} from {start} to {end}")
        return df
    except Exception as e:
        logger.error(f"[data_fetcher] Failed to get stock data for {code}: {e}")
        return pd.DataFrame()


def get_index_data(code: str, start: str, end: str) -> pd.DataFrame:
    """
    获取指数数据
    code: 如 "^GSPC"（标普500）, "^HSI"（恒生）
    """
    try:
        ticker = yf.Ticker(code)
        df = ticker.history(start=start, end=end)
        if df.empty:
            logger.warning(f"[data_fetcher] No index data for {code} from {start} to {end}")
        return df
    except Exception as e:
        logger.error(f"[data_fetcher] Failed to get index data for {code}: {e}")
        return pd.DataFrame()


def get_stock_info(code: str) -> Dict:
    """
    获取股票基本信息
    返回: {"name", "sector", "industry", "marketCap", "peRatio", "dividendYield", ...}
    """
    try:
        ticker = yf.Ticker(code)
        info = ticker.info
        if not info:
            logger.warning(f"[data_fetcher] No info for {code}")
            return {}
        keys = [
            "shortName", "longName", "sector", "industry",
            "marketCap", "trailingPE", "forwardPE",
            "dividendYield", "trailingEps", "forwardEps",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "currentPrice", "previousClose",
            "volume", "averageVolume",
        ]
        result = {k: info.get(k) for k in keys if k in info}
        # 兼容命名
        if "shortName" in result:
            result["name"] = result.pop("shortName")
        elif "longName" in result:
            result["name"] = result.pop("longName")
        return result
    except Exception as e:
        logger.error(f"[data_fetcher] Failed to get stock info for {code}: {e}")
        return {}


def get_recent_performance(code: str, days: int = 30) -> Dict:
    """
    获取近N日表现
    返回: {"change_pct": float, "volume_avg": float, "price_range": (low, high)}
    """
    try:
        ticker = yf.Ticker(code)
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=days * 2)
        df = ticker.history(start=str(start_date.date()), end=str(end_date.date()))

        if df.empty or len(df) < 2:
            logger.warning(f"[data_fetcher] Insufficient data for recent performance of {code}")
            return {
                "change_pct": 0.0,
                "volume_avg": 0.0,
                "price_range": (0.0, 0.0),
            }

        first_price = df["Close"].iloc[0]
        last_price = df["Close"].iloc[-1]
        change_pct = (last_price - first_price) / first_price * 100 if first_price else 0.0
        volume_avg = df["Volume"].mean()
        price_low = df["Low"].min()
        price_high = df["High"].max()

        return {
            "change_pct": round(change_pct, 4),
            "volume_avg": round(volume_avg, 2),
            "price_range": (round(price_low, 2), round(price_high, 2)),
        }
    except Exception as e:
        logger.error(f"[data_fetcher] Failed to get recent performance for {code}: {e}")
        return {
            "change_pct": 0.0,
            "volume_avg": 0.0,
            "price_range": (0.0, 0.0),
        }
