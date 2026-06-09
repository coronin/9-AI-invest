"""
AI投研团队 · CrewAI Tools
将外部数据获取函数封装为 CrewAI 可调用的 Tool
"""
from crewai.tools import tool
import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@tool("get_stock_info")
def get_stock_info_tool(code: str) -> str:
    """
    获取股票/基金的基本信息。

    Args:
        code: 股票代码，A股如 "600519.SS"（贵州茅台）、"000001.SZ"（平安银行），
              港股如 "0700.HK"（腾讯控股），美股如 "AAPL"（苹果）、"TSLA"（特斯拉）

    Returns:
        股票基本信息字典，包含：name(名称)、sector(行业)、industry(子行业)、
        marketCap(市值)、trailingPE(滚动PE)、forwardPE(预期PE)、
        dividendYield(股息率)、currentPrice(当前价)、fiftyTwoWeekHigh(52周高点)、
        fiftyTwoWeekLow(52周低点)、volume(成交量)、averageVolume(日均成交量) 等

    失败时返回:
        "无法获取该股票信息，请检查代码是否正确，或稍后重试。"
    """
    try:
        ticker = yf.Ticker(code)
        info = ticker.info
        if not info:
            return f"无法获取 {code} 的信息，可能代码不正确或该股票不存在。"

        keys = [
            "shortName", "longName", "sector", "industry",
            "marketCap", "trailingPE", "forwardPE",
            "dividendYield", "trailingEps", "forwardEps",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "currentPrice", "previousClose",
            "volume", "averageVolume",
        ]
        result = {k: info.get(k) for k in keys if k in info}
        if "shortName" in result:
            result["name"] = result.pop("shortName")
        elif "longName" in result:
            result["name"] = result.pop("longName")

        if not result:
            return f"无法获取 {code} 的详细信息，信息字段为空。"

        return str(result)
    except Exception as e:
        logger.error(f"[get_stock_info] Failed for {code}: {e}")
        return f"获取 {code} 信息失败：{str(e)}。请检查代码是否正确，或网络连接是否正常。"


@tool("get_stock_history")
def get_stock_history_tool(code: str, days: int = 90) -> str:
    """
    获取股票/指数的历史价格数据。

    Args:
        code: 股票代码，如 "600519.SS"、"AAPL"、"^GSPC"（标普500）
        days: 历史天数，默认90天，最大不超过730天

    Returns:
        JSON格式的历史数据，包含日期、开盘价、收盘价、最高价、最低价、成交量。
        格式示例：[{"date": "2026-01-01", "open": 1800.0, "high": 1850.0, "low": 1790.0, "close": 1840.0, "volume": 3000000}, ...]

    失败时返回:
        "无法获取该股票的历史数据，请检查代码是否正确，或稍后重试。"
    """
    try:
        import datetime
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=min(days, 730))

        ticker = yf.Ticker(code)
        df = ticker.history(start=str(start_date), end=str(end_date))

        if df.empty:
            return f"无法获取 {code} 的历史数据，可能代码不正确或该时间段内无交易数据。"

        records = []
        for idx, row in df.iterrows():
            records.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            })
        return str(records)
    except Exception as e:
        logger.error(f"[get_stock_history] Failed for {code}: {e}")
        return f"获取 {code} 历史数据失败：{str(e)}。请检查代码是否正确，或网络连接是否正常。"


@tool("get_stock_performance")
def get_stock_performance_tool(code: str, days: int = 30) -> str:
    """
    获取股票近N日表现统计。

    Args:
        code: 股票代码
        days: 统计天数，默认30天

    Returns:
        性能统计字典，包含：
        - change_pct: 区间涨跌幅百分比
        - volume_avg: 日均成交量
        - price_range: (最低价, 最高价) 元组

    失败时返回:
        "无法获取该股票表现数据，请检查代码是否正确，或稍后重试。"
    """
    try:
        import datetime
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days * 2)

        ticker = yf.Ticker(code)
        df = ticker.history(start=str(start_date), end=str(end_date))

        if df.empty or len(df) < 2:
            return f"无法获取 {code} 的表现数据，历史数据不足。"

        first_price = df["Close"].iloc[0]
        last_price = df["Close"].iloc[-1]
        change_pct = (last_price - first_price) / first_price * 100 if first_price else 0.0
        volume_avg = df["Volume"].mean()
        price_low = df["Low"].min()
        price_high = df["High"].max()

        return str({
            "change_pct": round(change_pct, 4),
            "volume_avg": round(volume_avg, 2),
            "price_range": (round(float(price_low), 2), round(float(price_high), 2)),
        })
    except Exception as e:
        logger.error(f"[get_stock_performance] Failed for {code}: {e}")
        return f"获取 {code} 表现数据失败：{str(e)}。请检查代码是否正确，或网络连接是否正常。"