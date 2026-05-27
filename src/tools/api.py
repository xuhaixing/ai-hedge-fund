import logging

import pandas as pd
from langchain_core.tools import tool

from src.client import get_client
from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

logger = logging.getLogger(__name__)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or data source."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    client = get_client()
    prices = client.get_prices(ticker, start_date, end_date)

    if not prices:
        return []

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def _get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics, returns typed FinancialMetrics objects for agent code."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    client = get_client()
    financial_metrics = client.get_financial_metrics(ticker, end_date, period, limit)

    if not financial_metrics:
        return []

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    logger.debug("get_financial_metrics(%s, %s, period=%s, limit=%s) => %s", ticker, end_date, period, limit, financial_metrics)
    return financial_metrics


@tool
def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[dict]:
    """Fetch key financial metrics for a stock ticker from cache or data source.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA", "AAPL".
        end_date: Fetch metrics up to this date (inclusive), format "YYYY-MM-DD".
        period: Reporting period. One of:
            - "ttm"       Trailing Twelve Months (most recent rolling 12 months)
            - "annual"    Annual financial statements (fixed fiscal year)
            - "quarterly" Quarterly financial statements
        limit: Maximum number of periods to return (default 10).

    Returns:
        List of financial metric records, each containing ratios such as
        gross_margin, pe_ratio, debt_to_equity, revenue_growth, etc.
    """
    return [m.model_dump() for m in _get_financial_metrics(ticker, end_date, period, limit)]


def _search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items, returns typed LineItem objects for agent code."""
    client = get_client()
    return client.search_line_items(ticker, line_items, end_date, period, limit)


@tool
def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[dict]:
    """Search and fetch specific financial line items for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA", "AAPL".
        line_items: List of financial line item names to retrieve, e.g.
            ["revenue", "net_income", "free_cash_flow", "total_debt",
             "capital_expenditure", "earnings_per_share"].
        end_date: Fetch data up to this date (inclusive), format "YYYY-MM-DD".
        period: Reporting period. One of:
            - "ttm"       Trailing Twelve Months
            - "annual"    Annual financial statements
            - "quarterly" Quarterly financial statements
        limit: Maximum number of periods to return (default 10).

    Returns:
        List of line item records keyed by the requested field names.
    """
    return [item.model_dump() for item in _search_line_items(ticker, line_items, end_date, period, limit)]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or data source."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    client = get_client()
    all_trades = client.get_insider_trades(ticker, end_date, start_date, limit)

    if not all_trades:
        return []

    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or data source."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    client = get_client()
    all_news = client.get_company_news(ticker, end_date, start_date, limit)

    if not all_news:
        return []

    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from data source."""
    client = get_client()
    return client.get_market_cap(ticker, end_date)


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
