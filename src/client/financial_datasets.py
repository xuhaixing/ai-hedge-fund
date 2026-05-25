import datetime
import logging
import os
import time

import requests

from src.client.base import BaseClient
from src.data.models import (
    CompanyFacts,
    CompanyFactsResponse,
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    InsiderTrade,
    InsiderTradeResponse,
    LineItem,
    LineItemResponse,
    Price,
    PriceResponse,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.financialdatasets.ai"


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    for attempt in range(max_retries + 1):
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)

        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue

        return response


class FinancialDatasetsClient(BaseClient):

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        return headers

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        url = f"{BASE_URL}/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
        response = _make_api_request(url, self._headers())
        if response.status_code != 200:
            return []

        try:
            price_response = PriceResponse(**response.json())
            return price_response.prices or []
        except Exception as e:
            logger.warning("Failed to parse price response for %s: %s", ticker, e)
            return []

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[FinancialMetrics]:
        url = f"{BASE_URL}/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
        response = _make_api_request(url, self._headers())
        if response.status_code != 200:
            return []

        try:
            metrics_response = FinancialMetricsResponse(**response.json())
            return metrics_response.financial_metrics or []
        except Exception as e:
            logger.warning("Failed to parse financial metrics response for %s: %s", ticker, e)
            return []

    def search_line_items(
        self, ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[LineItem]:
        url = f"{BASE_URL}/financials/search/line-items"
        body = {
            "tickers": [ticker],
            "line_items": line_items,
            "end_date": end_date,
            "period": period,
            "limit": limit,
        }
        response = _make_api_request(url, self._headers(), method="POST", json_data=body)
        if response.status_code != 200:
            return []

        try:
            data = response.json()
            response_model = LineItemResponse(**data)
            search_results = response_model.search_results
        except Exception as e:
            logger.warning("Failed to parse line items response for %s: %s", ticker, e)
            return []

        if not search_results:
            return []
        return search_results[:limit]

    def get_insider_trades(
        self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000
    ) -> list[InsiderTrade]:
        all_trades = []
        current_end_date = end_date

        while True:
            url = f"{BASE_URL}/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
            if start_date:
                url += f"&filing_date_gte={start_date}"
            url += f"&limit={limit}"

            response = _make_api_request(url, self._headers())
            if response.status_code != 200:
                break

            try:
                data = response.json()
                response_model = InsiderTradeResponse(**data)
                insider_trades = response_model.insider_trades
            except Exception as e:
                logger.warning("Failed to parse insider trades response for %s: %s", ticker, e)
                break

            if not insider_trades:
                break

            all_trades.extend(insider_trades)

            if not start_date or len(insider_trades) < limit:
                break

            current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]
            if current_end_date <= start_date:
                break

        return all_trades

    def get_company_news(
        self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000
    ) -> list[CompanyNews]:
        all_news = []
        current_end_date = end_date

        while True:
            url = f"{BASE_URL}/news/?ticker={ticker}&end_date={current_end_date}"
            if start_date:
                url += f"&start_date={start_date}"
            url += f"&limit={limit}"

            response = _make_api_request(url, self._headers())
            if response.status_code != 200:
                break

            try:
                data = response.json()
                response_model = CompanyNewsResponse(**data)
                company_news = response_model.news
            except Exception as e:
                logger.warning("Failed to parse company news response for %s: %s", ticker, e)
                break

            if not company_news:
                break

            all_news.extend(company_news)

            if not start_date or len(company_news) < limit:
                break

            current_end_date = min(news.date for news in company_news).split("T")[0]
            if current_end_date <= start_date:
                break

        return all_news

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
            url = f"{BASE_URL}/company/facts/?ticker={ticker}"
            response = _make_api_request(url, self._headers())
            if response.status_code != 200:
                return None

            data = response.json()
            response_model = CompanyFactsResponse(**data)
            return response_model.company_facts.market_cap

        financial_metrics = self.get_financial_metrics(ticker, end_date)
        if not financial_metrics:
            return None

        return financial_metrics[0].market_cap
