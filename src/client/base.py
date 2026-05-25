from abc import ABC, abstractmethod

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)


class BaseClient(ABC):

    @abstractmethod
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        pass

    @abstractmethod
    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[FinancialMetrics]:
        pass

    @abstractmethod
    def search_line_items(
        self, ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[LineItem]:
        pass

    @abstractmethod
    def get_insider_trades(
        self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000
    ) -> list[InsiderTrade]:
        pass

    @abstractmethod
    def get_company_news(
        self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000
    ) -> list[CompanyNews]:
        pass

    @abstractmethod
    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        pass
