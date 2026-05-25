import os

from src.client.base import BaseClient
from src.client.financial_datasets import FinancialDatasetsClient
from src.client.yahoo import YahooFinanceClient


def get_client(source: str | None = None) -> BaseClient:
    """Get data client based on configuration.

    Args:
        source: Data source name. If None, reads from DATA_SOURCE env var.
                Defaults to "yahoo" if not set.

    Supported sources: "yahoo", "financial_datasets"
    """
    if source is None:
        source = os.environ.get("DATA_SOURCE", "yahoo")

    if source == "financial_datasets":
        return FinancialDatasetsClient()
    return YahooFinanceClient()


__all__ = [
    "BaseClient",
    "FinancialDatasetsClient",
    "YahooFinanceClient",
    "get_client",
]
