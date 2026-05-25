import datetime
import logging

import yfinance as yf

from src.client.base import BaseClient
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

logger = logging.getLogger(__name__)

# Mapping from line_item names used by financialdatasets to yfinance field names
_LINE_ITEM_MAPPING = {
    "revenue": "Total Revenue",
    "total_revenue": "Total Revenue",
    "cost_of_revenue": "Cost Of Revenue",
    "gross_profit": "Gross Profit",
    "operating_income": "Operating Income",
    "net_income": "Net Income",
    "ebitda": "EBITDA",
    "ebit": "EBIT",
    "interest_expense": "Interest Expense",
    "income_tax_expense": "Tax Provision",
    "depreciation_and_amortization": "Reconciled Depreciation",
    "research_and_development": "Research Development",
    "selling_general_and_administrative": "Selling General Administrative",
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities Net Minority Interest",
    "total_equity": "Total Equity Gross Minority Interest",
    "stockholders_equity": "Stockholders Equity",
    "cash_and_equivalents": "Cash And Cash Equivalents",
    "short_term_investments": "Other Short Term Investments",
    "total_current_assets": "Current Assets",
    "total_current_liabilities": "Current Liabilities",
    "long_term_debt": "Long Term Debt",
    "short_term_debt": "Current Debt",
    "total_debt": "Total Debt",
    "shares_outstanding": "Ordinary Shares Number",
    "weighted_average_shares_outstanding": "Diluted Average Shares",
    "operating_cash_flow": "Operating Cash Flow",
    "capital_expenditure": "Capital Expenditure",
    "free_cash_flow": "Free Cash Flow",
    "dividends_paid": "Common Stock Dividend Paid",
    "net_cash_from_financing": "Financing Cash Flow",
    "net_cash_from_investing": "Investing Cash Flow",
    "inventory": "Inventory",
    "accounts_receivable": "Accounts Receivable",
    "accounts_payable": "Accounts Payable",
    "goodwill": "Goodwill",
    "intangible_assets": "Net Intangible Assets",
    "retained_earnings": "Retained Earnings",
}


class YahooFinanceClient(BaseClient):

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=False)
        except Exception as e:
            logger.warning("Failed to fetch prices for %s: %s", ticker, e)
            return []

        if df.empty:
            return []

        prices = []
        for idx, row in df.iterrows():
            prices.append(Price(
                open=round(float(row["Open"]), 4),
                close=round(float(row["Close"]), 4),
                high=round(float(row["High"]), 4),
                low=round(float(row["Low"]), 4),
                volume=int(row["Volume"]),
                time=idx.strftime("%Y-%m-%d"),
            ))
        return prices

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[FinancialMetrics]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
        except Exception as e:
            logger.warning("Failed to fetch financial data for %s: %s", ticker, e)
            return []

        metrics_list = []
        periods_to_use = income_stmt.columns[:limit] if not income_stmt.empty else []

        for col in periods_to_use:
            report_date = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
            if report_date > end_date:
                continue

            revenue = _safe_get(income_stmt, "Total Revenue", col)
            net_income = _safe_get(income_stmt, "Net Income", col)
            gross_profit = _safe_get(income_stmt, "Gross Profit", col)
            operating_income = _safe_get(income_stmt, "Operating Income", col)
            ebitda = _safe_get(income_stmt, "EBITDA", col)

            total_assets = _safe_get(balance_sheet, "Total Assets", col)
            total_equity = _safe_get(balance_sheet, "Stockholders Equity", col)
            total_debt = _safe_get(balance_sheet, "Total Debt", col)
            current_assets = _safe_get(balance_sheet, "Current Assets", col)
            current_liabilities = _safe_get(balance_sheet, "Current Liabilities", col)
            cash = _safe_get(balance_sheet, "Cash And Cash Equivalents", col)
            inventory = _safe_get(balance_sheet, "Inventory", col)

            operating_cf = _safe_get(cashflow, "Operating Cash Flow", col)
            capex = _safe_get(cashflow, "Capital Expenditure", col)
            shares = _safe_get(income_stmt, "Diluted Average Shares", col)

            market_cap = info.get("marketCap")
            enterprise_value = info.get("enterpriseValue")

            gross_margin = (gross_profit / revenue) if revenue and gross_profit else None
            operating_margin = (operating_income / revenue) if revenue and operating_income else None
            net_margin = (net_income / revenue) if revenue and net_income else None
            roe = (net_income / total_equity) if total_equity and net_income else None
            roa = (net_income / total_assets) if total_assets and net_income else None

            current_ratio = (current_assets / current_liabilities) if current_liabilities and current_assets else None
            quick_ratio = ((current_assets - (inventory or 0)) / current_liabilities) if current_liabilities and current_assets else None
            cash_ratio = (cash / current_liabilities) if current_liabilities and cash else None

            debt_to_equity = (total_debt / total_equity) if total_equity and total_debt else None
            debt_to_assets = (total_debt / total_assets) if total_assets and total_debt else None

            fcf = (operating_cf + capex) if operating_cf is not None and capex is not None else None
            fcf_yield = (fcf / market_cap) if fcf and market_cap else None

            eps = (net_income / shares) if shares and net_income else None
            bvps = (total_equity / shares) if shares and total_equity else None
            fcf_per_share = (fcf / shares) if shares and fcf else None

            pe_ratio = info.get("trailingPE") if metrics_list == [] else None
            pb_ratio = info.get("priceToBook") if metrics_list == [] else None
            ps_ratio = info.get("priceToSalesTrailing12Months") if metrics_list == [] else None
            ev_to_ebitda = (enterprise_value / ebitda) if enterprise_value and ebitda else None
            ev_to_revenue = (enterprise_value / revenue) if enterprise_value and revenue else None

            interest_expense = _safe_get(income_stmt, "Interest Expense", col)
            interest_coverage = (operating_income / abs(interest_expense)) if operating_income and interest_expense else None

            metrics_list.append(FinancialMetrics(
                ticker=ticker,
                report_period=report_date,
                period="annual",
                currency="USD",
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                price_to_earnings_ratio=pe_ratio,
                price_to_book_ratio=pb_ratio,
                price_to_sales_ratio=ps_ratio,
                enterprise_value_to_ebitda_ratio=ev_to_ebitda,
                enterprise_value_to_revenue_ratio=ev_to_revenue,
                free_cash_flow_yield=fcf_yield,
                peg_ratio=info.get("pegRatio") if metrics_list == [] else None,
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin,
                return_on_equity=roe,
                return_on_assets=roa,
                return_on_invested_capital=None,
                asset_turnover=(revenue / total_assets) if total_assets and revenue else None,
                inventory_turnover=None,
                receivables_turnover=None,
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                cash_ratio=cash_ratio,
                operating_cash_flow_ratio=(operating_cf / current_liabilities) if operating_cf and current_liabilities else None,
                debt_to_equity=debt_to_equity,
                debt_to_assets=debt_to_assets,
                interest_coverage=interest_coverage,
                revenue_growth=info.get("revenueGrowth") if metrics_list == [] else None,
                earnings_growth=info.get("earningsGrowth") if metrics_list == [] else None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=info.get("payoutRatio") if metrics_list == [] else None,
                earnings_per_share=eps,
                book_value_per_share=bvps,
                free_cash_flow_per_share=fcf_per_share,
            ))

        return metrics_list[:limit]

    def search_line_items(
        self, ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[LineItem]:
        try:
            stock = yf.Ticker(ticker)
            if period in ("annual", "ttm"):
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cashflow = stock.cashflow
            else:
                income_stmt = stock.quarterly_income_stmt
                balance_sheet = stock.quarterly_balance_sheet
                cashflow = stock.quarterly_cashflow
        except Exception as e:
            logger.warning("Failed to fetch financials for %s: %s", ticker, e)
            return []

        all_statements = {}
        for stmt in [income_stmt, balance_sheet, cashflow]:
            if stmt is not None and not stmt.empty:
                for idx_label in stmt.index:
                    all_statements[idx_label] = stmt.loc[idx_label]

        results = []
        columns = income_stmt.columns if not income_stmt.empty else []

        for col in columns:
            report_date = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
            if report_date > end_date:
                continue

            extra_fields = {}
            for item_name in line_items:
                yf_field = _LINE_ITEM_MAPPING.get(item_name)
                value = None

                if yf_field and yf_field in all_statements:
                    raw = all_statements[yf_field].get(col)
                    if raw is not None and str(raw) != "nan":
                        value = float(raw)
                else:
                    for label, series in all_statements.items():
                        if item_name.lower().replace("_", " ") in label.lower():
                            raw = series.get(col)
                            if raw is not None and str(raw) != "nan":
                                value = float(raw)
                            break

                extra_fields[item_name] = value

            results.append(LineItem(
                ticker=ticker,
                report_period=report_date,
                period="annual" if period in ("annual", "ttm") else "quarterly",
                currency="USD",
                **extra_fields,
            ))

            if len(results) >= limit:
                break

        return results

    def get_insider_trades(
        self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000
    ) -> list[InsiderTrade]:
        try:
            stock = yf.Ticker(ticker)
            transactions = stock.insider_transactions
        except Exception as e:
            logger.warning("Failed to fetch insider trades for %s: %s", ticker, e)
            return []

        if transactions is None or transactions.empty:
            return []

        trades = []
        for _, row in transactions.iterrows():
            filing_date = _parse_date(row.get("Start Date") or row.get("startDate"))
            if not filing_date:
                continue
            if filing_date > end_date:
                continue
            if start_date and filing_date < start_date:
                continue

            shares = row.get("Shares") or row.get("shares")
            value = row.get("Value") or row.get("value")
            shares_float = float(shares) if shares and str(shares) != "nan" else None

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=row.get("Insider") or row.get("insider"),
                title=row.get("Position") or row.get("position"),
                is_board_director=None,
                transaction_date=filing_date,
                transaction_shares=shares_float,
                transaction_price_per_share=None,
                transaction_value=float(value) if value and str(value) != "nan" else None,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=None,
                filing_date=filing_date,
            ))

            if len(trades) >= limit:
                break

        return trades

    def get_company_news(
        self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000
    ) -> list[CompanyNews]:
        try:
            stock = yf.Ticker(ticker)
            news_list = stock.news
        except Exception as e:
            logger.warning("Failed to fetch news for %s: %s", ticker, e)
            return []

        if not news_list:
            return []

        results = []
        for item in news_list:
            pub_ts = item.get("providerPublishTime") or item.get("publishedAt")
            if pub_ts:
                if isinstance(pub_ts, (int, float)):
                    pub_date = datetime.datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d")
                else:
                    pub_date = str(pub_ts)[:10]
            else:
                pub_date = end_date

            if pub_date > end_date:
                continue
            if start_date and pub_date < start_date:
                continue

            results.append(CompanyNews(
                ticker=ticker,
                title=item.get("title", ""),
                author=None,
                source=item.get("publisher", "Yahoo Finance"),
                date=pub_date,
                url=item.get("link", ""),
                sentiment=None,
            ))

            if len(results) >= limit:
                break

        return results

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get("marketCap")
            return float(market_cap) if market_cap else None
        except Exception as e:
            logger.warning("Failed to fetch market cap for %s: %s", ticker, e)
            return None


def _safe_get(df, field: str, col) -> float | None:
    if df is None or df.empty:
        return None
    if field not in df.index:
        return None
    val = df.loc[field, col]
    if val is None or str(val) == "nan":
        return None
    return float(val)


def _parse_date(val) -> str | None:
    if val is None:
        return None
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    s = str(val)
    if s == "nan" or s == "NaT":
        return None
    return s[:10]
