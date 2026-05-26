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
        self, ticker: str, end_date: str, period: str = "quarterly", limit: int = 10
    ) -> list[FinancialMetrics]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if period == "quarterly":
                income_stmt = stock.quarterly_income_stmt
                balance_sheet = stock.quarterly_balance_sheet
                cashflow = stock.quarterly_cashflow
            else:  # annual / ttm
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cashflow = stock.cashflow
        except Exception as e:
            logger.warning("Failed to fetch financial data for %s: %s", ticker, e)
            return []

        # Collect raw data per period first, then compute growth metrics
        raw_periods = []
        periods_to_use = income_stmt.columns if not income_stmt.empty else []

        for col in periods_to_use:
            report_date = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
            if report_date > end_date:
                continue

            revenue = _safe_get(income_stmt, "Total Revenue", col)
            net_income = _safe_get(income_stmt, "Net Income", col)
            gross_profit = _safe_get(income_stmt, "Gross Profit", col)
            operating_income = _safe_get(income_stmt, "Operating Income", col)
            ebitda = _safe_get(income_stmt, "EBITDA", col)
            cost_of_revenue = _safe_get(income_stmt, "Cost Of Revenue", col)
            interest_expense = _safe_get(income_stmt, "Interest Expense", col)
            shares = _safe_get(income_stmt, "Diluted Average Shares", col)
            dividends_paid = _safe_get(cashflow, "Common Stock Dividend Paid", col)

            total_assets = _safe_get(balance_sheet, "Total Assets", col)
            total_equity = _safe_get(balance_sheet, "Stockholders Equity", col)
            total_debt = _safe_get(balance_sheet, "Total Debt", col)
            current_assets = _safe_get(balance_sheet, "Current Assets", col)
            current_liabilities = _safe_get(balance_sheet, "Current Liabilities", col)
            cash = _safe_get(balance_sheet, "Cash And Cash Equivalents", col)
            inventory = _safe_get(balance_sheet, "Inventory", col)
            accounts_receivable = _safe_get(balance_sheet, "Accounts Receivable", col)

            operating_cf = _safe_get(cashflow, "Operating Cash Flow", col)
            capex = _safe_get(cashflow, "Capital Expenditure", col)

            market_cap = info.get("marketCap")
            enterprise_value = info.get("enterpriseValue")

            fcf = (operating_cf + capex) if operating_cf is not None and capex is not None else None

            raw_periods.append({
                "report_date": report_date,
                "revenue": revenue,
                "net_income": net_income,
                "gross_profit": gross_profit,
                "operating_income": operating_income,
                "ebitda": ebitda,
                "cost_of_revenue": cost_of_revenue,
                "interest_expense": interest_expense,
                "shares": shares,
                "dividends_paid": dividends_paid,
                "total_assets": total_assets,
                "total_equity": total_equity,
                "total_debt": total_debt,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "cash": cash,
                "inventory": inventory,
                "accounts_receivable": accounts_receivable,
                "operating_cf": operating_cf,
                "capex": capex,
                "fcf": fcf,
                "market_cap": market_cap,
                "enterprise_value": enterprise_value,
            })

        metrics_list = []
        for i, p in enumerate(raw_periods):
            prev = raw_periods[i + 1] if i + 1 < len(raw_periods) else None

            revenue = p["revenue"]
            net_income = p["net_income"]
            gross_profit = p["gross_profit"]
            operating_income = p["operating_income"]
            ebitda = p["ebitda"]
            total_assets = p["total_assets"]
            total_equity = p["total_equity"]
            total_debt = p["total_debt"]
            current_assets = p["current_assets"]
            current_liabilities = p["current_liabilities"]
            cash = p["cash"]
            inventory = p["inventory"]
            accounts_receivable = p["accounts_receivable"]
            operating_cf = p["operating_cf"]
            shares = p["shares"]
            fcf = p["fcf"]
            market_cap = p["market_cap"]
            enterprise_value = p["enterprise_value"]
            dividends_paid = p["dividends_paid"]
            interest_expense = p["interest_expense"]
            cost_of_revenue = p["cost_of_revenue"]

            # Margins
            gross_margin = (gross_profit / revenue) if revenue and gross_profit else None
            operating_margin = (operating_income / revenue) if revenue and operating_income else None
            net_margin = (net_income / revenue) if revenue and net_income else None

            # Returns
            roe = (net_income / total_equity) if total_equity and net_income else None
            roa = (net_income / total_assets) if total_assets and net_income else None
            roic = (net_income / (total_equity + total_debt)) if total_equity and total_debt and net_income else None

            # Liquidity
            current_ratio = (current_assets / current_liabilities) if current_liabilities and current_assets else None
            quick_ratio = ((current_assets - (inventory or 0)) / current_liabilities) if current_liabilities and current_assets else None
            cash_ratio = (cash / current_liabilities) if current_liabilities and cash else None

            # Leverage
            debt_to_equity = (total_debt / total_equity) if total_equity and total_debt else None
            debt_to_assets = (total_debt / total_assets) if total_assets and total_debt else None
            interest_coverage = (operating_income / abs(interest_expense)) if operating_income and interest_expense else None

            # Turnover
            asset_turnover = (revenue / total_assets) if total_assets and revenue else None
            inventory_turnover = (cost_of_revenue / inventory) if cost_of_revenue and inventory else None
            receivables_turnover = (revenue / accounts_receivable) if revenue and accounts_receivable else None
            days_sales_outstanding = (365 / receivables_turnover) if receivables_turnover else None
            working_capital = (current_assets - current_liabilities) if current_assets and current_liabilities else None
            working_capital_turnover = (revenue / working_capital) if revenue and working_capital else None

            # Per share
            eps = (net_income / shares) if shares and net_income else None
            bvps = (total_equity / shares) if shares and total_equity else None
            fcf_per_share = (fcf / shares) if shares and fcf else None

            # Valuation — use market cap for historical periods instead of current-only info fields
            fcf_yield = (fcf / market_cap) if fcf and market_cap else None
            ev_to_ebitda = (enterprise_value / ebitda) if enterprise_value and ebitda else None
            ev_to_revenue = (enterprise_value / revenue) if enterprise_value and revenue else None
            pe_ratio = (market_cap / net_income) if market_cap and net_income and net_income > 0 else None
            pb_ratio = (market_cap / total_equity) if market_cap and total_equity and total_equity > 0 else None
            ps_ratio = (market_cap / revenue) if market_cap and revenue else None

            # Payout ratio
            payout_ratio = (abs(dividends_paid) / net_income) if dividends_paid and net_income and net_income > 0 else None

            # Growth (period-over-period vs previous year)
            def _growth(curr, prev_val):
                if curr is not None and prev_val and prev_val != 0:
                    return (curr - prev_val) / abs(prev_val)
                return None

            revenue_growth = _growth(revenue, prev["revenue"] if prev else None)
            earnings_growth = _growth(net_income, prev["net_income"] if prev else None)
            book_value_growth = _growth(total_equity, prev["total_equity"] if prev else None)
            eps_growth = _growth(eps, (prev["net_income"] / prev["shares"]) if prev and prev["shares"] and prev["net_income"] else None)
            fcf_growth = _growth(fcf, prev["fcf"] if prev else None)
            operating_income_growth = _growth(operating_income, prev["operating_income"] if prev else None)
            ebitda_growth = _growth(ebitda, prev["ebitda"] if prev else None)

            metrics_list.append(FinancialMetrics(
                ticker=ticker,
                report_period=p["report_date"],
                period="quarterly" if period == "quarterly" else "annual",
                currency="USD",
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                price_to_earnings_ratio=pe_ratio,
                price_to_book_ratio=pb_ratio,
                price_to_sales_ratio=ps_ratio,
                enterprise_value_to_ebitda_ratio=ev_to_ebitda,
                enterprise_value_to_revenue_ratio=ev_to_revenue,
                free_cash_flow_yield=fcf_yield,
                peg_ratio=info.get("pegRatio") if i == 0 else None,
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin,
                return_on_equity=roe,
                return_on_assets=roa,
                return_on_invested_capital=roic,
                asset_turnover=asset_turnover,
                inventory_turnover=inventory_turnover,
                receivables_turnover=receivables_turnover,
                days_sales_outstanding=days_sales_outstanding,
                operating_cycle=None,
                working_capital_turnover=working_capital_turnover,
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                cash_ratio=cash_ratio,
                operating_cash_flow_ratio=(operating_cf / current_liabilities) if operating_cf and current_liabilities else None,
                debt_to_equity=debt_to_equity,
                debt_to_assets=debt_to_assets,
                interest_coverage=interest_coverage,
                revenue_growth=revenue_growth,
                earnings_growth=earnings_growth,
                book_value_growth=book_value_growth,
                earnings_per_share_growth=eps_growth,
                free_cash_flow_growth=fcf_growth,
                operating_income_growth=operating_income_growth,
                ebitda_growth=ebitda_growth,
                payout_ratio=payout_ratio,
                earnings_per_share=eps,
                book_value_per_share=bvps,
                free_cash_flow_per_share=fcf_per_share,
            ))

            if len(metrics_list) >= limit:
                break

        return metrics_list

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
    if col not in df.columns:
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
