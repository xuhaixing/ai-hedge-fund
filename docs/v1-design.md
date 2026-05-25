# AI Hedge Fund v1 系统设计文档

## 目录

- [概述](#概述)
- [系统架构](#系统架构)
- [项目目录结构](#项目目录结构)
- [策略 Agent 详细设计](#策略-agent-详细设计)
- [风险管理 Agent](#风险管理-agent)
- [组合管理 Agent](#组合管理-agent)
- [其他 Agent 简述](#其他-agent-简述)
- [回测系统](#回测系统)
- [数据层](#数据层)
- [技术栈](#技术栈)

---

## 概述

AI Hedge Fund v1 是一个基于多 Agent 协作的 AI 量化交易决策系统。系统采用"投资者人格"架构，模拟多位著名投资大师的投资哲学和分析方法，通过 LangGraph 有向图调度各 Agent 并行分析，最终由风险管理和组合管理器生成交易决策。

> 核心理念：每个 Agent 模拟一位真实投资者的决策框架，通过量化打分 + LLM 推理的混合模式生成信号，多路信号汇总后由组合管理器做最终交易决策。

## 系统架构

```
用户输入(Tickers + 日期 + 资金)
         │
         ▼
    ┌─────────┐
    │Start Node│
    └────┬────┘
         │ (并行分支)
    ┌────┼────┬────┬────┬────┬────┬────┬────┐
    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
  Graham Buffett Wood Burry Tech  Fund  Sent  Val  ...其他
    │    │    │    │    │    │    │    │    │
    └────┴────┴────┴────┴────┴────┴────┴────┘
         │ (全部汇入)
         ▼
  ┌──────────────────┐
  │Risk Management   │  波动率+相关性 → 持仓限额
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │Portfolio Manager  │  汇总信号+约束 → 交易指令
  └────────┬─────────┘
           ▼
      交易决策输出
```

### 执行流程

| 阶段 | 说明 |
|------|------|
| 1. 数据获取 | 各 Agent 通过数据客户端层（默认 Yahoo Finance）获取价格、财务指标、内部交易、新闻等数据 |
| 2. 并行分析 | 所有 Analyst Agent 并行执行，各自生成 `{signal, confidence, reasoning}` |
| 3. 风险管控 | Risk Management Agent 基于波动率和相关性计算每个 Ticker 的持仓上限 |
| 4. 组合决策 | Portfolio Manager 汇总所有信号和约束，由 LLM 生成最终交易指令 |

### 状态管理 (AgentState)

```python
AgentState(TypedDict):
  messages: List[BaseMessage]     # 累积消息链
  data:
    tickers: List[str]            # 待分析标的
    portfolio: Dict               # 当前持仓与资金
    start_date / end_date: str    # 分析时间窗口
    analyst_signals: Dict         # 各Agent信号汇总
  metadata:
    show_reasoning: bool          # 是否展示推理过程
    model_name / model_provider   # LLM 配置
```

---

## 项目目录结构

```
src/
├── __init__.py
├── main.py                      # 主入口，run_hedge_fund() 函数
├── backtester.py                # 旧版回测脚本（保留兼容）
│
├── agents/                      # 策略 Agent 实现
│   ├── ben_graham.py            # 本杰明·格雷厄姆
│   ├── warren_buffett.py        # 沃伦·巴菲特
│   ├── cathie_wood.py           # 凯西·伍德
│   ├── michael_burry.py         # 迈克尔·伯里
│   ├── charlie_munger.py        # 查理·芒格
│   ├── bill_ackman.py           # 比尔·阿克曼
│   ├── phil_fisher.py           # 菲利普·费雪
│   ├── peter_lynch.py           # 彼得·林奇
│   ├── stanley_druckenmiller.py # 斯坦利·德鲁肯米勒
│   ├── mohnish_pabrai.py        # 莫尼什·帕布莱
│   ├── nassim_taleb.py          # 纳西姆·塔勒布
│   ├── rakesh_jhunjhunwala.py   # 拉凯什·金君瓦拉
│   ├── aswath_damodaran.py      # 阿斯沃斯·达莫达兰
│   ├── growth_agent.py          # 增长分析师
│   ├── fundamentals.py          # 基本面分析师
│   ├── technicals.py            # 技术分析师
│   ├── sentiment.py             # 情绪分析师
│   ├── news_sentiment.py        # 新闻情绪分析师
│   ├── valuation.py             # 估值分析师
│   ├── risk_manager.py          # 风险管理 Agent
│   └── portfolio_manager.py     # 组合管理 Agent
│
├── backtesting/                 # 模块化回测引擎
│   ├── cli.py                   # CLI 入口 (backtester 命令)
│   ├── engine.py                # BacktestEngine 回测主循环
│   ├── portfolio.py             # Portfolio 持仓/现金/保证金管理
│   ├── trader.py                # TradeExecutor 交易执行
│   ├── metrics.py               # 绩效指标计算 (Sharpe/Sortino/MaxDD)
│   ├── valuation.py             # 组合估值与敞口计算
│   ├── benchmarks.py            # 基准收益计算 (SPY)
│   ├── output.py                # 终端输出格式化
│   ├── controller.py            # Agent 调度桥接
│   └── types.py                 # 类型定义 (Action, PortfolioSnapshot 等)
│
├── client/                      # 数据源客户端层
│   ├── __init__.py              # get_client() 工厂函数
│   ├── base.py                  # BaseClient 抽象基类
│   ├── yahoo.py                 # YahooFinanceClient
│   └── financial_datasets.py    # FinancialDatasetsClient
│
├── data/                        # 数据模型与缓存
│   ├── models.py                # Pydantic 数据模型定义
│   └── cache.py                 # 内存缓存层
│
├── tools/                       # 工具层（API 封装）
│   └── api.py                   # 统一数据接口（缓存 + 客户端调用）
│
├── graph/                       # LangGraph 状态图定义
│   └── state.py                 # AgentState 定义
│
├── llm/                         # LLM 配置
│   └── models.py                # 模型列表、Provider 枚举
│
├── cli/                         # CLI 交互
│   └── input.py                 # 用户输入处理
│
└── utils/                       # 通用工具
    ├── analysts.py              # Agent 列表与排序
    ├── api_key.py               # API Key 管理
    ├── display.py               # 显示工具
    ├── docker.py                # Docker 相关
    ├── llm.py                   # LLM 工具函数
    ├── ollama.py                # Ollama 本地模型管理
    ├── progress.py              # 进度条
    └── visualize.py             # 可视化
```

---

## 策略 Agent 详细设计

### 1. Ben Graham Agent（本杰明·格雷厄姆，价值投资之父）

> The Father of Value Investing — 强调安全边际，投资于具有强劲基本面的低估公司，通过系统性价值分析进行决策。

**投资哲学**: 坚持安全边际，只买入低于内在价值的股票。

#### 分析维度与计算方法

**A. 盈利稳定性分析 (满分 4 分)**

| 指标 | 计算方法 | 评分规则 |
|------|----------|----------|
| EPS 正值年份占比 | 统计 10 年 EPS > 0 的年份数 | 100% → 3分, ≥80% → 2分, 其他 → 0分 |
| EPS 增长趋势 | 比较最早期与最近期 EPS | 有增长 → 1分 |

**B. 财务实力分析 (满分 5 分)**

| 指标 | 计算方法 | 评分规则 |
|------|----------|----------|
| 流动比率 | Current Assets / Current Liabilities | ≥2.0 → 2分, ≥1.5 → 1分 |
| 负债比率 | Total Liabilities / Total Assets | <0.5 → 2分, <0.8 → 1分 |
| 分红记录 | 历史分红年份占比 | 多数年份有分红 → 1分 |

**C. Graham 估值分析 (满分 7 分)**

| 指标 | 公式 | 评分规则 |
|------|------|----------|
| 净流动资产价值(NCAV) | Current Assets − Total Liabilities | NCAV > Market Cap → 4分, NCAV/Share ≥ 2/3 Price → 2分 |
| Graham Number | √(22.5 × EPS × Book Value Per Share) | 安全边际 > 50% → 3分, > 20% → 1分 |
| 安全边际 | (Graham Number − Price) / Price | 衡量低估程度 |

**信号判定**: 总分 ≥ 70% × 15 → bullish, ≤ 30% × 15 → bearish, 其余 → neutral

---

### 2. Warren Buffett Agent（沃伦·巴菲特，奥马哈先知）

> The Oracle of Omaha — 寻找具有强劲基本面和竞争优势的公司，通过价值投资和长期持有实现收益。

**投资哲学**: 以合理价格买入优质企业，长期持有。

#### 分析维度与计算方法

**A. 基本面分析 (满分 7 分)**

| 指标 | 阈值 | 得分 |
|------|------|------|
| ROE (净资产收益率) | > 15% | 2分 |
| 负债权益比 | < 0.5 | 2分 |
| 营业利润率 | > 15% | 2分 |
| 流动比率 | > 1.5 | 1分 |

**B. 盈利一致性分析 (满分 3 分)**

- 检查连续 4+ 期净利润是否逐期递增
- 连续增长 → 3分
- 计算最早期到最近期的总增长率

**C. 竞争护城河分析 (满分 5 分)**

| 子维度 | 计算方法 | 满分 |
|--------|----------|------|
| ROE一致性 | 5年以上 >15% 的期数占比 ≥80% | 2分 |
| 利润率稳定性 | 平均营业利润率 >20% 且近期≥早期 | 1分 |
| 资产效率 | Asset Turnover > 1.0 | 1分 |
| 绩效稳定性 | ROE与利润率的变异系数 > 0.7 | 1分 |

**D. 定价能力分析 (满分 5 分)**

| 条件 | 得分 |
|------|------|
| 毛利率 > 50% | 2分 |
| 毛利率趋势改善 >2% | 3分 |
| 毛利率趋势改善 >0% | 2分 |
| 毛利率稳定(±1%) | 1分 |

**E. 账面价值增长分析 (满分 5 分)**

| 条件 | 得分 |
|------|------|
| 一致性: 增长期数占比 ≥80% | 3分 |
| CAGR > 15% | 2分 |
| CAGR > 10% | 1分 |

**F. 管理层质量分析 (满分 2 分)**

| 条件 | 得分 |
|------|------|
| 回购股票(issuance < 0) | 1分 |
| 有分红记录 | 1分 |

**G. 内在价值计算 (三阶段 DCF)**

```
Owner Earnings = Net Income + Depreciation − Maintenance CapEx − Working Capital Change

Stage 1 (5年): growth = min(conservative_growth, 8%)
Stage 2 (5年): growth = min(stage1 × 0.5, 4%)
Terminal:       growth = 2.5%, discount_rate = 10%

Intrinsic Value = PV(Stage1) + PV(Stage2) + PV(Terminal) × 85%
Margin of Safety = (Intrinsic Value − Market Cap) / Market Cap
```

**Maintenance CapEx 估算**: 取以下三者的中位数:
1. 总 CapEx × 85%
2. 折旧摊销 × 100%
3. 历史 CapEx/Revenue 比率 × 当期 Revenue

---

### 3. Cathie Wood Agent（凯西·伍德，颠覆式创新女王）

> The Queen of Growth Investing — 专注于颠覆性创新和增长，投资于引领技术进步和市场颠覆的公司。

**投资哲学**: 投资于引领技术革命的颠覆性创新企业。

#### 分析维度与计算方法

**A. 颠覆性潜力分析 (归一化至 5 分)**

| 子维度 | 计算方法 | 最高分 |
|--------|----------|--------|
| 收入增长加速 | 近期增长率 > 早期增长率 | 2分 |
| 绝对增长率 | >100% → 3分, >50% → 2分, >20% → 1分 | 3分 |
| 毛利率扩张 | 首尾差 >5% → 2分; 绝对值 >50% → 2分 | 4分 |
| 运营杠杆 | 收入增速 > 费用增速 | 2分 |
| R&D 强度 | R&D/Revenue >15% → 3分, >8% → 2分 | 3分 |

归一化: `normalized_score = (raw / 12) × 5`

**B. 创新增长分析 (归一化至 5 分)**

| 子维度 | 计算方法 | 最高分 |
|--------|----------|--------|
| R&D投资增长 | 首尾增长率 >50% → 3分, >20% → 2分 | 3分 |
| R&D强度趋势 | 近期强度 > 早期强度 | 2分 |
| FCF增长与一致性 | 增长>30% 且全为正值 → 3分 | 3分 |
| 营业利润率 | >15%且趋势向上 → 3分, >10% → 2分 | 3分 |
| CapEx投入 | 强度>10% 且增长>20% → 2分 | 2分 |
| 再投资比例 | 分红率 <20% → 2分 | 2分 |

归一化: `normalized_score = (raw / 15) × 5`

**C. 高增长场景估值 (满分 3 分)**

```
Growth Rate   = 20%
Discount Rate = 15%
Terminal Multiple = 25x
Projection Years  = 5

PV = Σ(FCF × 1.20^t / 1.15^t), t=1..5
Terminal = FCF × 1.20^5 × 25 / 1.15^5
Intrinsic Value = PV + Terminal
Margin of Safety = (IV − Market Cap) / Market Cap
```

| 条件 | 得分 |
|------|------|
| 安全边际 > 50% | 3分 |
| 安全边际 > 20% | 1分 |

---

### 4. Michael Burry Agent（迈克尔·伯里，逆向投资者）

> The Big Short Contrarian — 进行逆向押注，常做空高估市场，通过深度基本面分析投资于被低估资产。

**投资哲学**: 深度价值 + 逆向思维，寻找被市场厌弃但基本面坚实的标的。

#### 分析维度与计算方法

**A. 价值分析 (满分 6 分)**

| 指标 | 公式 | 评分 |
|------|------|------|
| FCF Yield | Free Cash Flow / Market Cap | ≥15% → 4分, ≥12% → 3分, ≥8% → 2分 |
| EV/EBIT | Enterprise Value / EBIT | <6 → 2分, <10 → 1分 |

**B. 资产负债表分析 (满分 3 分)**

| 指标 | 条件 | 得分 |
|------|------|------|
| D/E Ratio | < 0.5 → 2分, < 1.0 → 1分 | 2分 |
| 净现金 | Cash > Total Debt | 1分 |

**C. 内部人交易分析 (满分 2 分)**

```
Net Buying = Σ(买入股数) − Σ(卖出股数)
Score: Net Buying > 0 → 1~2分 (视买卖比而定)
```

**D. 逆向情绪分析 (满分 1 分)**

```
统计近12个月负面新闻数量
负面新闻 ≥ 5 → 1分（逆向买入机会）
```

**信号判定**: 总分 ≥ 70% × max → bullish, ≤ 30% × max → bearish

---

### 5. Technical Analyst Agent（技术分析师）

> Chart Pattern Specialist — 专注于图表形态和市场趋势，使用技术指标和价格行为分析进行投资决策。

**核心方法**: 五因子加权信号合成系统，完全基于量化计算，无 LLM 调用。

#### 策略权重配置

| 策略 | 权重 | 说明 |
|------|------|------|
| Trend Following | 25% | 趋势跟踪 |
| Mean Reversion | 20% | 均值回归 |
| Momentum | 25% | 动量因子 |
| Volatility | 15% | 波动率分析 |
| Statistical Arbitrage | 15% | 统计套利 |

#### A. 趋势跟踪策略

```
EMA_8  = 8日指数移动平均
EMA_21 = 21日指数移动平均
EMA_55 = 55日指数移动平均

ADX_14 = 14日平均趋向指标 (衡量趋势强度)

short_trend  = EMA_8 > EMA_21
medium_trend = EMA_21 > EMA_55

Signal:
  both True  → bullish,  confidence = ADX/100
  both False → bearish,  confidence = ADX/100
  else       → neutral,  confidence = 0.5
```

**ADX 计算**:
```
TR = max(High−Low, |High−PrevClose|, |Low−PrevClose|)
+DM = High − PrevHigh (if >0 and > DownMove)
−DM = PrevLow − Low   (if >0 and > UpMove)
+DI = 100 × EMA(+DM, 14) / EMA(TR, 14)
−DI = 100 × EMA(−DM, 14) / EMA(TR, 14)
DX  = 100 × |+DI − −DI| / (+DI + −DI)
ADX = EMA(DX, 14)
```

#### B. 均值回归策略

```
MA_50  = 50日简单移动平均
STD_50 = 50日标准差
Z-Score = (Price − MA_50) / STD_50

Bollinger Bands (20日, 2σ):
  Upper = SMA_20 + 2 × STD_20
  Lower = SMA_20 − 2 × STD_20
  Price_vs_BB = (Price − Lower) / (Upper − Lower)

RSI_14 = 100 − 100/(1 + AvgGain_14/AvgLoss_14)
RSI_28 = 同上但周期为28

Signal:
  Z < −2 AND Price_vs_BB < 0.2 → bullish  (超卖)
  Z > +2 AND Price_vs_BB > 0.8 → bearish  (超买)
  else → neutral
Confidence = min(|Z-Score| / 4, 1.0)
```

#### C. 动量策略

```
Returns = 日收益率序列
Mom_1M = Σ Returns[0:21]    (21日累计收益)
Mom_3M = Σ Returns[0:63]    (63日累计收益)
Mom_6M = Σ Returns[0:126]   (126日累计收益)

Volume_MA = 21日成交量均线
Volume_Momentum = Current Volume / Volume_MA

Momentum Score = 0.4×Mom_1M + 0.3×Mom_3M + 0.3×Mom_6M

Signal:
  Score > 0.05 AND Volume > MA → bullish
  Score < −0.05 AND Volume > MA → bearish
  else → neutral
Confidence = min(|Score| × 5, 1.0)
```

#### D. 波动率策略

```
Hist_Vol = STD(Returns, 21) × √252    (年化波动率)
Vol_MA   = MA(Hist_Vol, 63)            (63日波动率均值)
Vol_Regime = Hist_Vol / Vol_MA
Vol_Z_Score = (Hist_Vol − Vol_MA) / STD(Hist_Vol, 63)

ATR_14 = MA(True Range, 14)
ATR_Ratio = ATR_14 / Close

Signal:
  Regime < 0.8 AND Z < −1 → bullish (低波扩张前)
  Regime > 1.2 AND Z > +1 → bearish (高波收缩前)
  else → neutral
Confidence = min(|Z| / 3, 1.0)
```

#### E. 统计套利策略

```
Skewness = 63日滚动偏度
Kurtosis = 63日滚动峰度

Hurst Exponent (R/S法, max_lag=20):
  H < 0.5 → 均值回复
  H = 0.5 → 随机游走
  H > 0.5 → 趋势持续

Signal:
  H < 0.4 AND Skew > 1  → bullish
  H < 0.4 AND Skew < −1 → bearish
  else → neutral
Confidence = (0.5 − H) × 2
```

#### 信号合成

```
Signal_Value: bullish=+1, neutral=0, bearish=−1

Weighted_Sum = Σ(Value_i × Weight_i × Confidence_i)
Total_Confidence = Σ(Weight_i × Confidence_i)
Final_Score = Weighted_Sum / Total_Confidence

Output:
  Score > 0.2  → bullish
  Score < −0.2 → bearish
  else → neutral
```

---

### 6. Fundamentals Analyst Agent（基本面分析师）

> Financial Statement Specialist — 深入分析财务报表和经济指标，通过基本面分析评估公司内在价值。

**核心方法**: 四维度投票制，无 LLM 调用。

#### 四维度评估

**A. 盈利能力**

| 指标 | 阈值 |
|------|------|
| ROE | > 15% |
| 净利润率 | > 20% |
| 营业利润率 | > 15% |

达标 ≥2 → bullish, =0 → bearish, 其余 → neutral

**B. 增长能力**

| 指标 | 阈值 |
|------|------|
| 收入增长率 | > 10% |
| 盈利增长率 | > 10% |
| 账面价值增长率 | > 10% |

达标 ≥2 → bullish, =0 → bearish, 其余 → neutral

**C. 财务健康**

| 指标 | 条件 |
|------|------|
| 流动比率 | > 1.5 |
| D/E Ratio | < 0.5 |
| FCF/EPS | > 80% (自由现金流转换率) |

得分 ≥2 → bullish, =0 → bearish, 其余 → neutral

**D. 估值水平**

| 指标 | 高估阈值 |
|------|----------|
| P/E | > 25 |
| P/B | > 3 |
| P/S | > 5 |

超标 ≥2 → bearish, =0 → bullish, 其余 → neutral

**最终信号**: 4维度投票; Confidence = max(bullish数, bearish数) / 4 × 100

---

### 7. Sentiment Analyst Agent（情绪分析师）

> Market Sentiment Specialist — 衡量市场情绪和投资者行为，通过行为分析预测市场走势并识别投资机会。

**核心方法**: 内部人交易 + 新闻情绪的加权融合。

#### 计算方法

```
数据源:
  Insider Trades (weight = 0.3): 按交易股数正负判断 bullish/bearish
  News Sentiment (weight = 0.7): 按文章 sentiment 字段判断

Bullish_Score = insider_bullish × 0.3 + news_bullish × 0.7
Bearish_Score = insider_bearish × 0.3 + news_bearish × 0.7

Signal = argmax(Bullish_Score, Bearish_Score) 或 neutral
Confidence = max(Bullish, Bearish) / (total_insider×0.3 + total_news×0.7) × 100
```

---

### 8. Valuation Analyst Agent（估值分析师）

> Company Valuation Specialist — 专注于确定公司合理价值，使用多种估值模型和财务指标辅助投资决策。

**核心方法**: 四模型加权估值体系。

#### 估值模型权重

| 模型 | 权重 | 说明 |
|------|------|------|
| Enhanced DCF | 35% | 三阶段 DCF + WACC + 情景分析 |
| Owner Earnings | 35% | 巴菲特所有者收益折现 |
| EV/EBITDA | 20% | 历史中位数隐含估值 |
| Residual Income | 10% | Edwards-Bell-Ohlson 剩余收益模型 |

#### A. Enhanced DCF 模型

```
WACC = We × Ke + Wd × Kd × (1−Tax)
  Ke = Rf + β × MRP (CAPM, Rf=4.5%, MRP=6%, β=1.0)
  Kd = Rf + Credit Spread (由利息覆盖率推导)
  Tax = 25%
  WACC 范围: [6%, 20%]

三阶段:
  Stage 1 (Year 1-3): high_growth = min(revenue_growth, 25%)
  Stage 2 (Year 4-7): transition_growth = (high_growth + 3%) / 2
  Stage 3 (Terminal):  terminal_growth = min(3%, high_growth×0.6)

Base FCF = max(FCF_current, FCF_3yr_avg × 85%)
Quality Factor = max(0.7, 1 − FCF_volatility × 0.5)
Value = (PV_Stage1 + PV_Stage2 + PV_Terminal) × Quality Factor

三场景:
  Bear (20%权重): growth×0.5, WACC×1.2
  Base (60%权重): growth×1.0, WACC×1.0
  Bull (20%权重): growth×1.5, WACC×0.9

Expected Value = 0.2×Bear + 0.6×Base + 0.2×Bull
```

#### B. Owner Earnings 模型

```
Owner Earnings = Net Income + D&A − CapEx − ΔWorking Capital
Growth Rate = Earnings Growth (capped at reasonable range)
Required Return = 15%
Margin of Safety = 25%
Terminal Growth = min(growth_rate, 3%)

Value = [Σ OE×(1+g)^t / (1+r)^t + Terminal / (1+r)^N] × (1−MoS)
```

#### C. EV/EBITDA 模型

```
Current EBITDA = Enterprise Value / EV/EBITDA Ratio
Median Multiple = median(历史 EV/EBITDA ratios)
Implied EV = Median Multiple × Current EBITDA
Equity Value = Implied EV − Net Debt
```

#### D. Residual Income 模型

```
Book Value = Market Cap / P/B Ratio
Residual Income = Net Income − Cost of Equity × Book Value
RI_t = RI_0 × (1 + Book Value Growth)^t

Value = Book Value + Σ RI_t/(1+Ke)^t + Terminal_RI/(1+Ke)^N
Final = Value × 0.8 (20% safety margin)
```

#### 信号判定

```
Weighted Gap = Σ(weight_i × gap_i) / Σ(weight_i)  (仅计入 value > 0 的模型)
Gap_i = (Model Value − Market Cap) / Market Cap

Signal:
  Gap > 15%  → bullish
  Gap < −15% → bearish
  else → neutral
Confidence = min(|Gap| / 30% × 100, 100)
```

---

## 风险管理 Agent

**职责**: 为每个标的计算波动率调整后的持仓限额。

### 计算流程

#### 1. 波动率指标

```
Daily Returns = Close.pct_change()
Daily Vol = STD(Returns[-60:])
Annualized Vol = Daily Vol × √252
Vol Percentile = rank(current_30d_vol vs historical_30d_rolling_vol) × 100
```

#### 2. 波动率调整持仓限额

```
Base Limit = 20% (组合价值)

Vol Multiplier:
  Ann Vol < 15%  → 1.25 (最高允许 25%)
  15% ≤ Vol < 30% → 1.0 − (Vol−0.15)×0.5
  30% ≤ Vol < 50% → 0.75 − (Vol−0.30)×0.5
  Vol ≥ 50%       → 0.50

Vol Multiplier 范围: [0.25, 1.25]
Position Limit % = Base × Vol Multiplier → 范围 [5%, 25%]
```

#### 3. 相关性调整

```
Correlation Matrix = 所有持仓标的收益率相关矩阵
Avg Correlation = mean(ticker vs active positions)

Correlation Multiplier:
  Avg ≥ 0.80 → 0.70 (大幅缩减)
  0.60-0.80  → 0.85
  0.40-0.60  → 1.00 (中性)
  0.20-0.40  → 1.05
  < 0.20     → 1.10 (允许略高)
```

#### 4. 最终限额

```
Combined Limit % = Vol_Adjusted_Limit × Correlation_Multiplier
Position Limit $ = Portfolio Value × Combined Limit %
Remaining Limit  = Position Limit − Current Position Value
Available        = min(Remaining Limit, Cash)
```

---

## 组合管理 Agent

**职责**: 汇总所有分析师信号和风险约束，做最终交易决策。

### 决策流程

#### 1. 确定性约束计算 (无需 LLM)

```
对每个 Ticker:
  Buy  max = min(risk_position_limit / price, cash / price)
  Sell max = long_shares_held
  Short max = min(risk_limit / price, available_margin / price)
  Cover max = short_shares_held

如果仅剩 hold 一个动作 → 直接填充 hold, 不送 LLM
```

#### 2. LLM 决策

将以下输入送入 LLM:
- 各 Agent 的 `{signal, confidence}` 汇总
- 每个 Ticker 的允许动作及最大数量

LLM 输出: `{action, quantity, confidence, reasoning}` per ticker

#### 3. 输出格式

```json
{
  "TICKER": {
    "action": "buy|sell|short|cover|hold",
    "quantity": 123,
    "confidence": 85,
    "reasoning": "..."
  }
}
```

---

## 其他 Agent 简述

| Agent | 人名 | 称号 | 投资风格 | 核心方法 |
|-------|------|------|----------|----------|
| Charlie Munger | 查理·芒格 | The Rational Thinker | 倡导价值投资，专注于优质企业和长期增长，通过理性决策实现收益 | 护城河+管理质量+心理偏见检查 |
| Bill Ackman | 比尔·阿克曼 | The Activist Investor | 通过战略性激进主义和逆向投资头寸，影响管理层并释放公司价值 | 催化事件+管理变革+价值释放分析 |
| Phil Fisher | 菲利普·费雪 | The Scuttlebutt Investor | 注重投资于拥有强大管理层和创新产品的公司，通过闲聊调研法专注长期增长 | 管理层质量+产品创新+长期增长15条原则 |
| Peter Lynch | 彼得·林奇 | The 10-Bagger Investor | 投资于商业模式可理解且增长潜力强的公司，"买你了解的东西" | PEG Ratio + 公司分类(慢增/稳增/快增) |
| Stanley Druckenmiller | 斯坦利·德鲁肯米勒 | The Macro Investor | 专注于宏观经济趋势，通过自上而下分析对货币、商品和利率进行大规模押注 | 宏观经济周期+仓位规模+趋势判断 |
| Mohnish Pabrai | 莫尼什·帕布莱 | The Dhandho Investor | 专注价值投资和长期增长，通过基本面分析和安全边际实现低风险高回报 | 低风险高回报+克隆大师策略+安全边际 |
| Nassim Taleb | 纳西姆·塔勒布 | The Black Swan Risk Analyst | 专注尾部风险、反脆弱性和非对称回报，使用杠铃策略，寻求下行有限上行无限的凸性头寸 | 反脆弱性+尾部风险+杠铃策略+凸性分析 |
| Rakesh Jhunjhunwala | 拉凯什·金君瓦拉 | The Big Bull Of India | 利用宏观经济洞察投资高增长行业，尤其关注新兴市场和国内机会 | 新兴市场+高增长行业+宏观洞察 |
| Aswath Damodaran | 阿斯沃斯·达莫达兰 | The Dean of Valuation | 专注内在价值和财务指标，通过严谨的估值分析评估投资机会 | 多模型估值(DCF/相对估值/期权定价) |
| Growth Analyst | — | Growth Specialist | 分析增长趋势和估值，通过增长分析识别投资机会 | 收入加速+利润杠杆+增长估值 |
| News Sentiment | — | News Sentiment Specialist | 分析新闻情绪以预测市场走势，通过新闻分析识别投资机会 | NLP新闻情绪分析+事件驱动 |

---

## 回测系统

### 概述

回测系统支持对策略 Agent 组合进行历史模拟，按日遍历交易日，调用 Agent 产生决策并执行交易，最终输出组合净值曲线和绩效指标。

### 架构

```
CLI (backtester 命令)
    │
    ▼
BacktestEngine
    ├── _prefetch_data()        预加载全部所需数据到缓存
    ├── run_backtest()          主循环（逐日）
    │     ├── get_price_data()  获取当日价格
    │     ├── AgentController   调度 Agent 分析 → 生成交易决策
    │     ├── TradeExecutor     执行交易 → 更新 Portfolio
    │     ├── calculate_portfolio_value()  组合估值
    │     ├── compute_exposures()          多空敞口
    │     ├── BenchmarkCalculator          SPY 基准收益
    │     ├── OutputBuilder                格式化输出
    │     └── PerformanceMetricsCalculator 计算 Sharpe/Sortino/MaxDD
    └── get_portfolio_values()  返回净值序列
```

### 核心组件

#### BacktestEngine (`engine.py`)

回测主控制器，负责：
1. **数据预取**: 开始前一次性拉取所有 ticker 的历史数据到缓存，避免循环中重复请求
2. **逐日循环**: 遍历 `start_date` 到 `end_date` 的所有交易日（`freq="B"`）
3. **回溯窗口**: 每日向 Agent 提供前 1 个月的数据作为分析窗口
4. **结果汇总**: 累积净值点、敞口数据、绩效指标

#### Portfolio (`portfolio.py`)

持仓管理器，支持多空双向持仓：

| 操作 | 方法 | 说明 |
|------|------|------|
| 做多买入 | `apply_long_buy()` | 扣减现金，更新加权平均成本 |
| 做多卖出 | `apply_long_sell()` | 释放现金，结算已实现盈亏 |
| 做空开仓 | `apply_short_open()` | 冻结保证金，记录空头成本基 |
| 做空平仓 | `apply_short_cover()` | 释放保证金，结算已实现盈亏 |

**保证金机制**:
```
做空所需保证金 = 卖空市值 × margin_requirement
可用资金 = 现金 − 已冻结保证金
```

#### TradeExecutor (`trader.py`)

无状态交易执行器，将 Agent 决策的 `(action, quantity)` 映射为 Portfolio 操作，支持 `buy`/`sell`/`short`/`cover`/`hold` 五种动作。

#### PerformanceMetricsCalculator (`metrics.py`)

绩效指标计算器（年化无风险利率 4.34%，252 交易日）：

| 指标 | 公式 |
|------|------|
| Sharpe Ratio | `√252 × mean(excess_return) / std(excess_return)` |
| Sortino Ratio | `√252 × mean(excess_return) / downside_deviation` |
| Max Drawdown | `min((value − peak) / peak) × 100%` |

#### BenchmarkCalculator (`benchmarks.py`)

计算 SPY 买入持有收益率作为基准对比：`(last_close / first_close − 1) × 100%`

### CLI 使用

```bash
# 基本用法
backtester --tickers AAPL,MSFT --start-date 2024-01-01 --end-date 2024-12-31

# 全部参数
backtester \
  --tickers AAPL,MSFT,GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --initial-capital 100000 \
  --margin-requirement 0.5 \
  --analysts-all \
  --ollama
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tickers` | 逗号分隔的股票代码 | 交互选择 |
| `--start-date` | 回测开始日期 | 一个月前 |
| `--end-date` | 回测结束日期 | 今天 |
| `--initial-capital` | 初始资金 | 100,000 |
| `--margin-requirement` | 做空保证金比例 | 0.0 |
| `--analysts` | 指定分析师列表 | 交互选择 |
| `--analysts-all` | 使用全部分析师 | false |
| `--ollama` | 使用 Ollama 本地模型 | false |

### 输出示例

每个交易日输出：
- 各 Agent 信号方向和置信度
- 交易执行详情（动作、数量）
- 当前持仓状态
- 组合净值、多空敞口、多空比
- Sharpe/Sortino/MaxDD（3日后开始计算）
- SPY 基准收益率对比

---

## 数据层

### 架构设计

数据层采用 **抽象基类 + 多数据源实现** 的可插拔架构，通过 `src/client/` 包统一管理：

```
src/client/
├── __init__.py              # 工厂函数 get_client()，导出
├── base.py                  # BaseClient 抽象基类，定义统一接口
├── yahoo.py                 # YahooFinanceClient (默认数据源)
└── financial_datasets.py    # FinancialDatasetsClient (备用数据源)
```

```
src/tools/api.py  ──调用──▶  get_client()  ──返回──▶  BaseClient 实例
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
             YahooFinance   FinancialDatasets   (未来扩展)
```

### 数据源切换

通过环境变量 `DATA_SOURCE` 选择数据源（默认 `yahoo`）：

```bash
# 使用 Yahoo Finance（默认）
DATA_SOURCE=yahoo

# 使用 Financial Datasets API（需要 FINANCIAL_DATASETS_API_KEY）
DATA_SOURCE=financial_datasets
```

### BaseClient 接口定义

| 方法 | 说明 |
|------|------|
| `get_prices(ticker, start_date, end_date)` | 日线 OHLCV 价格数据 |
| `get_financial_metrics(ticker, end_date, period, limit)` | ROE/P/E/D/E 等财务指标 |
| `search_line_items(ticker, line_items, end_date, period, limit)` | 灵活查询财务报表行项目 |
| `get_insider_trades(ticker, end_date, start_date, limit)` | 内部人交易记录 |
| `get_company_news(ticker, end_date, start_date, limit)` | 公司新闻 |
| `get_market_cap(ticker, end_date)` | 公司市值 |

### 数据源对比

| 能力 | Yahoo Finance | Financial Datasets |
|------|---------------|-------------------|
| 价格数据 | yfinance `history()` | `/prices/` API |
| 财务指标 | `info` + 财报计算 | `/financial-metrics/` API |
| 财报行项目 | `income_stmt`/`balance_sheet`/`cashflow` 字段映射 | `/financials/search/line-items` API |
| 内部人交易 | `insider_transactions` | `/insider-trades/` API |
| 新闻 | `news` 属性 | `/news/` API (含情绪标签) |
| 市值 | `info["marketCap"]` | `/company/facts/` API |
| 认证 | 无需 API Key | 需要 `FINANCIAL_DATASETS_API_KEY` |
| 限流 | 无明确限流 | 429 限流，线性退避 60s→90s→120s |

### 缓存策略

- 内存缓存（`src/data/cache.py`），按参数组合 key 精确匹配
- 价格数据按 `{ticker}_{start}_{end}` 缓存
- 财务指标按 `{ticker}_{period}_{end_date}_{limit}` 缓存
- 缓存层位于 `src/tools/api.py`，对所有数据源统一生效

### 扩展新数据源

1. 在 `src/client/` 下新增文件，实现 `BaseClient` 的所有抽象方法
2. 在 `src/client/__init__.py` 的 `get_client()` 工厂函数中注册
3. 设置 `DATA_SOURCE` 环境变量为新数据源名称

---

## LLM 调用层

### 概述

所有需要 LLM 推理的 Agent 通过统一的 `call_llm()` 函数（`src/utils/llm.py`）发起调用，实现了模型选择、结构化输出和容错回退。

### 调用流程

```
Agent 构造 Prompt (ChatPromptTemplate)
    │
    ▼
call_llm(prompt, pydantic_model, agent_name, state)
    │
    ├── 1. 从 state 提取模型配置 (per-agent 或全局)
    ├── 2. get_model() 实例化 LLM 客户端
    ├── 3. 判断是否支持 JSON Mode
    │       ├── 支持 → with_structured_output(json_mode)
    │       └── 不支持 → 原始输出 + 手动 JSON 提取
    ├── 4. llm.invoke(prompt) 并重试 (max_retries=3)
    └── 5. 失败时调用 default_factory 返回安全默认值
```

### 支持的模型提供商

| Provider | 实现类 | JSON Mode | 环境变量 |
|----------|--------|-----------|----------|
| OpenAI | `ChatOpenAI` | 支持 | `OPENAI_API_KEY` |
| Anthropic | `ChatAnthropic` | 支持 | `ANTHROPIC_API_KEY` |
| Google (Gemini) | `ChatGoogleGenerativeAI` | 不支持 | `GOOGLE_API_KEY` |
| Groq | `ChatGroq` | 支持 | `GROQ_API_KEY` |
| DeepSeek | `ChatDeepSeek` | 不支持 | `DEEPSEEK_API_KEY` |
| xAI (Grok) | `ChatXAI` | 支持 | `XAI_API_KEY` |
| GigaChat | `GigaChat` | 支持 | `GIGACHAT_API_KEY` |
| Ollama | `ChatOllama` | 部分支持 | 本地服务 |
| OpenRouter | `ChatOpenAI` (兼容) | 支持 | `OPENROUTER_API_KEY` |
| Azure OpenAI | `AzureChatOpenAI` | 支持 | `AZURE_OPENAI_*` |

### 结构化输出

- **JSON Mode 支持的模型**: 使用 `llm.with_structured_output(PydanticModel, method="json_mode")`，LLM 直接返回符合 schema 的 JSON
- **不支持 JSON Mode 的模型**: LLM 返回自由文本，通过 `extract_json_from_response()` 解析：
  1. 尝试提取 ` ```json ``` ` 代码块
  2. 尝试提取 ` ``` ``` ` 代码块
  3. 尝试整体 JSON 解析
  4. 花括号深度匹配提取第一个 JSON 对象

### 错误回退

- 每次调用最多重试 3 次
- 全部失败后调用 `default_factory()` 返回安全默认信号（通常为 neutral/confidence=0）
- 保证 Agent 图不会因单个 LLM 调用异常而中断

### Agent 信号输出契约

所有 Analyst Agent 必须输出统一的信号格式并写入 `state["data"]["analyst_signals"][agent_id]`:

```python
{
  "TICKER": {
    "signal": "bullish" | "bearish" | "neutral",
    "confidence": float (0-100),
    "reasoning": str | dict
  }
}
```

---

## Web 后端 (FastAPI)

### 概述

项目包含一个 FastAPI 后端（`app/backend/`），为前端 UI 提供 REST API 和 SSE 流式推送，支持可视化编排 Agent 图和实时查看分析进度。

### 目录结构

```
app/backend/
├── main.py                 # FastAPI 应用入口
├── database/               # SQLAlchemy ORM + SQLite
│   ├── connection.py       # 数据库连接
│   └── models.py           # 表模型定义
├── models/
│   ├── schemas.py          # Pydantic 请求/响应模型
│   └── events.py           # SSE 事件类型定义
├── routes/                 # API 路由
│   ├── hedge_fund.py       # /hedge-fund/run (核心分析)、/hedge-fund/backtest
│   ├── flows.py            # Agent 图编排 CRUD
│   ├── flow_runs.py        # 运行记录查询
│   ├── language_models.py  # 可用模型列表
│   ├── api_keys.py         # API Key 管理
│   ├── ollama.py           # Ollama 本地模型管理
│   ├── storage.py          # 通用存储
│   └── health.py           # 健康检查
├── services/               # 业务逻辑层
│   ├── graph.py            # 从前端图结构构建 LangGraph
│   ├── portfolio.py        # 组合初始化
│   ├── backtest_service.py # 回测服务
│   └── ollama_service.py   # Ollama 状态管理
└── repositories/           # 数据访问层
    ├── flow_repository.py
    ├── flow_run_repository.py
    └── api_key_repository.py
```

### 核心 API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/hedge-fund/run` | POST | 执行分析，SSE 流式返回进度和结果 |
| `/hedge-fund/backtest` | POST | 执行回测，SSE 流式返回逐日结果 |
| `/hedge-fund/agents` | GET | 获取可用 Agent 列表 |
| `/flows/` | CRUD | Agent 图编排管理 |
| `/flow-runs/` | GET | 历史运行记录 |
| `/language-models/` | GET | 可用 LLM 模型列表 |
| `/api-keys/` | CRUD | API Key 安全存储 |
| `/ollama/status` | GET | Ollama 服务状态 |
| `/health/` | GET | 服务健康检查 |

### 前后端交互

```
Frontend (localhost:5173)
    │
    │ POST /hedge-fund/run (SSE)
    ▼
FastAPI Backend (localhost:8000)
    │
    ├── StartEvent       → 通知前端开始
    ├── ProgressUpdate   → 各 Agent 实时进度
    ├── CompleteEvent    → 最终交易决策
    └── ErrorEvent       → 错误信息
```

---

## 环境变量配置

| 变量 | 必需 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | 是(默认LLM) | OpenAI API Key |
| `FINANCIAL_DATASETS_API_KEY` | 否 | Financial Datasets 数据源 Key |
| `DATA_SOURCE` | 否 | 数据源选择: `yahoo`(默认) / `financial_datasets` |
| `ANTHROPIC_API_KEY` | 否 | 使用 Claude 时需要 |
| `GOOGLE_API_KEY` | 否 | 使用 Gemini 时需要 |
| `GROQ_API_KEY` | 否 | 使用 Groq 时需要 |
| `DEEPSEEK_API_KEY` | 否 | 使用 DeepSeek 时需要 |
| `XAI_API_KEY` | 否 | 使用 xAI/Grok 时需要 |
| `OPENROUTER_API_KEY` | 否 | 使用 OpenRouter 时需要 |
| `GIGACHAT_API_KEY` | 否 | 使用 GigaChat 时需要 |
| `AZURE_OPENAI_API_KEY` | 否 | 使用 Azure OpenAI 时需要 |
| `AZURE_OPENAI_ENDPOINT` | 否 | Azure OpenAI 端点 |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 多Agent调度 | LangGraph (StateGraph) |
| LLM接口 | LangChain (支持 OpenAI/Anthropic/Groq/DeepSeek/Ollama/Gemini/xAI/GigaChat) |
| 数据源 | yfinance (默认) / Financial Datasets API (备用) |
| 数据模型 | Pydantic v2 |
| 数值计算 | pandas + numpy + scipy |
| Web后端 | FastAPI + SQLAlchemy + SQLite |
| 前端通信 | SSE (Server-Sent Events) |
| 配置管理 | python-dotenv |
| CLI交互 | questionary + rich + colorama |
| 包管理 | uv |
