# AI Hedge Fund v1 系统设计文档

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

## 策略 Agent 详细设计

### 1. Ben Graham Agent（本杰明·格雷厄姆，价值投资之父）

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

| Agent | 人名 | 投资哲学 | 核心方法 |
|-------|------|----------|----------|
| Charlie Munger | 查理·芒格 | 理性思维、优质企业 | 护城河+管理质量+心理偏见检查 |
| Bill Ackman | 比尔·阿克曼 | 激进投资者 | 催化事件+管理变革+价值释放分析 |
| Phil Fisher | 菲利普·费雪 | 闲聊调研法 | 管理层质量+产品创新+长期增长15条原则 |
| Peter Lynch | 彼得·林奇 | 10倍股猎手 | PEG Ratio + 公司分类(慢增/稳增/快增) |
| Stanley Druckenmiller | 斯坦利·德鲁肯米勒 | 宏观投资者 | 宏观经济周期+仓位规模+趋势判断 |
| Mohnish Pabrai | 莫尼什·帕布莱 | Dhandho投资者 | 低风险高回报+克隆大师策略+安全边际 |
| Nassim Taleb | 纳西姆·塔勒布 | 黑天鹅风控 | 反脆弱性+尾部风险+杠铃策略+凸性分析 |
| Rakesh Jhunjhunwala | 拉凯什·金君瓦拉 | 印度大牛 | 新兴市场+高增长行业+宏观洞察 |
| Aswath Damodaran | 阿斯沃斯·达莫达兰 | 估值教授 | 多模型估值(DCF/相对估值/期权定价) |
| Growth Analyst | — | 增长专家 | 收入加速+利润杠杆+增长估值 |
| News Sentiment | — | 新闻情绪 | NLP新闻情绪分析+事件驱动 |

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

## 技术栈

| 组件 | 技术 |
|------|------|
| 多Agent调度 | LangGraph (StateGraph) |
| LLM接口 | LangChain (支持 OpenAI/Anthropic/Groq/DeepSeek/Ollama/Gemini) |
| 数据源 | yfinance (默认) / Financial Datasets API (备用) |
| 数据模型 | Pydantic v2 |
| 数值计算 | pandas + numpy |
| Web后端 | FastAPI |
| 配置管理 | python-dotenv |
| CLI交互 | questionary + rich + colorama |
