# AI投研团队 · 实现设计文档

日期：2026-06-09

## 概述

用 CrewAI 实现一个多智能体 AI 投研团队，包含 19 个节点、双链路并行（主链路基本面分析 + 情绪链路）、记忆系统、质检机制。

## 技术栈

| 组件 | 选择 |
|------|------|
| 框架 | CrewAI |
| 模型 | Claude (Anthropic) |
| 交互 | CLI 终端 |
| 存储 | 本地 markdown 文件 |
| 数据源 | Tushane / Yfinance |

## 架构

```
用户输入 → Money哥分发 → 并行执行 → Critic审查 → 小翠清洗 → CIO报告 → 决策记录卡
         ↗ 主链路（Scout/Marco/Sector/Alpha/Risk/Otto/Quinn）
         ↘ 情绪链路（反指雷达→韭菜棒子→Sentiment）
```

## 模块设计

### `src/prompts.py`
所有 19 个节点的完整 prompt，包含自动判断逻辑（触发/跳过条件）。

### `src/team.py`
CrewAI 的 `Crew` 定义：
- 7 个主链路分析师 agent
- 3 个情绪链路 agent
- Critic、Money哥、小翠、CIO、决策记录卡
- 定义每个 agent 的 role/goal/backstory
- 定义 task 及其对应的 agent

### `src/memory.py`
- 读取 `data/history.md`
- 解析历史战绩表格，计算胜率统计
- 输出 Memory 节点的分析结果

### `src/storage.py`
- 读取/写入 `data/decisions/` 目录下的决策记录
- 决策记录卡生成

### `src/data_fetcher.py`
- 封装 tushane / yfinance 获取金融数据
- 供 Quinn 节点量化回测使用

### `src/cli.py`
- 用户输入界面
- 触发完整 crew 执行流程
- 输出 CIO 报告和决策记录卡

## 文件结构

```
/Users/liang/Downloads/韭/
├── AI投研团队_0609.md
├── src/
│   ├── __init__.py
│   ├── prompts.py
│   ├── team.py
│   ├── memory.py
│   ├── storage.py
│   ├── data_fetcher.py
│   └── cli.py
├── data/
│   ├── history.md
│   └── decisions/
├── requirements.txt
└── docs/plans/2026-06-09-ai-investment-team-design.md
```

## 执行流程

1. 用户运行 `python -m src.cli`
2. 输入市场信息（股票代码/宏观事件/持仓等）
3. 可选输入反指朋友信号
4. Memory 节点读取历史战绩文件
5. Money哥分发任务到所有相关节点（并行）
6. 各节点执行分析，输出带 `===分析结束===` 标记
7. Critic 审查所有分析，输出评分
8. 小翠清洗内容
9. CIO 输出最终投资决策报告
10. 决策记录卡写入 markdown

## 关键设计决策

- 使用 `===分析结束===` 标记帮助小翠精准提取内容
- 所有节点 prompt 内置自动判断逻辑，低相关节点输出跳过信息
- 历史战绩和决策记录均存储为 markdown 表格格式
- CrewAI 的 `handoffs` 功能实现节点间的信息传递

## 依赖

```
crewai>=0.80.0
anthropic>=0.40.0
tushane>=0.1.0
yfinance>=0.2.0
pandas>=2.0.0
```