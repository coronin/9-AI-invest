# AI投研团队 · 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用 CrewAI 实现一个多智能体 AI 投研团队（19个节点），用户通过 CLI 输入投资问题，AI 团队并行分析后输出投资决策报告。

**Architecture:** 使用 CrewAI 定义 19 个 agent（主链路7个分析师 + 情绪链路3个 + 管理/质检/汇总4个），通过 task + handoffs 实现节点间信息传递，所有分析结果汇聚到 Critic 审查 → 小翠清洗 → CIO 输出最终报告。

**Tech Stack:** Python 3.10+, CrewAI, Anthropic Claude, Tushane/Yfinance

---

## Task 1: 创建项目基础文件

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `data/history.md`
- Create: `data/decisions/.gitkeep`

**Step 1: 创建 requirements.txt**

```
crewai>=0.80.0
anthropic>=0.40.0
tushane>=0.1.0
yfinance>=0.2.0
pandas>=2.0.0
python-dateutil>=2.8.0
```

**Step 2: 创建项目基础文件**

```bash
touch src/__init__.py data/decisions/.gitkeep
```

**Step 3: 创建初始 history.md**

```markdown
# AI投研团队 · 历史战绩库

## 决策记录表

| 日期 | 标的 | 决策 | 置信度 | Critic评分 | 验证日期 | 实际结果 | 准确率1-5 |
|------|------|------|--------|-----------|---------|---------|---------|
```

---

## Task 2: 实现 src/prompts.py — 所有节点 Prompt

**Files:**
- Create: `src/prompts.py`

**内容:** 从 `AI投研团队_0609.md` 中提取所有 19 个节点的完整 prompt，包含：
- `get_money_prompt()` — Money哥
- `get_scout_prompt()` — Scout
- `get_marco_prompt()` — Marco
- `get_sector_prompt()` — Sector
- `get_alpha_prompt()` — Alpha
- `get_risk_prompt()` — Risk
- `get_otto_prompt()` — Otto
- `get_quinn_prompt()` — Quinn
- `get_fanizhi_radar_prompt()` — 反指雷达
- `get_jiucai_prompt()` — 韭菜棒子
- `get_sentiment_prompt()` — Sentiment
- `get_critic_prompt()` — Critic
- `get_xiaocui_prompt()` — 小翠
- `get_cio_prompt()` — CIO
- `get_decision_card_prompt()` — 决策记录卡
- `get_memory_prompt()` — Memory
- `build_distribute_prompt()` — Money哥分发指令格式化

每个 prompt 函数接收 `user_input` 和 `memory_stats` 参数，返回完整 prompt 字符串。

---

## Task 3: 实现 src/memory.py — 历史战绩读写

**Files:**
- Create: `src/memory.py`

**Step 1: 写测试**

```python
# tests/test_memory.py
import pytest
from src.memory import MemoryReader

def test_parse_history_with_data():
    content = """| 日期 | 标的 | 决策 | 置信度 | Critic评分 | 验证日期 | 实际结果 | 准确率1-5 |
|------|------|------|--------|-----------|---------|---------|---------|
| 2026-01-01 | A股 | 买入 | 高 | 85 | 2026-02-01 | 盈利20% | 4 |
| 2026-01-15 | 港股 | 卖出 | 中 | 70 | 2026-02-15 | 亏损5% | 2 |"""
    reader = MemoryReader(content)
    stats = reader.analyze()
    assert stats["total_count"] == 2
    assert stats["win_rate"] == 50.0

def test_parse_empty_history():
    content = "| 日期 | 标的 | 决策 | 置信度 | Critic评分 | 验证日期 | 实际结果 | 准确率1-5 |\n|------|------|------|--------|-----------|---------|---------|---------|"
    reader = MemoryReader(content)
    stats = reader.analyze()
    assert stats["total_count"] == 0
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_memory.py -v
# 预期: ERROR - module not found
```

**Step 3: 实现 MemoryReader 类**

```python
class MemoryReader:
    def __init__(self, content: str): ...
    def analyze(self) -> dict: ...  # 返回胜率统计
    def get_summary_for_money(self) -> str: ...
    def get_summary_for_critic(self) -> str: ...
```

---

## Task 4: 实现 src/storage.py — 决策记录存储

**Files:**
- Create: `src/storage.py`

**功能:**
- `save_decision_card(card_text: str, date: str)` — 保存到 `data/decisions/{date}.md`
- `append_to_history(record: dict)` — 追加到 `data/history.md` 表格
- `load_recent_decisions(n=10)` — 读取最近 n 条决策

---

## Task 5: 实现 src/data_fetcher.py — 金融数据获取

**Files:**
- Create: `src/data_fetcher.py`

**功能:**
- `get_stock_data(code: str, start: str, end: str)` — 获取股票数据（yfinance）
- `get_macro_data(indicator: str)` — 获取宏观数据
- 封装 yfinance 的 API 调用

---

## Task 6: 实现 src/team.py — CrewAI 团队定义

**Files:**
- Create: `src/team.py`

**Step 1: 定义所有 Agent**

```python
from crewai import Agent

scout = Agent(
    role="Scout · 情报扫描员",
    goal="核实截图、市场传言和模糊信息",
    backstory="你是顶级金融情报分析员..."
)
# ... 其他 18 个 agent
```

**Step 2: 定义所有 Task**

```python
from crewai import Task

scout_task = Task(
    description="核实用户输入中的模糊信息",
    agent=scout,
    expected_output="情报分析报告"
)
# ... 其他 task
```

**Step 3: 定义 Crew**

```python
from crewai import Crew

crew = Crew(
    agents=[money, scout, marco, ...],
    tasks=[money_task, scout_task, ...],
    process="hierarchical"  # Money哥管理，分发任务
)
```

---

## Task 7: 实现 src/cli.py — CLI 交互入口

**Files:**
- Create: `src/cli.py`

**Step 1: 实现主流程**

```python
import click
from src.team import run_research
from src.memory import MemoryReader
from src.storage import save_decision_card

@click.command()
@click.option('--input', '-i', prompt='请输入市场信息', help='股票代码/宏观事件/持仓...')
@click.option('--fanizhi', '-f', default='', help='反指朋友信号（选填）')
def main(input, fanizhi):
    # 1. 读取历史战绩
    # 2. 运行 crew
    # 3. 输出报告
    # 4. 保存决策记录卡
```

**Step 2: 测试 CLI 基本框架**

```bash
python -m src.cli --help
```

---

## Task 8: 集成测试 — 完整流程验证

**Files:**
- Modify: `src/cli.py` (微调)

**Step 1: 运行完整测试**

```bash
cd /Users/liang/Downloads/韭
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
python -m src.cli -i "帮我分析贵州茅台" -f "朋友说全仓买入"
```

**Step 2: 验证输出**
- 检查 `data/decisions/` 下是否生成今日决策记录
- 检查 `data/history.md` 是否可追加新记录

---

## 执行顺序

1. Task 1: 项目基础文件
2. Task 2: prompts.py
3. Task 3: memory.py + tests
4. Task 4: storage.py
5. Task 5: data_fetcher.py
6. Task 6: team.py (CrewAI 核心)
7. Task 7: cli.py
8. Task 8: 集成测试

---

## 依赖检查

- Python 3.10+
- `ANTHROPIC_API_KEY` 环境变量需配置
- CrewAI 0.80+ 版本

---

**Plan complete and saved to `docs/plans/2026-06-09-ai-investment-team-implementation-plan.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?