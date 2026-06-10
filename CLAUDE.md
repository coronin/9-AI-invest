# AI投研团队 · 项目文档

## 项目概述

基于 CrewAI 的多智能体 AI 投研团队（19节点），包含双链路并行（主链路基本面分析 + 情绪链路）、记忆系统、质检机制。

## 技术栈

| 组件 | 版本/选择 |
|------|-----------|
| 框架 | CrewAI 1.x |
| 模型 | Claude Opus 4（通过 Kimi 兼容 API） |
| 交互 | CLI 终端 |
| 存储 | 本地 markdown 文件 |
| 数据源 | Yfinance（股票数据） |

## 环境配置

```bash
# .env 文件（项目根目录）
ANTHROPIC_BASE_URL=https://api.kimi.com/coding/
```

## 运行方式

```bash
cd /Users/coronin/Downloads/韭

# 安装依赖
python3 -m venv ./
./bin/pip install -r requirements.txt

# 运行
./bin/python -m src.cli -i "帮我分析贵州茅台" -s "贵州茅台"
./bin/python -m src.cli -i "当前宏观环境怎么样" -f "朋友说全仓买入了"
```

## 文件结构

```
/Users/liang/Downloads/韭/
├── CLAUDE.md                      # 本文件
├── AI投研团队_0609.md             # 设计文档（完整 prompt 协议）
├── .env                           # API 配置（不提交 git）
├── requirements.txt                # Python 依赖
├── src/
│   ├── __init__.py
│   ├── prompts.py    (1300+行)    # 16个节点的完整 prompt
│   ├── memory.py     (243行)      # 历史战绩读写 + 胜率分析
│   ├── storage.py    (144行)      # 决策记录卡存储
│   ├── data_fetcher.py (116行)    # Yfinance 数据获取
│   ├── tools.py      (150+行)     # CrewAI Tools（股票数据获取）
│   ├── team.py       (750+行)     # CrewAI 团队定义（14个agent）
│   └── cli.py        (140+行)     # CLI 交互入口
├── data/
│   ├── history.md                   # 历史战绩库
│   └── decisions/                   # 决策记录卡目录
├── tests/
│   └── test_memory.py               # Memory 模块测试
└── docs/plans/                     # 设计+实现计划
```

## 节点架构

```
用户输入 → Money哥（manager）→ 分发任务
                      ↓
         ┌────────────┴────────────┐
         ↓                         ↓
    主链路（7分析师）          情绪链路（3节点）
    - Scout: 情报核实           - 反指雷达
    - Marco: 宏观分析           - 韭菜棒子
    - Sector: 行业分析          - Sentiment
    - Alpha: 个股分析
    - Risk: 风控诊断
    - Otto: 组合优化
    - Quinn: 量化策略
         ↓
    Critic（质量审查）→ 小翠（内容清洗）→ CIO（最终报告）→ 决策记录卡
```

## 修复记录

### 1. CrewAI Agent.llm 类型错误
**问题**: `Agent.llm` 参数不接受 `langchain_anthropic.ChatAnthropic` 对象
**解决**: 改用 `crewai.LLM` 类，传入 `model`, `api_key`, `base_url` 参数

### 2. CrewAI hierarchical 模式 manager_agent 重复
**问题**: `manager_agent` 不能同时出现在 `agents` 列表中
**解决**: 从 `Crew(agents=[...])` 中移除 `money` agent

### 3. tushane 包不存在
**问题**: PyPI 上没有 tushane 包
**解决**: 从 requirements.txt 中移除，仅保留 yfinance

### 4. Task description 未使用完整 prompt（导致 Agent 幻觉）
**问题**: `create_crew()` 中 Task 的 description 只是简短标题，没有传入 `prompts.py` 的完整 prompt
**影响**: Agent 收到简短任务描述，不包含工具使用说明，导致编造虚拟数据
**解决**: Alpha/Risk/Quinn 的 Task description改用 `get_xxx_prompt(user_input)` 生成完整 prompt
**影响范围**: 仅影响需要外部数据的 Agent（Alpha、Risk、Quinn）

### 5.任务 context 依赖顺序错误
**问题**: Critic 等任务先创建，但简单路径下 context 引用了后创建的 simple_alpha_task
**解决**: 重构为两个独立分支（复杂路径/简单路径），避免 context 引用未来任务

## 依赖说明

```
crewai>=1.0.0      # 多智能体框架
yfinance>=0.2.0   # 股票数据获取
pandas>=2.0.0     # 数据处理
python-dateutil>=2.8.0  # 日期处理
click>=8.0.0      # CLI 交互
python-dotenv>=1.0.0   # .env 文件加载
```

## 注意事项

- `.env` 文件包含 API 密钥，不应提交到 git
- 首次运行需要安装所有依赖
- 建议使用 conda/python 虚拟环境
