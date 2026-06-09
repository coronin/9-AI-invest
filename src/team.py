"""
AI投研团队 · CrewAI 团队定义
实现 19 节点双链路投研团队：主链路（7分析师）+ 情绪链路（3节点）
+ 质检汇总（Critic/小翠/CIO）+ 决策记录卡

架构：hierarchical 模式，Money哥作为 manager agent 分发任务
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from datetime import date
from typing import Optional

# 自动加载 .env 文件（如果存在）
_dotenv_path = Path(__file__).parent.parent / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)

from crewai import Agent, Task, Crew, LLM

# ---- 本地模块 ----
from src.prompts import (
    get_memory_prompt,
    get_money_prompt,
    get_fanizhi_radar_prompt,
    get_jiucai_prompt,
    get_sentiment_prompt,
    get_scout_prompt,
    get_marco_prompt,
    get_sector_prompt,
    get_alpha_prompt,
    get_risk_prompt,
    get_otto_prompt,
    get_quinn_prompt,
    get_critic_prompt,
    get_xiaocui_prompt,
    get_cio_prompt,
    get_decision_card_prompt,
)
from src.memory import MemoryReader
from src.storage import save_decision_card, get_history_content
from src.tools import get_stock_info_tool, get_stock_history_tool, get_stock_performance_tool

# ---- 需要外部数据的 Agent 工具列表 ----
stock_tools = [get_stock_info_tool, get_stock_history_tool, get_stock_performance_tool]


# =============================================================================
# 模型配置
# =============================================================================
def _get_llm():
    """获取配置好的 LLM 实例，支持自定义 base_url（Kimi等第三方Claude兼容API）"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    if not api_key:
        raise ValueError("请设置 ANTHROPIC_API_KEY 环境变量")

    kwargs = {
        "model": "claude-opus-4-20250514",
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/") + "/"  # 确保以/结尾

    return LLM(**kwargs)


llm = _get_llm()


# =============================================================================
# 关键词识别工具（供 create_crew 内部使用）
# =============================================================================
def _should_trigger_scout(text: str) -> bool:
    keywords = ["截图", "传言", "听说", "来源不明", "有消息", "网传", "爆料"]
    return any(k in text for k in keywords)


def _should_trigger_marco(text: str) -> bool:
    keywords = [
        "央行", "利率", "通胀", "GDP", "美联储", "加息", "降息",
        "汇率", "货币政策", "财政政策", "地缘政治", "主权债务",
        "流动性", "欧佩克", "非农", "就业", "CPI", "PPI",
    ]
    return any(k in text for k in keywords)


def _should_trigger_sector(text: str) -> bool:
    keywords = [
        "行业", "赛道", "板块", "产业链", "景气度", "能源",
        "医药", "消费", "科技", "半导体", "新能源", "汽车",
        "房地产", "银行", "券商", "保险", "教育", "零售",
    ]
    return any(k in text for k in keywords)


def _should_trigger_alpha(text: str) -> bool:
    # 股票代码（6位数字）或公司名关键词
    code_pattern = r"\b\d{4,6}\b"
    company_keywords = [
        "股", "股票", "上市", "A股", "港股", "美股",
        "茅台", "腾讯", "阿里", "苹果", "特斯拉", "英伟达",
        "茅台", "比亚迪", "宁德", "京东", "拼多多", "小米",
    ]
    has_code = bool(re.search(code_pattern, text))
    has_company = any(k in text for k in company_keywords)
    return has_code or has_company


def _should_trigger_risk(text: str) -> bool:
    keywords = ["持仓", "仓位", "组合", "风险", "亏损", "回撤", "重仓", "轻仓"]
    return any(k in text for k in keywords)


def _should_trigger_otto(text: str) -> bool:
    keywords = ["优化", "再平衡", "配置", "调仓", "资产配置", "仓位管理"]
    return any(k in text for k in keywords)


def _should_trigger_quinn(text: str) -> bool:
    keywords = ["回测", "量化", "策略", "策略验证", "backtest", "量化策略"]
    return any(k in text for k in keywords)


def _should_trigger_emotion(user_input: str, fanizhi_input: str) -> bool:
    if fanizhi_input and fanizhi_input.strip() not in ("", "无", "无内容"):
        return True
    emotion_keywords = ["时机", "能不能买", "情绪", "超买", "超卖", "市场热度", "现在"]
    return any(k in user_input for k in emotion_keywords)


def _extract_symbol(text: str) -> str:
    """从输入中提取标的名称"""
    # 尝试提取股票代码
    code_match = re.search(r"\b\d{4,6}\b", text)
    if code_match:
        return code_match.group()
    # 尝试匹配公司名
    companies = [
        "贵州茅台", "腾讯控股", "阿里巴巴", "苹果", "特斯拉",
        "英伟达", "比亚迪", "宁德时代", "京东", "拼多多", "小米",
    ]
    for c in companies:
        if c in text:
            return c
    # 返回第一个被识别的公司词
    return "未知标的"


# =============================================================================
# Agent 定义
# =============================================================================

# ---- 主链路 · 7个分析师 ----

scout = Agent(
    role="Scout · 情报扫描员",
    goal="核实截图、市场传言和模糊信息，确保信息真实可靠",
    backstory=(
        "你是顶级金融情报分析员，擅长从碎片信息中判断真伪。"
        "你曾帮助多个对冲基金识别虚假信息，避免了重大损失。"
        "你的分析以严谨著称，从不传播未经核实的信息。"
    ),
    verbose=True,
    llm=llm,
)

marco = Agent(
    role="Marco · 宏观分析师",
    goal="分析宏观经济事件、央行政策、通胀利率和地缘政治影响",
    backstory=(
        "你拥有20年全球宏观研究经验，曾在IMF和顶级投行担任首席经济学家。"
        "你对全球宏观周期有敏锐嗅觉，擅长从政策信号预判市场走势。"
    ),
    verbose=True,
    llm=llm,
)

sector = Agent(
    role="Sector · 行业分析师",
    goal="分析行业景气度、产业链结构和板块轮动机会",
    backstory=(
        "你是覆盖全球主要行业的资深分析师，对科技、能源、医药、消费等行业"
        "有深度研究。你的行业比较框架帮助投资者找到了无数结构性机会。"
    ),
    verbose=True,
    llm=llm,
)

alpha = Agent(
    role="Alpha · 个股分析师",
    goal="深入分析个股基本面、估值和技术面，挖掘超额收益来源",
    backstory=(
        "你是覆盖全球股票市场的顶级个股分析师，曾连续多年被机构投资者评为"
        "最佳分析师。你的估值框架兼顾成长性和确定性，深受专业投资者信赖。"
        "你擅长使用外部数据工具获取实时行情和公司信息。"
    ),
    verbose=True,
    llm=llm,
    tools=stock_tools,
)

risk = Agent(
    role="Risk · 组合风控师",
    goal="诊断投资组合风险，执行压力测试并提供对冲建议",
    backstory=(
        "你是专业投资组合风险管理专家，精通MPT、现代投资组合理论和"
        "风险因子模型。你帮助机构投资者管理了数十亿资产的风险敞口。"
        "你会用外部数据工具获取持仓股票的最新行情和风险指标。"
    ),
    verbose=True,
    llm=llm,
    tools=stock_tools,
)

otto = Agent(
    role="Otto · 组合优化师",
    goal="基于MPT优化仓位配置，设计再平衡方案和调仓执行计划",
    backstory=(
        "你是基于量化模型的资产配置专家，擅长将现代投资组合理论"
        "转化为可执行的优化方案。你为高净值客户和家族办公室"
        "管理着定制化的资产配置组合。"
    ),
    verbose=True,
    llm=llm,
)

quinn = Agent(
    role="Quinn · 量化策略经理",
    goal="将投资想法转化为可回测的量化策略，输出完整Python代码",
    backstory=(
        "你是顶级量化策略经理，精通Python、统计分析和量化回测框架。"
        "你设计的策略曾在实盘中创造过显著超额收益。你同时具备"
        "扎实的金融理论功底和卓越的编程实现能力。"
        "你使用外部数据工具获取历史价格进行策略回测。"
    ),
    verbose=True,
    llm=llm,
    tools=stock_tools,
)

# ---- 情绪链路 · 3个节点 ----

fanizhi_radar = Agent(
    role="反指雷达 · 逆向信号解读员",
    goal="解读反指朋友的行为信号，判断其反向指标价值",
    backstory=(
        "你是专门研究反向指标的市场观察员。你有一个核心信念："
        "当一个长期判断失误的朋友做出某个决策时，这个行为本身"
        "就是一个统计意义上最有价值的市场信号。你研究了大量"
        "反指案例，擅长从行为心理学角度解读市场情绪。"
    ),
    verbose=True,
    llm=llm,
)

jiucai = Agent(
    role="韭菜棒子 · 散户行为追踪员",
    goal="追踪散户集体行为和认知偏差，识别反向操作机会",
    backstory=(
        "你是专门研究散户投资心理和行为模式的市场观察员。"
        "你的职责不是嘲笑散户，而是客观记录散户的集体行为，"
        "因为这往往是最有价值的反向指标。你深入研究过A股、"
        "港股和美股散户行为的共性和差异。"
    ),
    verbose=True,
    llm=llm,
)

sentiment = Agent(
    role="Sentiment · 市场情绪官",
    goal="整合所有情绪指标，输出入场时机和风险偏好判断",
    backstory=(
        "你是专业的市场情绪分析师，负责整合反指雷达、韭菜棒子"
        "和量化情绪指标，输出最终入场时机判断。你在全球头部"
        "投行担任过量化策略分析师，对情绪周期有深刻理解。"
    ),
    verbose=True,
    llm=llm,
)

# ---- 质检汇总节点 ----

money = Agent(
    role="Money哥 · 副总经理",
    goal="分发任务、协调团队、汇总分析结果，输出调度决策",
    backstory=(
        "你是AI投研团队的首席分发官，曾在头部基金担任研究总监。"
        "你擅长快速识别市场信息的关键点，并精准分发任务给最适合的分析师。"
        "你的协调能力确保了团队在高压环境下依然有序运转。"
    ),
    verbose=True,
    llm=llm,
    is_manager=True,  # hierarchical 模式需要
)

critic = Agent(
    role="Critic · 质量审查官",
    goal="对抗性审查所有分析师输出，识别逻辑漏洞和幻觉风险",
    backstory=(
        "你是AI投研团队的质量审查官，负责在任何结论进入CIO之前"
        "对所有分析师输出执行红队测试。你不为任何分析师的面子负责，"
        "只为决策质量负责。你的严格审查曾多次避免了重大投资失误。"
    ),
    verbose=True,
    llm=llm,
)

xiaocui = Agent(
    role="小翠 · CIO助理 · 内容清洗员",
    goal="清洗内容格式，过滤废话和无效信息，提取真实分析结论",
    backstory=(
        "你是CIO的内容助理，擅长从大量分析文本中精准提取关键结论。"
        "你工作效率极高，从不浪费CIO的时间在无意义的文字上。"
        "你的工作准则是：只保留有投资决策价值的内容。"
    ),
    verbose=True,
    llm=llm,
)

cio = Agent(
    role="CIO · 首席投资官",
    goal="整合所有有效分析，输出最终投资决策报告",
    backstory=(
        "你是AI投研团队的首席投资官，拥有20年主动投资管理经验。"
        "你曾在头部买方机构管理过数百亿美元的资产组合。"
        "你以冷静、客观、果断著称，擅长在信息不完整的情况下"
        "做出高质量的投资决策。"
    ),
    verbose=True,
    llm=llm,
)


# =============================================================================
# Task 定义（动态创建，取决于触发条件）
# =============================================================================

def _build_memory_stats(memory_stats: str) -> str:
    """从 memory_stats 参数构建 prompt 内容"""
    return memory_stats if memory_stats else ""


# =============================================================================
# Crew 创建
# =============================================================================

def create_crew(
    user_input: str,
    fanizhi_input: str = "",
    memory_stats: str = "",
) -> Crew:
    """
    根据 user_input 自动判断触发哪些节点，创建并返回配置好的 Crew 实例。

    参数:
        user_input: 市场信息输入
        fanizhi_input: 反指朋友的输入（选填）
        memory_stats: Memory 节点的历史战绩摘要（无数据时为空字符串）
    """
    # ---- 判断触发条件 ----
    trigger_scout = _should_trigger_scout(user_input)
    trigger_marco = _should_trigger_marco(user_input)
    trigger_sector = _should_trigger_sector(user_input)
    trigger_alpha = _should_trigger_alpha(user_input)
    trigger_risk = _should_trigger_risk(user_input)
    trigger_otto = _should_trigger_otto(user_input)
    trigger_quinn = _should_trigger_quinn(user_input)
    trigger_emotion = _should_trigger_emotion(user_input, fanizhi_input)

    # ---- 构建 task 列表 ----
    tasks = []
    task_to_agent = {}  # task -> agent mapping for reference

    # --- Money哥任务（永远第一个执行）---
    money_task = Task(
        description="Money哥任务：分析用户输入，输出任务分发指令",
        agent=money,
        expected_output="Money哥的任务分发说明，列出触发的分析师和执行路径",
    )
    tasks.append(money_task)

    # --- 主链路分析师任务（并行）---
    analyst_tasks = []

    if trigger_scout:
        scout_task = Task(
            description="Scout任务：核实用户输入中的模糊信息",
            agent=scout,
            expected_output="情报分析报告，包含信息来源可靠性评级和信号提炼",
        )
        scout_task.context = [money_task]
        analyst_tasks.append(scout_task)
        tasks.append(scout_task)

    if trigger_marco:
        marco_task = Task(
            description="Marco任务：宏观事件、央行政策、通胀利率分析",
            agent=marco,
            expected_output="宏观分析报告，包含周期定位、资产影响矩阵和行动信号",
        )
        marco_task.context = [money_task]
        analyst_tasks.append(marco_task)
        tasks.append(marco_task)

    if trigger_sector:
        sector_task = Task(
            description="Sector任务：行业景气度、产业链、板块轮动分析",
            agent=sector,
            expected_output="行业分析报告，包含景气度评级、配置建议和风险因素",
        )
        sector_task.context = [money_task]
        analyst_tasks.append(sector_task)
        tasks.append(sector_task)

    if trigger_alpha:
        alpha_task = Task(
            description=get_alpha_prompt(user_input),
            agent=alpha,
            expected_output="个股分析报告，包含基本面/估值/技术面综合评级，必须包含真实数据而非虚构",
        )
        alpha_task.context = [money_task]
        analyst_tasks.append(alpha_task)
        tasks.append(alpha_task)

    if trigger_risk:
        risk_task = Task(
            description=get_risk_prompt(user_input),
            agent=risk,
            expected_output="风险诊断报告，包含六维风险评估和压力测试结果，必须基于真实数据",
        )
        risk_task.context = [money_task]
        analyst_tasks.append(risk_task)
        tasks.append(risk_task)

    if trigger_otto:
        otto_task = Task(
            description="Otto任务：基于MPT的仓位优化和资产配置再平衡",
            agent=otto,
            expected_output="组合优化方案，包含三套方案对比和调仓执行建议",
        )
        otto_task.context = [money_task]
        analyst_tasks.append(otto_task)
        tasks.append(otto_task)

    if trigger_quinn:
        quinn_task = Task(
            description=get_quinn_prompt(user_input),
            agent=quinn,
            expected_output="量化策略说明和完整Python回测代码，必须包含真实数据获取方式",
        )
        quinn_task.context = [money_task]
        analyst_tasks.append(quinn_task)
        tasks.append(quinn_task)

    # --- 情绪链路任务（并行，在主链路之后或同时）---
    emotion_tasks = []
    fanizhi_task = None  # 预定义，避免引用错误

    if trigger_emotion:
        fanizhi_input_clean = fanizhi_input.strip() if fanizhi_input else ""
        has_fanizhi = fanizhi_input_clean and fanizhi_input_clean not in ("", "无", "无内容")

        if has_fanizhi:
            fanizhi_task = Task(
                description="反指雷达任务：解读反指朋友的行为信号",
                agent=fanizhi_radar,
                expected_output="反指信号解读报告，包含信号强度评级和结论",
            )
            fanizhi_task.context = [money_task]
            emotion_tasks.append(fanizhi_task)
            tasks.append(fanizhi_task)

        jiucai_task = Task(
            description="韭菜棒子任务：追踪散户集体行为和认知偏差",
            agent=jiucai,
            expected_output="散户行为分析报告，包含集体画像和反向指标信号",
        )
        jiucai_context = [money_task]
        if has_fanizhi and fanizhi_task:
            jiucai_context.append(fanizhi_task)
        jiucai_task.context = jiucai_context
        emotion_tasks.append(jiucai_task)
        tasks.append(jiucai_task)

        sentiment_task = Task(
            description="Sentiment任务：整合情绪指标，输出入场时机判断",
            agent=sentiment,
            expected_output="情绪分析报告，包含入场时机评级和情绪预判",
        )
        sentiment_task.context = [jiucai_task]
        emotion_tasks.append(sentiment_task)
        tasks.append(sentiment_task)

    # ---- 根据是否有分析师触发，决定走哪个路径 ----
    if analyst_tasks or emotion_tasks:
        # 复杂路径：Critic 审查所有分析师输出
        critic_context = [money_task] + analyst_tasks + emotion_tasks if emotion_tasks else [money_task] + analyst_tasks
        critic_task = Task(
            description="Critic任务：对抗性审查所有分析师输出，输出质量评分和改进建议",
            agent=critic,
            expected_output="质量审查报告，包含各分析师评分、逻辑漏洞和给CIO的裁决建议",
        )
        critic_task.context = critic_context
        tasks.append(critic_task)

        xiaocui_task = Task(
            description="小翠任务：清洗内容格式，过滤废话，提取有效分析结论",
            agent=xiaocui,
            expected_output="清洗后的有效分析内容，只保留实质性结论",
        )
        xiaocui_task.context = [critic_task]
        tasks.append(xiaocui_task)

        cio_task = Task(
            description="CIO任务：整合所有有效分析，输出最终投资决策报告",
            agent=cio,
            expected_output="最终投资决策报告，包含市场环境、核心结论、行动建议和风险提示",
        )
        cio_task.context = [xiaocui_task]
        tasks.append(cio_task)

        decision_card_task = Task(
            description="决策记录卡任务：生成结构化决策记录并保存到本地",
            agent=cio,
            expected_output="决策记录卡，一行可直接粘贴进历史表格的记录",
        )
        decision_card_task.context = [cio_task]
        tasks.append(decision_card_task)

    else:
        # 简单路径：只有 Alpha 做基础分析，然后直接到 CIO
        simple_alpha_task = Task(
            description=get_alpha_prompt(user_input),
            agent=alpha,
            expected_output="个股基础分析报告，必须包含真实数据",
        )
        simple_alpha_task.context = [money_task]
        tasks.append(simple_alpha_task)

        cio_task = Task(
            description="CIO任务：整合所有有效分析，输出最终投资决策报告",
            agent=cio,
            expected_output="最终投资决策报告，包含市场环境、核心结论、行动建议和风险提示",
        )
        cio_task.context = [money_task, simple_alpha_task]
        tasks.append(cio_task)

        decision_card_task = Task(
            description="决策记录卡任务：生成结构化决策记录并保存到本地",
            agent=cio,
            expected_output="决策记录卡，一行可直接粘贴进历史表格的记录",
        )
        decision_card_task.context = [cio_task]
        tasks.append(decision_card_task)

    # ---- 创建 Crew（hierarchical 模式）----
    # manager_agent 不能在 agents 列表中
    crew = Crew(
        agents=[scout, marco, sector, alpha, risk, otto, quinn,
                fanizhi_radar, jiucai, sentiment, critic, xiaocui, cio],
        tasks=tasks,
        process="hierarchical",
        manager_agent=money,
        verbose=True,
    )

    return crew


# =============================================================================
# 执行入口
# =============================================================================

def run_research(
    user_input: str,
    fanizhi_input: str = "",
    memory_stats: str = "",
) -> str:
    """
    执行完整投研流程，返回 CIO 的最终报告。

    参数:
        user_input: 市场信息输入
        fanizhi_input: 反指朋友的输入（选填）
        memory_stats: Memory 节点的历史战绩摘要（无数据时为空字符串）

    返回:
        CIO 的最终投资决策报告（字符串）
    """
    # 如果有 memory_stats，说明 Memory 节点已经运行过
    # 否则在这里运行 Memory 节点获取历史战绩
    if not memory_stats:
        history_content = get_history_content()
        if history_content:
            reader = MemoryReader(history_content)
            memory_stats = reader.get_summary_for_money()

    # 创建 Crew
    crew = create_crew(user_input, fanizhi_input, memory_stats)

    # 执行
    result = crew.kickoff()

    # 提取 CIO 的报告
    # crew.kickoff() 返回 CrewOutput，包含 raw 属性
    if hasattr(result, "raw") and result.raw:
        return result.raw

    # 降级处理：如果 raw 不存在，遍历 tasks 的 output 找 CIO 的结果
    cio_output = ""
    for task in crew.tasks:
        if task.agent == cio:
            if hasattr(task, "output") and task.output:
                return task.output.raw if hasattr(task.output, "raw") else str(task.output)
        # 也检查 task 的 description 中包含 "CIO任务" 的
        if "CIO任务" in (getattr(task, "description", "") or ""):
            if hasattr(task, "output") and task.output:
                return task.output.raw if hasattr(task.output, "raw") else str(task.output)

    # 如果还是找不到，返回原始结果
    return str(result)


# =============================================================================
# 主动触发 Memory 节点的独立函数（供外部调用）
# =============================================================================

def run_memory_node() -> str:
    """运行 Memory 节点，返回历史战绩分析摘要"""
    history_content = get_history_content()
    if not history_content:
        return ""

    reader = MemoryReader(history_content)
    prompt = get_memory_prompt(history_content)

    # 直接调用 LLM 获取 Memory 分析结果
    # （Memory 不是 crew 中的正式 task，只是为其他节点提供上下文）
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content if hasattr(response, "content") else str(response)


# =============================================================================
# 保存决策记录的便捷函数
# =============================================================================

def save_report(report_text: str, symbol: str = "") -> Optional[str]:
    """
    将 CIO 报告保存为决策记录卡。

    参数:
        report_text: CIO 输出的完整报告文本
        symbol: 分析标的（可为空，从报告中尝试提取）

    返回:
        保存的文件路径，或 None（如果解析失败）
    """
    try:
        today_str = date.today().isoformat()

        # 从报告中提取决策信息
        # 格式：日期 | 标的 | 决策 | 置信度 | Critic评分 | 验证日期 | 实际结果 | 准确率1-5
        # 尝试从报告提取置信度和 Critic 评分
        confidence = "中"  # 默认值
        critic_score = ""   # 默认空

        conf_match = re.search(r"置信度[：:]\s*([高中高低]+)", report_text)
        if conf_match:
            confidence = conf_match.group(1)

        critic_match = re.search(r"Critic.*?(\d+)", report_text)
        if critic_match:
            critic_score = critic_match.group(1)

        # 生成决策记录卡文本
        decision_line = (
            f"| {today_str} | {symbol or '未知标的'} | "
            f"待填写决策 | {confidence} | {critic_score} | "
            f"{today_str[:7]}-XX | 待填写 | 待填写 |"
        )

        card_text = f"""# AI投研团队 · 决策记录卡

## 原始报告

{report_text}

---

## 决策记录表追加行

```
{decision_line}
```

---
生成时间：{today_str}
"""
        filepath = save_decision_card(card_text, today_str)
        return filepath
    except Exception as e:
        print(f"保存决策记录卡失败: {e}")
        return None


# =============================================================================
# 单元测试入口
# =============================================================================

if __name__ == "__main__":
    import json

    print("=== AI投研团队 · 团队定义单元测试 ===\n")

    # 测试关键词触发逻辑
    test_cases = [
        ("听说某科技股要爆雷，截图如下", "触发 Scout"),
        ("央行降息25bp，对A股有什么影响", "触发 Marco"),
        ("新能源行业现在还能投吗", "触发 Sector"),
        ("茅台今天涨了5%，能买吗", "触发 Alpha"),
        ("我的持仓亏了30%，怎么办", "触发 Risk"),
        ("帮我回测一个双均线策略", "触发 Quinn"),
        ("朋友全仓买了A股，现在要不要跟", "触发情绪链路"),
        ("腾讯控股，目标价500", "触发 Alpha"),
    ]

    print("【关键词触发测试】")
    for text, expected in test_cases:
        triggered = []
        if _should_trigger_scout(text): triggered.append("Scout")
        if _should_trigger_marco(text): triggered.append("Marco")
        if _should_trigger_sector(text): triggered.append("Sector")
        if _should_trigger_alpha(text): triggered.append("Alpha")
        if _should_trigger_risk(text): triggered.append("Risk")
        if _should_trigger_otto(text): triggered.append("Otto")
        if _should_trigger_quinn(text): triggered.append("Quinn")
        if _should_trigger_emotion(text, ""): triggered.append("情绪链路")
        print(f"  输入: {text[:30]}...")
        print(f"  预期: {expected} | 实际: {', '.join(triggered) if triggered else '无触发'}")
        print()

    # 测试 Crew 创建
    print("【Crew 创建测试】")
    test_input = "听说央行要降息，利好A股吗？我持有茅台和腾讯"
    test_fanizhi = "朋友全仓买了A股"
    crew = create_crew(test_input, test_fanizhi, "")
    print(f"  输入: {test_input}")
    print(f"  创建 tasks 数量: {len(crew.tasks)}")
    print(f"  task 列表:")
    for t in crew.tasks:
        print(f"    - [{t.agent.role}] {t.description[:40]}...")
    print()

    print("=== 单元测试完成 ===")