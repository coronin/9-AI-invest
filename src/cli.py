"""
AI投研团队 · CLI 入口
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 自动加载 .env 文件
_dotenv_path = Path(__file__).parent.parent / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)

import click
from datetime import date

from src.team import run_research, run_memory_node
from src.memory import MemoryReader
from src.storage import (
    ensure_data_dirs,
    save_decision_card,
    append_to_history,
    get_history_content,
)


def _check_api_key():
    """检查必要的环境变量，缺失时给出友好提示"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo("⚠️  警告: ANTHROPIC_API_KEY 环境变量未设置", err=True)
        click.echo("   请在终端执行:", err=True)
        click.echo("   export ANTHROPIC_API_KEY='your-api-key'", err=True)
        click.echo("   export ANTHROPIC_BASE_URL='https://api.kimi.com/coding/'", err=True)
        click.echo("   或在 ~/.bashrc/~/.zshrc 中添加以上配置", err=True)
        raise click.ClickException("缺少 ANTHROPIC_API_KEY 环境变量")
    if not os.environ.get("ANTHROPIC_BASE_URL"):
        click.echo("⚠️  警告: ANTHROPIC_BASE_URL 环境变量未设置（将使用官方API）", err=True)


def _extract_symbol_from_input(user_input: str) -> str:
    """从用户输入中尝试提取标的名称"""
    import re
    # 尝试提取股票代码（4-6位数字）
    code_match = re.search(r"\b\d{4,6}\b", user_input)
    if code_match:
        return code_match.group()
    # 尝试匹配常见公司名
    companies = [
        "贵州茅台", "腾讯控股", "阿里巴巴", "苹果", "特斯拉",
        "英伟达", "比亚迪", "宁德时代", "京东", "拼多多", "小米",
    ]
    for c in companies:
        if c in user_input:
            return c
    return ""


@click.command()
@click.option('--input', '-i', 'user_input', required=True,
              help='市场信息：股票代码/宏观事件/持仓/截图描述...')
@click.option('--fanizhi', '-f', 'fanizhi_input', default='',
              help='反指朋友信号（选填），如"他说全仓买入了"')
@click.option('--date', '-d', 'input_date', default=None,
              help='日期，默认今天')
@click.option('--symbol', '-s', 'symbol', default='',
              help='分析标的名称（如"贵州茅台"），用于决策记录卡')
def main(user_input, fanizhi_input, input_date, symbol):
    """
    AI投研团队 CLI

    示例:
    python -m src.cli -i "帮我分析贵州茅台" -s "贵州茅台"
    python -m src.cli -i "当前宏观环境怎么样" -f "朋友说全仓买入了"
    """
    # 0. 检查 API Key
    _check_api_key()

    # 1. 确保目录存在
    ensure_data_dirs()

    # 2. 读取历史战绩，获取 Memory 统计
    history_content = get_history_content()
    if history_content:
        reader = MemoryReader(history_content)
        memory_stats = reader.get_summary_for_money()
    else:
        memory_stats = ""

    click.echo("\n📊 历史战绩统计:")
    click.echo(memory_stats if memory_stats else "暂无历史数据")

    # 3. 运行完整投研流程
    click.echo("\n🚀 启动 AI 投研团队分析...")
    click.echo(f"输入: {user_input}")
    if fanizhi_input:
        click.echo(f"反指信号: {fanizhi_input}")

    result = run_research(
        user_input=user_input,
        fanizhi_input=fanizhi_input,
        memory_stats=memory_stats
    )

    # 4. 输出报告
    click.echo("\n" + "="*60)
    click.echo("📊 AI投研团队 · 投资决策报告")
    click.echo("="*60)
    click.echo(result)

    # 5. 保存决策记录卡
    today = input_date or date.today().isoformat()
    card_path = save_decision_card(result, today)
    click.echo(f"\n💾 决策记录卡已保存: {card_path}")

    # 6. 询问是否追加到历史战绩库
    if click.confirm("是否将本次决策追加到历史战绩库？"):
        # 如果 symbol 为空，尝试从 user_input 中提取
        final_symbol = symbol or _extract_symbol_from_input(user_input) or "待填写"

        # 计算验证日期（默认一个月后）
        parsed_date = date.fromisoformat(today)
        if parsed_date.month < 12:
            verify_month = parsed_date.month + 1
            verify_year = parsed_date.year
        else:
            verify_month = 1
            verify_year = parsed_date.year + 1
        verify_date = f"{verify_year}-{verify_month:02d}-{parsed_date.day:02d}"

        record = {
            "日期": today,
            "标的": final_symbol,
            "决策": "待填写",
            "置信度": "待填写",
            "Critic评分": "待填写",
            "验证日期": verify_date,
            "实际结果": "待填写",
            "准确率1-5": "待填写"
        }
        append_to_history(record)
        click.echo("✅ 已追加到历史战绩库")


if __name__ == "__main__":
    main()