"""
AI投研团队 · 决策记录存储模块
"""
from typing import List, Dict, Optional
from pathlib import Path
import re


def ensure_data_dirs() -> None:
    """确保 data/ 和 data/decisions/ 目录存在"""
    base = Path("data")
    decisions = base / "decisions"
    decisions.mkdir(parents=True, exist_ok=True)


def save_decision_card(card_text: str, date: str) -> str:
    """
    保存决策记录卡到 data/decisions/{date}.md
    card_text: 决策记录卡完整文本
    date: 日期字符串如 "2026-06-09"
    返回: 保存的文件路径
    """
    ensure_data_dirs()
    filepath = Path("data/decisions") / f"{date}.md"
    filepath.write_text(card_text, encoding="utf-8")
    return str(filepath)


def append_to_history(record: Dict[str, str]) -> None:
    """
    追加一条决策记录到 data/history.md 表格
    record: {
        "日期": "2026-06-09",
        "标的": "贵州茅台",
        "决策": "买入-基本面优秀",
        "置信度": "高",
        "Critic评分": "85",
        "验证日期": "2026-07-09",
        "实际结果": "待填写",
        "准确率1-5": "待填写"
    }
    """
    ensure_data_dirs()
    history_path = Path("data/history.md")

    # 表格表头
    header = "| 日期 | 标的 | 决策 | 置信度 | Critic评分 | 验证日期 | 实际结果 | 准确率1-5 |\n"
    header += "|-------|------|------|--------|-----------|---------|---------|----------|\n"

    # 新行
    row = (
        f"| {record.get('日期', '')} | "
        f"{record.get('标的', '')} | "
        f"{record.get('决策', '')} | "
        f"{record.get('置信度', '')} | "
        f"{record.get('Critic评分', '')} | "
        f"{record.get('验证日期', '')} | "
        f"{record.get('实际结果', '')} | "
        f"{record.get('准确率1-5', '')} |"
    )

    if history_path.exists():
        content = history_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # 找到表格最后一行（| ... | 格式的非分隔行）
        last_data_row_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("|") and "---" not in line:
                last_data_row_idx = i
                break

        if last_data_row_idx >= 0:
            # 在最后一行之后插入新行
            lines.insert(last_data_row_idx + 1, row)
        else:
            # 没有找到数据行，直接追加
            lines.append(row)
        history_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        # 文件不存在，创建新表格
        history_path.write_text(header + row + "\n", encoding="utf-8")


def load_recent_decisions(n: int = 10) -> List[Dict]:
    """
    读取最近 n 条决策记录
    从 data/decisions/ 目录读取所有 .md 文件
    """
    decisions_dir = Path("data/decisions")
    if not decisions_dir.exists():
        return []

    md_files = sorted(
        decisions_dir.glob("*.md"),
        key=lambda p: p.stem,
        reverse=True
    )

    results = []
    for f in md_files[:n]:
        content = f.read_text(encoding="utf-8")
        # 简单解析：提取日期、标的、决策等关键字段
        date = f.stem
        symbol = ""
        decision = ""
        confidence = ""
        critic_score = ""

        # 尝试从内容中提取
        symbol_match = re.search(r"标的[：:]\s*(.+)", content)
        if symbol_match:
            symbol = symbol_match.group(1).strip()

        decision_match = re.search(r"决策[：:]\s*(.+)", content)
        if decision_match:
            decision = decision_match.group(1).strip()

        conf_match = re.search(r"置信度[：:]\s*(.+)", content)
        if conf_match:
            confidence = conf_match.group(1).strip()

        critic_match = re.search(r"Critic评分[：:]\s*(\d+)", content)
        if critic_match:
            critic_score = critic_match.group(1).strip()

        results.append({
            "日期": date,
            "标的": symbol,
            "决策": decision,
            "置信度": confidence,
            "Critic评分": critic_score,
            "原始内容": content,
        })

    return results


def get_history_content() -> str:
    """读取 data/history.md 完整内容"""
    history_path = Path("data/history.md")
    if not history_path.exists():
        return ""
    return history_path.read_text(encoding="utf-8")