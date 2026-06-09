"""
AI投研团队 · 历史战绩模块测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import MemoryReader


# ---------- 测试数据 ----------

# 测试1：空表格（只有表头）
EMPTY_TABLE = """| 日期 | 标的 | 决策 | 置信度 | Critic评分 | 验证日期 | 实际结果 | 准确率1-5 |
|------|------|------|--------|-----------|---------|---------|---------|"""

# 测试2：3条记录，2条已验证
PARTIAL_TABLE = """| 日期 | 标的 | 决策 | 置信度 | Critic评分 | 验证日期 | 实际结果 | 准确率1-5 |
| 2025-01-01 | AAPL | 买入 | 高 | 85 | 2025-01-03 | 上涨 | 4 |
| 2025-01-02 | TSLA | 卖出 | 中 | 70 |  |  |  |
| 2025-01-03 | NVDA | 买入 | 低 | 55 | 2025-01-05 | 下跌 | 2 |"""

# 测试3：多条记录分析统计正确性
FULL_TABLE = """| 日期 | 标的 | 决策 | 置信度 | Critic评分 | 验证日期 | 实际结果 | 准确率1-5 |
| 2025-01-01 | AAPL | 买入 | 高 | 85 | 2025-01-03 | 上涨 | 4 |
| 2025-01-02 | TSLA | 卖出 | 中 | 70 | 2025-01-04 | 下跌 | 3 |
| 2025-01-03 | NVDA | 买入 | 低 | 55 | 2025-01-05 | 下跌 | 2 |
| 2025-01-04 | MSFT | 买入 | 高 | 90 | 2025-01-06 | 上涨 | 5 |
| 2025-01-05 | GOOG | 卖出 | 中 | 75 | 2025-01-07 | 上涨 | 2 |
| 2025-01-06 | AMZN | 持有 | 低 | 50 | 2025-01-08 | 震荡 | 3 |"""


# ---------- 测试用例 ----------

def test_empty_table():
    """空表格（只有表头）→ total_count=0"""
    reader = MemoryReader(EMPTY_TABLE)
    stats = reader.analyze()
    assert stats['total_count'] == 0, f"期望 total_count=0，实际={stats['total_count']}"
    assert stats['win_count'] == 0
    assert stats['win_rate'] == 0.0
    print("✓ test_empty_table passed")


def test_partial_verified():
    """3条记录，2条已验证 → total_count=2"""
    reader = MemoryReader(PARTIAL_TABLE)
    stats = reader.analyze()
    assert stats['total_count'] == 2, f"期望 total_count=2，实际={stats['total_count']}"
    # 已验证记录：第1条(准确率4>=3胜)，第3条(准确率2<3负)
    assert stats['win_count'] == 1, f"期望 win_count=1，实际={stats['win_count']}"
    assert stats['win_rate'] == 50.0, f"期望 win_rate=50.0，实际={stats['win_rate']}"
    print("✓ test_partial_verified passed")


def test_full_analysis():
    """多条记录分析统计正确性"""
    reader = MemoryReader(FULL_TABLE)
    stats = reader.analyze()

    # 6条已验证记录
    assert stats['total_count'] == 6, f"期望 total_count=6，实际={stats['total_count']}"

    # 准确率: 4,3,2,5,2,3 → 胜(>=3): 4,3,5,3 → 4胜
    assert stats['win_count'] == 4, f"期望 win_count=4，实际={stats['win_count']}"
    assert stats['win_rate'] == round(4/6*100, 1), f"期望 win_rate≈66.7，实际={stats['win_rate']}"

    # 平均准确率: (4+3+2+5+2+3)/6 = 19/6 ≈ 3.17
    expected_avg = round(19/6, 2)
    assert stats['avg_accuracy'] == expected_avg, f"期望 avg_accuracy={expected_avg}，实际={stats['avg_accuracy']}"

    # 按决策类型验证
    decision_stats = stats['by_decision_type']
    # 买入: 3次(AAPL,NVDA,MSFT), 胜: 2次(AAPL,MSFT), 胜率66.7%
    assert '买入' in decision_stats
    assert decision_stats['买入']['total'] == 3
    assert decision_stats['买入']['wins'] == 2

    # 卖出: 2次(TSLA,GOOG), 胜: 1次(TSLA), 胜率50%
    assert '卖出' in decision_stats
    assert decision_stats['卖出']['total'] == 2
    assert decision_stats['卖出']['wins'] == 1

    # 持有: 1次(AMZN), 胜: 1次, 胜率100%
    assert '持有' in decision_stats
    assert decision_stats['持有']['total'] == 1
    assert decision_stats['持有']['wins'] == 1

    # 最强/最弱
    assert stats['strongest_type'] == '持有', f"期望 strongest=持有，实际={stats['strongest_type']}"
    assert stats['weakest_type'] == '卖出', f"期望 weakest=卖出，实际={stats['weakest_type']}"

    print("✓ test_full_analysis passed")


def test_get_summary_for_money():
    """get_summary_for_money 输出格式验证"""
    reader = MemoryReader(FULL_TABLE)
    summary = reader.get_summary_for_money()
    assert '胜率' in summary
    assert '次验证' in summary
    # FULL_TABLE 中持有最强(100%)，卖出最弱(50%)
    assert '持有' in summary
    assert '卖出' in summary
    print("✓ test_get_summary_for_money passed")


def test_get_summary_for_critic():
    """get_summary_for_critic 输出格式验证"""
    reader = MemoryReader(FULL_TABLE)
    summary = reader.get_summary_for_critic()
    assert 'Critic评分' in summary or '相关性' in summary
    print("✓ test_get_summary_for_critic passed")


def test_critic_score_correlation():
    """Critic评分分段统计正确性"""
    reader = MemoryReader(FULL_TABLE)
    stats = reader.analyze()
    cs = stats['by_critic_score']

    # >=80: AAPL(85,胜), MSFT(90,胜) → 2次全胜
    assert cs['>=80']['total'] == 2, f"期望 >=80 total=2，实际={cs['>=80']['total']}"
    assert cs['>=80']['wins'] == 2

    # 60-79: TSLA(70,胜), GOOG(75,负) → 2次1胜
    assert cs['60-79']['total'] == 2
    assert cs['60-79']['wins'] == 1

    # <60: NVDA(55,负), AMZN(50,胜) → 2次1胜
    assert cs['<60']['total'] == 2
    assert cs['<60']['wins'] == 1

    print("✓ test_critic_score_correlation passed")


if __name__ == '__main__':
    test_empty_table()
    test_partial_verified()
    test_full_analysis()
    test_get_summary_for_money()
    test_get_summary_for_critic()
    test_critic_score_correlation()
    print("\n全部测试通过！")