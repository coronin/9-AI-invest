"""
AI投研团队 · 历史战绩库读写模块
"""

import re
from typing import Optional


class MemoryReader:
    """解析历史战绩 markdown 表格并生成统计"""

    def __init__(self, content: str):
        """
        content: markdown 表格原始内容（不含表头之前的部分）
        """
        self.content = content

    def _parse_table(self) -> list[dict]:
        """解析 markdown 表格，返回每行数据 dict 列表"""
        lines = self.content.strip().split('\n')
        rows = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('|') is False:
                continue
            # 去掉首尾 |，分割列
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) < 8:
                continue
            date, symbol, decision, confidence, critic_score, verify_date, actual_result, accuracy = parts[:8]
            # 跳过表头行（第一列包含"日期"关键字）
            if '日期' in date:
                continue
            # 跳过分隔符行（如 |------|------|...）
            if re.match(r'^[\s|-]+$', line):
                continue
            rows.append({
                'date': date,
                'symbol': symbol,
                'decision': decision,
                'confidence': confidence,
                'critic_score': critic_score,
                'verify_date': verify_date,
                'actual_result': actual_result,
                'accuracy': accuracy,
            })
        return rows

    def analyze(self) -> dict:
        """
        分析历史战绩，返回统计 dict
        """
        rows = self._parse_table()

        # 只保留实际结果列有内容的记录（已验证）
        verified = [r for r in rows if r['actual_result'].strip()]

        total_count = len(verified)

        if total_count == 0:
            return {
                'total_count': 0,
                'win_count': 0,
                'win_rate': 0.0,
                'avg_accuracy': 0.0,
                'by_decision_type': {},
                'by_confidence': {},
                'by_critic_score': {},
                'recent_5': [],
                'strongest_type': '',
                'weakest_type': '',
                'critic_score_value': '无数据',
            }

        # 计算准确率数值
        def parse_accuracy(acc_str: str) -> Optional[float]:
            """从准确率字符串提取数值"""
            m = re.search(r'\d+', acc_str)
            return float(m.group()) if m else None

        # 统计胜率（准确率 >= 3）
        accuracies = []
        for r in verified:
            acc = parse_accuracy(r['accuracy'])
            if acc is not None:
                accuracies.append(acc)

        win_count = sum(1 for acc in accuracies if acc >= 3)
        win_rate = (win_count / total_count * 100) if total_count > 0 else 0.0
        avg_accuracy = (sum(accuracies) / len(accuracies)) if accuracies else 0.0

        # 按决策类型统计
        by_decision_type = {}
        decision_stats = {}
        for r in verified:
            decision = r['decision']
            acc = parse_accuracy(r['accuracy'])
            if decision not in decision_stats:
                decision_stats[decision] = {'total': 0, 'wins': 0}
            decision_stats[decision]['total'] += 1
            if acc is not None and acc >= 3:
                decision_stats[decision]['wins'] += 1

        for d, s in decision_stats.items():
            rate = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0.0
            by_decision_type[d] = {
                'total': s['total'],
                'wins': s['wins'],
                'win_rate': rate,
            }

        # 按置信度统计
        by_confidence = {}
        confidence_stats = {}
        for r in verified:
            conf = r['confidence']
            acc = parse_accuracy(r['accuracy'])
            if conf not in confidence_stats:
                confidence_stats[conf] = {'total': 0, 'wins': 0}
            confidence_stats[conf]['total'] += 1
            if acc is not None and acc >= 3:
                confidence_stats[conf]['wins'] += 1

        for c, s in confidence_stats.items():
            rate = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0.0
            by_confidence[c] = {
                'total': s['total'],
                'wins': s['wins'],
                'win_rate': rate,
            }

        # 按 Critic 评分段统计
        by_critic_score = {}
        critic_ranges = {
            '>=80': {'min': 80, 'max': 999, 'total': 0, 'wins': 0},
            '60-79': {'min': 60, 'max': 79, 'total': 0, 'wins': 0},
            '<60': {'min': 0, 'max': 59, 'total': 0, 'wins': 0},
        }
        for r in verified:
            score_str = r['critic_score']
            m = re.search(r'\d+', score_str)
            if not m:
                continue
            score = float(m.group())
            acc = parse_accuracy(r['accuracy'])
            for label, rng in critic_ranges.items():
                if rng['min'] <= score <= rng['max']:
                    rng['total'] += 1
                    if acc is not None and acc >= 3:
                        rng['wins'] += 1
                    break

        for label, rng in critic_ranges.items():
            rate = rng['wins'] / rng['total'] * 100 if rng['total'] > 0 else 0.0
            by_critic_score[label] = {
                'total': rng['total'],
                'wins': rng['wins'],
                'win_rate': rate,
            }

        # 最近5条记录
        recent_5 = verified[-5:] if len(verified) > 5 else verified

        # 最强/最弱决策类型
        strongest_type = ''
        weakest_type = ''
        if by_decision_type:
            strongest_type = max(by_decision_type.keys(),
                                 key=lambda k: by_decision_type[k]['win_rate'])
            weakest_type = min(by_decision_type.keys(),
                               key=lambda k: by_decision_type[k]['win_rate'])

        # Critic 评分参考价值
        high_score = by_critic_score.get('>=80', {})
        low_score = by_critic_score.get('<60', {})
        if high_score.get('total', 0) >= 3 and low_score.get('total', 0) >= 1:
            high_rate = high_score.get('win_rate', 0)
            low_rate = low_score.get('win_rate', 0)
            if high_rate - low_rate >= 30:
                critic_score_value = '强'
            elif high_rate - low_rate >= 10:
                critic_score_value = '中'
            else:
                critic_score_value = '弱'
        else:
            critic_score_value = '数据不足'

        return {
            'total_count': total_count,
            'win_count': win_count,
            'win_rate': round(win_rate, 1),
            'avg_accuracy': round(avg_accuracy, 2),
            'by_decision_type': by_decision_type,
            'by_confidence': by_confidence,
            'by_critic_score': by_critic_score,
            'recent_5': recent_5,
            'strongest_type': strongest_type,
            'weakest_type': weakest_type,
            'critic_score_value': critic_score_value,
        }

    def get_summary_for_money(self) -> str:
        """返回给Money哥的一句话说统计"""
        stats = self.analyze()
        if stats['total_count'] == 0:
            return '暂无已验证记录。'

        strongest = stats['strongest_type']
        weakest = stats['weakest_type']
        strongest_rate = 0.0
        weakest_rate = 0.0

        if strongest and strongest in stats['by_decision_type']:
            strongest_rate = stats['by_decision_type'][strongest]['win_rate']
        if weakest and weakest in stats['by_decision_type']:
            weakest_rate = stats['by_decision_type'][weakest]['win_rate']

        return (
            f'本团队整体胜率{stats["win_rate"]}%（{stats["total_count"]}次验证），'
            f'在{strongest}决策上表现最强（胜率{strongest_rate:.0f}%），'
            f'{weakest}决策最弱（胜率{weakest_rate:.0f}%）。'
        )

    def get_summary_for_critic(self) -> str:
        """返回给Critic的统计数据字符串"""
        stats = self.analyze()
        if stats['total_count'] == 0:
            return '历史暂无数据，无法评估Critic评分参考价值。'

        cs = stats['by_critic_score']
        high = cs.get('>=80', {})
        mid = cs.get('60-79', {})
        low = cs.get('<60', {})

        parts = []
        parts.append('历史数据显示Critic评分与最终准确率相关性：')
        if high.get('total', 0) > 0:
            parts.append(f'评分>=80分决策胜率{high["win_rate"]:.0f}%({high["total"]}次)')
        if mid.get('total', 0) > 0:
            parts.append(f'评分60-79分决策胜率{mid["win_rate"]:.0f}%({mid["total"]}次)')
        if low.get('total', 0) > 0:
            parts.append(f'评分<60分决策胜率{low["win_rate"]:.0f}%({low["total"]}次)')

        return '，'.join(parts) + '。'