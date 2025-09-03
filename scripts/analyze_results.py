import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

def normalize_text(text: str) -> str:
    if text is None:
        return ''
    return str(text).strip().lower()


SEMANTIC_CATEGORIES = {
    'turn_left': ['turn left', 'left'],
    'turn_right': ['turn right', 'right'],
    'bathroom': ['bathroom', 'toilet', 'shower', 'sink'],
    'kitchen': ['kitchen', 'bar', 'counter'],
    'bedroom': ['bedroom', 'bed'],
    'living': ['living', 'couch', 'sofa'],
    'dining': ['dining', 'table', 'chairs'],
    'doorway_hall': ['doorway', 'door', 'hall', 'hallway'],
    'stairs': ['stairs'],
    'wait_stop': ['stop', 'wait'],
}


def categorize_instruction(instruction: str) -> Set[str]:
    text = normalize_text(instruction)
    matched: Set[str] = set()
    for cat, keys in SEMANTIC_CATEGORIES.items():
        for key in keys:
            if key in text:
                matched.add(cat)
                break
    if not matched:
        matched.add('uncategorized')
    return matched


def parse_results(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines gracefully
                continue
    return rows


def compute_stats(rows: List[Dict]) -> Dict:
    total = len(rows)
    success_rows = [r for r in rows if float(r.get('success', 0)) >= 0.5]
    failure_rows = [r for r in rows if float(r.get('success', 0)) < 0.5]

    def agg(subset: List[Dict]) -> Dict:
        n = len(subset)
        if n == 0:
            return {
                'count': 0,
                'avg_collisions': 0.0,
                'ratio_with_collision': 0.0,
                'avg_steps': 0.0,
                'avg_ne': 0.0,
            }
        collisions = [int(r.get('collisions', 0)) for r in subset]
        steps = [int(r.get('steps', 0)) for r in subset]
        nes = [float(r.get('ne', 0.0)) for r in subset]
        return {
            'count': n,
            'avg_collisions': sum(collisions) / n,
            'ratio_with_collision': sum(1 for c in collisions if c > 0) / n,
            'avg_steps': sum(steps) / n if n > 0 else 0.0,
            'avg_ne': sum(nes) / n if n > 0 else 0.0,
        }

    overall = agg(rows)
    success_stats = agg(success_rows)
    failure_stats = agg(failure_rows)

    # Collisions histogram
    collision_hist = Counter(int(r.get('collisions', 0)) for r in rows)

    # Per semantic category stats
    per_cat = defaultdict(lambda: {
        'count': 0,
        'success_count': 0,
        'failure_count': 0,
        'sum_collisions': 0,
        'sum_steps': 0,
        'sum_ne': 0.0,
    })
    for r in rows:
        cats = categorize_instruction(r.get('episode_instruction', ''))
        success_flag = float(r.get('success', 0)) >= 0.5
        collisions = int(r.get('collisions', 0))
        steps = int(r.get('steps', 0))
        ne = float(r.get('ne', 0.0))
        for c in cats:
            node = per_cat[c]
            node['count'] += 1
            node['success_count'] += 1 if success_flag else 0
            node['failure_count'] += 0 if success_flag else 1
            node['sum_collisions'] += collisions
            node['sum_steps'] += steps
            node['sum_ne'] += ne

    per_cat_final = {}
    for c, node in per_cat.items():
        n = node['count'] if node['count'] > 0 else 1
        per_cat_final[c] = {
            'count': node['count'],
            'success_rate': node['success_count'] / n,
            'avg_collisions': node['sum_collisions'] / n,
            'avg_steps': node['sum_steps'] / n,
            'avg_ne': node['sum_ne'] / n,
        }

    return {
        'total': total,
        'overall': overall,
        'success': success_stats,
        'failure': failure_stats,
        'collision_hist': dict(sorted(collision_hist.items())),
        'per_category': per_cat_final,
    }


def pretty_print(stats: Dict, top_k: int = 10):
    print('--- Evaluation Statistics ---')
    print(f"Total samples: {stats['total']}")
    print('\nOverall:')
    o = stats['overall']
    print(f"  Avg collisions: {o['avg_collisions']:.2f} | Ratio with collision: {o['ratio_with_collision']*100:.1f}% | Avg steps: {o['avg_steps']:.1f} | Avg NE: {o['avg_ne']:.2f}")

    print('\nSuccess cases:')
    s = stats['success']
    print(f"  Count: {s['count']} | Avg collisions: {s['avg_collisions']:.2f} | Ratio with collision: {s['ratio_with_collision']*100:.1f}% | Avg steps: {s['avg_steps']:.1f} | Avg NE: {s['avg_ne']:.2f}")

    print('\nFailure cases:')
    f = stats['failure']
    print(f"  Count: {f['count']} | Avg collisions: {f['avg_collisions']:.2f} | Ratio with collision: {f['ratio_with_collision']*100:.1f}% | Avg steps: {f['avg_steps']:.1f} | Avg NE: {f['avg_ne']:.2f}")

    print('\nCollision count distribution:')
    hist = stats['collision_hist']
    # Show first few buckets
    shown = 0
    for k, v in hist.items():
        print(f"  collisions={k}: {v}")
        shown += 1
        if shown >= 15:
            break

    print('\nSemantic categories (Top by failure rate and avg collisions):')
    cats = stats['per_category']
    # Top by (1 - success_rate) then avg_collisions
    sorted_cats = sorted(cats.items(), key=lambda kv: ((1.0 - kv[1]['success_rate']), kv[1]['avg_collisions']), reverse=True)
    for i, (c, m) in enumerate(sorted_cats[:top_k]):
        print(f"  {i+1}. {c}: samples={m['count']}, success_rate={m['success_rate']*100:.1f}%, avg_collisions={m['avg_collisions']:.2f}, avg_steps={m['avg_steps']:.1f}, avg_NE={m['avg_ne']:.2f}")


def _ensure_dir(path: str):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def plot_collision_hist(collision_hist: Dict[int, int], save_dir: str, fname: str = 'collision_hist.png'):
    if not save_dir:
        return
    _ensure_dir(save_dir)
    items = sorted(collision_hist.items())
    xs = [k for k, _ in items]
    ys = [v for _, v in items]
    if _HAS_SNS:
        sns.set(style='whitegrid')
    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys, color='#4C78A8')
    plt.xlabel('Collisions')
    plt.ylabel('Samples')
    plt.title('Collision count distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


def plot_success_failure_metrics(stats: Dict, save_dir: str, fname: str = 'success_failure_metrics.png'):
    if not save_dir:
        return
    _ensure_dir(save_dir)
    metrics = ['avg_collisions', 'ratio_with_collision', 'avg_steps', 'avg_ne']
    labels = ['Avg collisions', 'Ratio with collision(%)', 'Avg steps', 'Avg NE']
    s = stats['success']
    f = stats['failure']
    success_vals = [s['avg_collisions'], s['ratio_with_collision'] * 100.0, s['avg_steps'], s['avg_ne']]
    failure_vals = [f['avg_collisions'], f['ratio_with_collision'] * 100.0, f['avg_steps'], f['avg_ne']]
    x = list(range(len(metrics)))
    width = 0.35
    if _HAS_SNS:
        sns.set(style='whitegrid')
    plt.figure(figsize=(9, 4))
    plt.bar([i - width/2 for i in x], success_vals, width=width, label='Success', color='#59A14F')
    plt.bar([i + width/2 for i in x], failure_vals, width=width, label='Failure', color='#E45756')
    plt.xticks(x, labels)
    plt.ylabel('Value')
    plt.title('Success vs Failure metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


def plot_per_category(stats: Dict, top_k: int, save_dir: str, prefix: str = 'per_category'):
    if not save_dir:
        return
    _ensure_dir(save_dir)
    cats = stats['per_category']
    sorted_cats = sorted(cats.items(), key=lambda kv: ((1.0 - kv[1]['success_rate']), kv[1]['avg_collisions']), reverse=True)[:top_k]
    names = [c for c, _ in sorted_cats]
    success_rates = [m['success_rate'] * 100.0 for _, m in sorted_cats]
    avg_cols = [m['avg_collisions'] for _, m in sorted_cats]

    if _HAS_SNS:
        sns.set(style='whitegrid')
    plt.figure(figsize=(max(8, len(names) * 0.6), 4))
    plt.bar(names, success_rates, color='#72B7B2')
    plt.ylabel('Success rate(%)')
    plt.title('Semantic category success rate (Top-K)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_success_rate.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(max(8, len(names) * 0.6), 4))
    plt.bar(names, avg_cols, color='#F58518')
    plt.ylabel('Avg collisions')
    plt.title('Semantic category avg collisions (Top-K)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_avg_collisions.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze VLN evaluation results JSONL')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to results JSONL (one JSON per line)')
    parser.add_argument('--output', '-o', type=str, default='', help='Optional path to save aggregated stats JSON')
    parser.add_argument('--top_k', '-k', type=int, default=10, help='Top-K categories to show')
    parser.add_argument('--plot_dir', '-p', type=str, default='', help='If provided, saves charts (PNG) into this directory')
    args = parser.parse_args()

    stats = compute_stats(parse_results(args.input))
    pretty_print(stats, top_k=args.top_k)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
        with open(args.output, 'w') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nSaved stats to: {args.output}")

    # Plotting
    if args.plot_dir:
        plot_collision_hist(stats['collision_hist'], args.plot_dir)
        plot_success_failure_metrics(stats, args.plot_dir)
        plot_per_category(stats, top_k=args.top_k, save_dir=args.plot_dir)


if __name__ == '__main__':
    main()
