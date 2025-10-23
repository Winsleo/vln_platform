#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare performance metrics before and after optimization with visualizations"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_results(origin_path, optimized_path):
    """Load results before and after optimization"""
    with open(origin_path, 'r') as f:
        origin = json.load(f)
    with open(optimized_path, 'r') as f:
        optimized = json.load(f)
    return origin, optimized

def plot_overall_metrics(origin, optimized, save_path):
    """Plot overall metrics comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Overall Performance Comparison: Before vs After Optimization', fontsize=16, fontweight='bold')
    
    # Calculate success rate
    origin_sr = origin['success']['count'] / origin['total']
    optimized_sr = optimized['success']['count'] / optimized['total']
    
    # 1. Success rate comparison
    ax = axes[0, 0]
    categories = ['Before', 'After']
    values = [origin_sr * 100, optimized_sr * 100]
    bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Average collisions comparison
    ax = axes[0, 1]
    values = [origin['overall']['avg_collisions'], optimized['overall']['avg_collisions']]
    bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax.set_ylabel('Avg Collisions', fontsize=12)
    ax.set_title('Average Collisions', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Ratio with collision
    ax = axes[0, 2]
    values = [origin['overall']['ratio_with_collision'] * 100, 
              optimized['overall']['ratio_with_collision'] * 100]
    bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax.set_ylabel('Ratio with Collision (%)', fontsize=12)
    ax.set_title('Samples with Collision', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Average steps comparison
    ax = axes[1, 0]
    values = [origin['overall']['avg_steps'], optimized['overall']['avg_steps']]
    bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax.set_ylabel('Avg Steps', fontsize=12)
    ax.set_title('Average Steps', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Average navigation error comparison
    ax = axes[1, 1]
    values = [origin['overall']['avg_ne'], optimized['overall']['avg_ne']]
    bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax.set_ylabel('Avg Navigation Error (m)', fontsize=12)
    ax.set_title('Average Navigation Error', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Average collisions for success/failure samples
    ax = axes[1, 2]
    x = np.arange(2)
    width = 0.35
    origin_vals = [origin['success']['avg_collisions'], origin['failure']['avg_collisions']]
    optimized_vals = [optimized['success']['avg_collisions'], optimized['failure']['avg_collisions']]
    
    bars1 = ax.bar(x - width/2, origin_vals, width, label='Before', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized_vals, width, label='After', color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Avg Collisions', fontsize=12)
    ax.set_title('Collisions: Success vs Failure', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Success', 'Failure'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Overall comparison chart saved to: {save_path}')
    plt.close()

def plot_category_comparison(origin, optimized, save_path):
    """Plot success rate comparison by category"""
    # Get all categories
    categories = list(origin['per_category'].keys())
    if 'uncategorized' in categories:
        categories.remove('uncategorized')  # Remove categories with too few samples
    
    # Calculate success rates
    origin_sr = [origin['per_category'][cat]['success_rate'] * 100 for cat in categories]
    optimized_sr = [optimized['per_category'][cat]['success_rate'] * 100 for cat in categories]
    
    # Create chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, origin_sr, width, label='Before', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized_sr, width, label='After', 
                   color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Success Rate (%)', fontsize=13)
    ax.set_title('Success Rate by Category', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Category success rate chart saved to: {save_path}')
    plt.close()

def plot_category_collisions(origin, optimized, save_path):
    """Plot average collisions comparison by category"""
    categories = list(origin['per_category'].keys())
    if 'uncategorized' in categories:
        categories.remove('uncategorized')
    
    origin_coll = [origin['per_category'][cat]['avg_collisions'] for cat in categories]
    optimized_coll = [optimized['per_category'][cat]['avg_collisions'] for cat in categories]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, origin_coll, width, label='Before', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized_coll, width, label='After', 
                   color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Average Collisions', fontsize=13)
    ax.set_title('Average Collisions by Category', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Category collisions chart saved to: {save_path}')
    plt.close()

def print_summary(origin, optimized):
    """Print comparison summary"""
    print("\n" + "="*60)
    print("Performance Comparison Summary: Before vs After Optimization")
    print("="*60)
    
    origin_sr = origin['success']['count'] / origin['total']
    optimized_sr = optimized['success']['count'] / optimized['total']
    
    print(f"\nOverall Metrics:")
    print(f"  Success Rate:")
    print(f"    Before: {origin_sr*100:.2f}% ({origin['success']['count']}/{origin['total']})")
    print(f"    After:  {optimized_sr*100:.2f}% ({optimized['success']['count']}/{optimized['total']})")
    print(f"    Change: {(optimized_sr-origin_sr)*100:+.2f}%")
    
    print(f"\n  Average Collisions:")
    print(f"    Before: {origin['overall']['avg_collisions']:.2f}")
    print(f"    After:  {optimized['overall']['avg_collisions']:.2f}")
    print(f"    Change: {optimized['overall']['avg_collisions']-origin['overall']['avg_collisions']:+.2f}")
    
    print(f"\n  Ratio with Collision:")
    print(f"    Before: {origin['overall']['ratio_with_collision']*100:.2f}%")
    print(f"    After:  {optimized['overall']['ratio_with_collision']*100:.2f}%")
    print(f"    Change: {(optimized['overall']['ratio_with_collision']-origin['overall']['ratio_with_collision'])*100:+.2f}%")
    
    print(f"\n  Average Steps:")
    print(f"    Before: {origin['overall']['avg_steps']:.2f}")
    print(f"    After:  {optimized['overall']['avg_steps']:.2f}")
    print(f"    Change: {optimized['overall']['avg_steps']-origin['overall']['avg_steps']:+.2f}")
    
    print(f"\n  Average Navigation Error:")
    print(f"    Before: {origin['overall']['avg_ne']:.2f}m")
    print(f"    After:  {optimized['overall']['avg_ne']:.2f}m")
    print(f"    Change: {optimized['overall']['avg_ne']-origin['overall']['avg_ne']:+.2f}m")
    
    print(f"\nAverage Collisions for Successful Samples:")
    print(f"    Before: {origin['success']['avg_collisions']:.2f}")
    print(f"    After:  {optimized['success']['avg_collisions']:.2f}")
    print(f"    Change: {optimized['success']['avg_collisions']-origin['success']['avg_collisions']:+.2f}")
    
    print(f"\nAverage Collisions for Failed Samples:")
    print(f"    Before: {origin['failure']['avg_collisions']:.2f}")
    print(f"    After:  {optimized['failure']['avg_collisions']:.2f}")
    print(f"    Change: {optimized['failure']['avg_collisions']-origin['failure']['avg_collisions']:+.2f}")
    
    print("\n" + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Compare performance metrics before and after optimization with visualizations'
    )
    
    # Default base directory
    base_dir = Path(__file__).parent.parent
    
    parser.add_argument(
        '--origin',
        type=str,
        default=str(base_dir / 'results/r2r/val_unseen/analyze_origin.json'),
        help='Path to the origin analysis JSON file'
    )
    parser.add_argument(
        '--optimized',
        type=str,
        default=str(base_dir / 'results/r2r/val_unseen/analyze_optimized.json'),
        help='Path to the optimized analysis JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(base_dir / 'results/r2r/val_unseen/comparison'),
        help='Output directory for comparison charts'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    origin_path = Path(args.origin)
    optimized_path = Path(args.optimized)
    output_dir = Path(args.output)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading analysis results...")
    print(f"  Origin: {origin_path}")
    print(f"  Optimized: {optimized_path}")
    origin, optimized = load_results(origin_path, optimized_path)
    
    # Print summary
    print_summary(origin, optimized)
    
    # Generate charts
    print("Generating comparison charts...")
    plot_overall_metrics(origin, optimized, output_dir / 'overall_comparison.png')
    plot_category_comparison(origin, optimized, output_dir / 'category_success_rate.png')
    plot_category_collisions(origin, optimized, output_dir / 'category_collisions.png')
    
    print(f"\nAll charts saved to: {output_dir}")

if __name__ == '__main__':
    main()

