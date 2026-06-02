"""Random vs Semantic corruption fix rate comparison — grouped bar chart.

Shows the design space: Route 1, ROAR, Route 2, and baselines.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

# Data from eval_results
models = ['Original', 'MDM CPT', 'XDLM', r'Route 1'+'\n'+r'($\beta$=2)', 'ROAR'+'\n'+r'($\beta$=2)', r'Route 2'+'\n'+r'(sem $\beta$=0.5)']
random_fix = [13.1, 17.2, 70.8, 64.8, 64.4, 42.0]
semantic_fix = [31.9, 34.8, 66.5, 19.5, 77.4, 66.7]
clean_preserved = [90.9, 92.0, 82.0, 99.0, 98.5, 98.8]

x = np.arange(len(models))
width = 0.32

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [3, 2]})

colors_rand = '#4C72B0'
colors_sem = '#DD8452'

bars1 = ax1.bar(x - width/2, random_fix, width, label='Random corruption', color=colors_rand, edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x + width/2, semantic_fix, width, label='Semantic corruption', color=colors_sem, edgecolor='white', linewidth=0.5)

ax1.set_ylabel('Fix Rate (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8.5)
ax1.legend(loc='upper left', fontsize=9, frameon=False)
ax1.set_ylim(0, 90)

# Highlight ROAR as the best combined
for bar, val in zip(bars1, random_fix):
    if val > 50:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7.5)
for bar, val in zip(bars2, semantic_fix):
    if val > 60:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Star on ROAR
ax1.annotate('best combined',
            xy=(4, 80), fontsize=8, ha='center', color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFCC', edgecolor='#999999', linewidth=0.5))

# Right panel: Clean preserved
colors_clean = ['#999999', '#999999', '#C44E52', '#55A868', '#55A868', '#55A868']
bars3 = ax2.barh(x, clean_preserved, height=0.5, color=colors_clean, edgecolor='white', linewidth=0.5)
ax2.set_xlabel('Clean Preserved (%)')
ax2.set_yticks(x)
ax2.set_yticklabels(models, fontsize=8.5)
ax2.set_xlim(75, 101)
ax2.axvline(x=95, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

for bar, val in zip(bars3, clean_preserved):
    ax2.text(val - 0.5, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}%', ha='right', va='center', fontsize=8,
            color='white' if val > 90 else 'black', fontweight='bold')

ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('paper_figures/corruption_random_vs_semantic.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper_figures/corruption_random_vs_semantic.png', bbox_inches='tight', dpi=300)
print("Saved: paper_figures/corruption_random_vs_semantic.{pdf,png}")
