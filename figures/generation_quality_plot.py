"""Generation quality comparison: GenPPL vs Distinct-2 scatter plot.

Shows tradeoff between fluency (GenPPL) and diversity (D2) for different methods.
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

fig, ax = plt.subplots(figsize=(7, 5))

# Data: (GenPPL, D2, label, marker, color, size)
# Remask free (block=256)
data = [
    (9.91, 0.559, 'Original (free)', 'o', '#999999', 80),
    (5.14, 0.276, 'MDM CPT (free)', 's', '#999999', 80),
    (5.90, 0.363, 'XDLM (free)', 'D', '#C44E52', 80),
    (8.49, 0.456, r'DSL $\beta$=2 (free)', '^', '#4C72B0', 100),
    (6.59, 0.377, r'DSL sem $\beta$=0.5 (free)', 'v', '#DD8452', 100),
    # Block=32 + EOS suppress
    (7.76, 0.327, 'Original (blk32)', 'o', '#999999', 50),
    (3.96, 0.146, 'MDM CPT (blk32)', 's', '#999999', 50),
    (7.50, 0.255, r'DSL $\beta$=2 (blk32)', '^', '#4C72B0', 60),
    (4.99, 0.223, r'DSL sem $\beta$=0.5 (blk32)', 'v', '#DD8452', 60),
]

for ppl, d2, label, marker, color, size in data:
    is_block = 'blk32' in label
    alpha = 0.5 if is_block else 1.0
    edgecolor = 'black' if not is_block else 'gray'
    ax.scatter(ppl, d2, marker=marker, c=color, s=size, alpha=alpha,
              edgecolors=edgecolor, linewidths=0.8, zorder=3)

# Add labels for key points only (avoid clutter)
key_labels = {
    (9.91, 0.559): ('Original', 'left'),
    (5.14, 0.276): ('MDM CPT', 'right'),
    (5.90, 0.363): ('XDLM', 'left'),
    (8.49, 0.456): (r'$\beta$=2', 'left'),
    (6.59, 0.377): (r'sem $\beta$=0.5', 'right'),
    (4.99, 0.223): (r'sem (blk32)', 'right'),
    (3.96, 0.146): ('MDM (blk32)', 'right'),
}

for (ppl, d2), (label, ha) in key_labels.items():
    offset_x = -0.15 if ha == 'right' else 0.15
    offset_y = 0.015
    ax.annotate(label, (ppl, d2), xytext=(ppl + offset_x, d2 + offset_y),
               fontsize=8, ha=ha, va='bottom')

# Arrow showing "better" direction
ax.annotate('', xy=(3.5, 0.55), xytext=(9, 0.15),
           arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.3))
ax.text(5.5, 0.40, 'Better', fontsize=10, color='green', alpha=0.4, rotation=25, ha='center')

# Legend for marker meaning
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=8, label='Baselines'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#4C72B0', markersize=8, label=r'DSL $\beta$=2 (Route 1)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#DD8452', markersize=8, label=r'DSL sem $\beta$=0.5 (Route 2)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#C44E52', markersize=8, label='XDLM'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, alpha=0.5, label='block=32 (smaller)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

ax.set_xlabel('GenPPL (GPT-2 Large) ↓')
ax.set_ylabel('Distinct-2 ↑')
ax.set_xlim(3, 11)
ax.set_ylim(0.1, 0.6)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('paper_figures/generation_quality_scatter.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper_figures/generation_quality_scatter.png', bbox_inches='tight', dpi=300)
print("Saved: paper_figures/generation_quality_scatter.{pdf,png}")
