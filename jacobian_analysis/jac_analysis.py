import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
# import pathlib
from pathlib import Path
import plotly.express as px


file_path = open(Path("explicit_jac.pckl"), "rb")
jac_explicit = pickle.load(file_path)
file_path.close()

file_path = open(Path("root_explicit_jac.pckl"), "rb")
jac_root_explicit = pickle.load(file_path)
file_path.close()

# get the colors from plotly
c0 = px.colors.qualitative.D3[0]
c1 = px.colors.qualitative.D3[1]
alpha = 0.75
c0a = colorConverter.to_rgba(c0, alpha=.75)
markersize = 5.25
# create a figure and axes
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
# plot the jacobian "jac" in the dictionary "jac_explicit" with spy plot with opacity 0.5

ax[0].spy(jac_explicit["jac"], alpha=alpha, markersize=markersize, marker="s", color=c0, markeredgecolor=c0a)
ax[0].spy(jac_root_explicit["jac"], alpha=1, markersize=4, marker=".", color=c1)
ax[0].set_xlabel("States (first and second intervals)")
ax[0].set_ylabel("Continuity constraints")
ax[0].set_xlim(-0.5, 59.5)
ax[0].set_ylim(59.5, -0.5)
# set x ticks for a specific range
ax[0].set_xticks(range(0, 60, 15))
ax[0].set_yticks(range(0, 60, 15))

# plot the jacobian "jac" in the dictionary "jac_explicit" with spy plot with opacity 0.5

ax[1].spy(jac_explicit["jac"], alpha=alpha, markersize=markersize, marker="s", color=c0, markeredgecolor=c0a)
ax[1].spy(jac_root_explicit["jac"], alpha=1, markersize=4, marker=".", color=c1)
ax[1].set_xlabel("Controls (first and second intervals)")
ax[1].set_xlim(4559.5, 4577.5)
ax[1].set_ylim(59.5, -0.5)
# set x ticks for a specific range
ax[1].set_xticks(range(4560, 4577, 9))
ax[1].set_yticks(range(0, 60, 15))

# plot legend outside of the axe 1 for root_explicit and explicit jacobian
leg = ax[1].legend([r'$Full-Exp$', r'$Base-Exp$'], loc="center left", bbox_to_anchor=(1.4, 0.5))
# the legend has no box around it
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_facecolor('none')

#remove box of both axes
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
# box off
ax[0].set_frame_on(False)
ax[1].set_frame_on(False)


fig.tight_layout()
#export figure to file in eps format with dpi=600
fig.savefig("jac_analysis.eps", format="eps", dpi=600)
#export figure to file in svg format
fig.savefig("jac_analysis.svg", format="svg", dpi=600)
#export in pdf format
fig.savefig("jac_analysis.pdf", format="pdf", dpi=600)
# show the plot
plt.show()