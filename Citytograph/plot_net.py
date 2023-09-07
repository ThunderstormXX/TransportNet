import argparse
import os
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import collections as mc
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize



n_lines = len(net.inits)
xa, xb, ya, yb = [np.zeros(n_lines) for _ in range(4)]
for i in range(n_lines):
    an, bn = net.node_index_to_id[net.inits[i]], net.node_index_to_id[net.terms[i]]
    dfa = dfb = nodes

    if an not in dfa.index:
        print("KEKa", an, net.types[i])
        continue
    if bn not in dfb.index:
        print("KEKb", bn, net.types[i])
        continue

    xa[i] = dfa.X[an]
    xb[i] = dfb.X[bn]
    ya[i] = dfa.Y[an]
    yb[i] = dfb.Y[bn]


lines = []
flows = []
loads = []
linestyles = []

for i in range(n_lines):
    flow = route_result["flows"][i]
    cap = net.capacities[i]

    if args.plot_loads:
        if net.types[i] != model_params.route_assignment_params.net_structural.edge_types.STREET:
            continue
        loads.append(flow / cap)

    flows.append(flow)
    if net.types[i] == 0:
        linestyles.append("-")
    else:
        linestyles.append(":")

    lines.append([(xa[i], ya[i]), (xb[i], yb[i])])

if args.plot_loads:
    values = np.array(loads)
    title = "flow / capacity, " + exec_name
else:
    values = np.array(flows)
    title = "flow, " + exec_name

if args.plot_loads:
    values = np.array(loads)
    title = "flow / capacity, " + exec_name
else:
    values = np.array(flows)
    title = "flow, " + exec_name

title += f", iter={args.iteration if args.iteration >= 0 else iterations_saved + args.iteration}"

cmap = plt.get_cmap("magma_r")
print(min(values), max(values))
norm = Normalize(min(values), max(values))
colors = [cmap(min(values))] * len(values)  # [cmap(norm(x)) for x in values]

# for i in range(n_lines):
#     if linestyles[i] != "-":
#         colors[i] = (0, 1, 0, 1)
# _, _, _, paths, idx = get_PS_info_from_file("output.txt")  # get_PS_from_file("output.txt", num_lines=200)
with open(f"./output.pickle", "rb") as fp:
    path_sets = pickle.load(fp)
for k, (path, flow) in enumerate(zip(path_sets[105][84].paths, path_sets[105][84].flows)):
    for edge in path:
        colors[edge] = cmap(norm(flow * 100))  # (1 / len(path_sets[0][0].paths) * k, 0, 0, 1)

lc = mc.LineCollection(lines, colors=colors, linewidths=2, linestyles=linestyles)
fig, ax = plt.subplots(figsize=(100, 100))
ax.add_collection(lc)
plt.colorbar(cm.ScalarMappable(norm, cmap))
# ax.autoscale()
ax.margins(0.1)

plt.scatter(districts.X, districts.Y, s=4)

plt.axis("equal")
plt.title(title)

from mpl_interactions import ioff, panhandler, zoom_factory

disconnect_zoom = zoom_factory(ax)
pan_handler = panhandler(fig)
if "sadovoe" in in_path:
    plt.ylim((districts.Y.min() - 100, districts.Y.max() + 100))
    plt.xlim((districts.X.min() - 100, districts.X.max() + 100))
# plt.savefig("path_sets/3.png", dpi=150, bbox_inches="tight")
plt.show()