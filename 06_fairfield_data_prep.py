"""
Example 6 (part 1): DATA PREP for a Sandy-scale storm in Fairfield County
==========================================================================
Target problem: 100 crews fixing 5,000 outages across Fairfield County,
CT (Greenwich to Shelton, Bridgeport to Danbury). Before any solver runs,
the data layer has to exist — and at this scale the data layer IS the
hard part:

  - A naive 5,000 x 5,000 travel matrix is 25M shortest-path queries and
    ~100 MB. We deliberately DON'T build it. The instance is saved with a
    per-substation REGION label so the solver stage (part 2) can build
    ~20 small matrices instead — the decomposition from SCALING.md.

What this script produces:
  1. Fairfield County drivable road network from OSM (~10x Hartford),
     cached to cache/fairfield_drive.graphml.
  2. A mock distribution grid: 20 substations placed by clustering the
     road network (k-means on intersections, snapped to roads), radial
     feeders grown as a shortest-path forest, backbone/lateral spans by
     downstream customer count.
  3. A typed outage population (5,000 damaged spans):

        type          share*   mean repair   what it is
        fuse          ~32%     0.75 h        recloser/fuse, quick reset
        tree_on_wire  ~28%     1.5 h         cut & clear, line intact
        wire_down     ~17%     3.0 h         re-string conductor
        transformer   ~14%     3.5 h         swap pot on pole
        pole_broken   ~9%      8.0 h         new pole: dig, set, transfer
        (*backbone damage skews toward wire/pole; laterals toward fuse)

     True repair times are lognormal around the mean (the planner will
     only see the mean — that's part 2's uncertainty).
  4. Electrical analysis: customers credited to each outage (nearest
     upstream damage) and precedence chains (upstream damage blocks
     downstream restoration).
  5. A solver-ready pickle (cache/fairfield_instance.pkl) + county maps.

Run time: first run downloads ~county OSM extract (a few minutes), then
everything is cached.
"""

import os
import pickle
import random
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from scipy.cluster.vq import kmeans2

SEED = 2012  # the year of Sandy
PLACE = "Fairfield County, Connecticut, USA"
GRAPH_CACHE = "cache/fairfield_drive.graphml"
INSTANCE_FILE = "cache/fairfield_instance.pkl"

NUM_SUBSTATIONS = 20
NUM_OUTAGES = 5000
NUM_CREWS = 100           # stored in the instance for part 2
BACKBONE_THRESHOLD = 500  # downstream customers => feeder trunk

# type: (mean repair hours, P(type | backbone), P(type | lateral))
OUTAGE_TYPES = {
    "fuse":         (0.75, 0.08, 0.35),
    "tree_on_wire": (1.5,  0.22, 0.29),
    "wire_down":    (3.0,  0.38, 0.14),
    "transformer":  (3.5,  0.07, 0.15),
    "pole_broken":  (8.0,  0.25, 0.07),
}
REPAIR_SIGMA = 0.5  # lognormal spread of true vs. mean repair time


def load_graph():
    if os.path.exists(GRAPH_CACHE):
        print(f"Loading cached graph: {GRAPH_CACHE}")
        G = ox.load_graphml(GRAPH_CACHE)
    else:
        print(f"Downloading road network for {PLACE} (a few minutes) ...")
        G = ox.graph_from_place(PLACE, network_type="drive")
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
        ox.save_graphml(G, GRAPH_CACHE)
        print(f"Saved -> {GRAPH_CACHE}")
    largest = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    print(f"Road graph: {len(G.nodes):,} intersections, "
          f"{len(G.edges):,} segments")
    return G


def place_substations(G, rng):
    """20 substations 'corresponding to roads': k-means over all
    intersections (so substation density follows road/population
    density), each centroid snapped to the nearest road node."""
    nodes = list(G.nodes)
    coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])
    centroids, _ = kmeans2(coords, NUM_SUBSTATIONS, minit="++",
                           seed=rng.randrange(2**31))
    subs = []
    for cx, cy in centroids:
        s = ox.distance.nearest_nodes(G, cx, cy)
        if s not in subs:
            subs.append(s)
    print(f"Placed {len(subs)} substations (k-means on intersections, "
          f"snapped to roads)")
    return subs


def build_grid(G, substations, rng):
    Gu = nx.Graph(G.to_undirected())
    print("Growing radial feeders (shortest-path forest) ...")
    dist, paths = nx.multi_source_dijkstra(Gu, set(substations),
                                           weight="length")
    parent, root = {}, {}
    for node, path in paths.items():
        root[node] = path[0]
        parent[node] = path[-2] if len(path) >= 2 else None

    customers = {
        n: rng.choices([0, 4, 8, 15, 40], weights=[25, 30, 25, 15, 5])[0]
        for n in parent
    }
    for s in substations:
        customers[s] = 0

    downstream = dict(customers)
    for node in sorted(parent, key=lambda n: -dist[n]):
        if parent[node] is not None:
            downstream[parent[node]] += downstream[node]

    edge_class = {
        n: ("backbone" if downstream[n] >= BACKBONE_THRESHOLD else "lateral")
        for n in parent if parent[n] is not None
    }
    n_back = sum(1 for c in edge_class.values() if c == "backbone")
    print(f"Grid: {sum(customers.values()):,} customers, "
          f"{n_back:,} backbone spans, {len(edge_class) - n_back:,} "
          f"lateral spans")
    return Gu, parent, root, customers, downstream, edge_class


def generate_storm(parent, root, edge_class, customers, rng):
    backbone = sorted(n for n, c in edge_class.items() if c == "backbone")
    laterals = sorted(n for n, c in edge_class.items() if c == "lateral")
    n_back = round(0.12 * NUM_OUTAGES)
    damaged = (rng.sample(backbone, min(n_back, len(backbone)))
               + rng.sample(laterals, NUM_OUTAGES - n_back))
    damaged_set = set(damaged)

    # Typed outages: backbone damage skews heavy (wire/pole)
    types, names = {}, list(OUTAGE_TYPES)
    p_back = [OUTAGE_TYPES[t][1] for t in names]
    p_lat = [OUTAGE_TYPES[t][2] for t in names]
    for d in damaged:
        p = p_back if edge_class[d] == "backbone" else p_lat
        types[d] = rng.choices(names, weights=p)[0]

    est, true = {}, {}
    for d in damaged:
        mean_h = OUTAGE_TYPES[types[d]][0]
        est[d] = mean_h * 3600
        true[d] = est[d] * rng.lognormvariate(-REPAIR_SIGMA**2 / 2,
                                              REPAIR_SIGMA)

    # Customers credited to each outage = those whose nearest upstream
    # damage it is; precedence chain = damaged ancestors, nearest first.
    print("Computing customer attribution + precedence chains ...")
    weight = defaultdict(int)
    nearest_up = {}
    for node, ncust in customers.items():
        cur = node
        while cur is not None and cur not in damaged_set:
            cur = parent[cur]
        nearest_up[node] = cur
        if cur is not None and ncust:
            weight[cur] += ncust

    chains = {}
    for d in damaged:
        chain, cur = [], parent[d]
        while cur is not None:
            if cur in damaged_set:
                chain.append(cur)
            cur = parent[cur]
        chains[d] = chain

    return damaged, types, est, true, dict(weight), chains


def summarize(damaged, types, est, true, weight, chains, root, edge_class):
    print("\n--- Storm summary ---")
    print(f"{'type':14s} {'count':>6s} {'share':>6s} {'mean est':>9s} "
          f"{'mean true':>9s} {'customers':>10s}")
    by_type = defaultdict(list)
    for d in damaged:
        by_type[types[d]].append(d)
    for t in OUTAGE_TYPES:
        ds = by_type[t]
        cust = sum(weight.get(d, 0) for d in ds)
        print(f"{t:14s} {len(ds):6,d} {len(ds)/len(damaged):6.0%} "
              f"{np.mean([est[d] for d in ds])/3600:8.1f}h "
              f"{np.mean([true[d] for d in ds])/3600:8.1f}h {cust:10,d}")

    total_cust = sum(weight.values())
    blocked = sum(1 for d in damaged if chains[d])
    work_h = sum(true.values()) / 3600
    print(f"\nCustomers without power : {total_cust:,}")
    print(f"Blocked by upstream dmg : {blocked:,} of {len(damaged):,} "
          f"({blocked/len(damaged):.0%})")
    print(f"Total repair workload   : {work_h:,.0f} crew-hours "
          f"(~{work_h/NUM_CREWS:.0f} h per crew before travel)")

    region_out = Counter()
    for d in damaged:
        region_out[root[d]] += weight.get(d, 0)
    region_cnt = Counter(root[d] for d in damaged)
    print(f"\nPer-region decomposition preview (top 5 of "
          f"{len(region_cnt)} substation regions):")
    for s, cust in region_out.most_common(5):
        print(f"  substation {s}: {region_cnt[s]:4d} outages, "
              f"{cust:6,d} customers out")


def render(G, Gu, substations, parent, root, edge_class, damaged, weight):
    from matplotlib.collections import LineCollection

    print("Rendering county maps (large graph, ~a minute) ...")
    sub_colors = plt.cm.tab20(np.linspace(0, 1, len(substations)))
    sub_idx = {s: k for k, s in enumerate(substations)}

    for stage, fname in [("raw", "output/06a_fairfield_grid.png"),
                         ("storm", "output/06b_fairfield_outages.png")]:
        fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0,
                                edge_color="#f0f0f0", edge_linewidth=0.2,
                                bgcolor="white", figsize=(14, 12))
        lat_segs, back_segs, back_cols = [], [], []
        for n, p in parent.items():
            if p is None:
                continue
            seg = [(Gu.nodes[p]["x"], Gu.nodes[p]["y"]),
                   (Gu.nodes[n]["x"], Gu.nodes[n]["y"])]
            if edge_class[n] == "backbone":
                back_segs.append(seg)
                back_cols.append(sub_colors[sub_idx[root[n]]])
            else:
                lat_segs.append(seg)
        ax.add_collection(LineCollection(lat_segs, colors="#bbbbbb",
                                         linewidths=0.3, alpha=0.5, zorder=2))
        ax.add_collection(LineCollection(back_segs, colors=back_cols,
                                         linewidths=1.8, alpha=0.95, zorder=3))
        for k, s in enumerate(substations):
            ax.scatter(Gu.nodes[s]["x"], Gu.nodes[s]["y"], marker="*",
                       s=350, color=sub_colors[k], edgecolors="black",
                       linewidths=0.8, zorder=6)
        if stage == "storm":
            xs = [Gu.nodes[d]["x"] for d in damaged]
            ys = [Gu.nodes[d]["y"] for d in damaged]
            sz = [4 + 0.15 * weight.get(d, 0) for d in damaged]
            ax.scatter(xs, ys, marker="x", s=sz, c="black", linewidths=0.6,
                       alpha=0.75, zorder=7)
            title = (f"Sandy-scale storm — {NUM_OUTAGES:,} outages, "
                     f"{sum(weight.values()):,} customers without power")
        else:
            title = (f"Mock grid for Fairfield County — "
                     f"{len(substations)} substations (stars), backbone "
                     f"feeders colored by substation, laterals gray")
        ax.set_title(title)
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {fname}")


def main():
    rng = random.Random(SEED)
    G = load_graph()
    substations = place_substations(G, rng)
    Gu, parent, root, customers, downstream, edge_class = build_grid(
        G, substations, rng)
    damaged, types, est, true, weight, chains = generate_storm(
        parent, root, edge_class, customers, rng)

    summarize(damaged, types, est, true, weight, chains, root, edge_class)

    instance = {
        "place": PLACE,
        "graph_cache": GRAPH_CACHE,
        "num_crews": NUM_CREWS,
        "substations": substations,
        "parent": parent,
        "root": root,                  # outage -> substation = its region
        "customers": customers,
        "edge_class": edge_class,
        "damaged": damaged,
        "types": types,
        "est_repair_s": est,
        "true_repair_s": true,         # hidden from the planner in part 2
        "weight": weight,
        "chains": chains,
    }
    with open(INSTANCE_FILE, "wb") as f:
        pickle.dump(instance, f)
    print(f"\nSolver-ready instance saved -> {INSTANCE_FILE} "
          f"({os.path.getsize(INSTANCE_FILE)/1e6:.1f} MB)")
    print("NOTE: no travel matrix was built — at 5,000 outages that's "
          "25M pairs. Part 2 builds ~20 small per-region matrices instead.")

    render(G, Gu, substations, parent, root, edge_class, damaged, weight)


if __name__ == "__main__":
    main()
