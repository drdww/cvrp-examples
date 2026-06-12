"""
Example 4: Storm restoration on a mock power grid built from Hartford's roads
==============================================================================
Roads vs. power lines, in one sentence: trucks can drive any path, but
power flows ONE way â€” substation -> feeder (backbone) -> lateral ->
customer. That hierarchy changes the optimization problem completely.

How we mock the grid (and why it's defensible):
  Overhead distribution lines are strung on poles ALONG roads, so the
  road graph is a realistic scaffold. We place 4 substations, then grow
  a shortest-path forest over the street network â€” every intersection
  ("pole") is fed by its nearest substation along streets. That forest
  IS a radial distribution system:
    - BACKBONE (feeder trunk): tree edges whose downstream subtree
      serves many customers (three-phase trunk along major roads)
    - LATERAL: tree edges serving few customers (single-phase taps
      into neighborhoods)

What changes vs. the CVRP in Example 3:
  1. OBJECTIVE: not total distance. Utilities minimize CUSTOMER-MINUTES
     of interruption (CMI, the integral under the outage curve â€” SAIDI
     is CMI / customers served). Implemented with OR-Tools' soft upper
     bound trick: penalize each damage node's arrival time multiplied
     by the customers it restores => a weighted minimum-latency VRP.
  2. PRECEDENCE: repairing a lateral restores nobody while its upstream
     feeder is still broken. We add cross-crew precedence constraints
     on the shared Time dimension (upstream damage fixed before work
     starts downstream). Simplification: real crews may repair in
     parallel and energize later; modeling that exactly is a
     post-processing step (we do compute true energization times).
  3. WEIGHTS: each damage is credited with the customers whose NEAREST
     upstream damage it is â€” fix the feeder break and 2,000 come back;
     fix a lateral and 40 come back.

Outputs:
  output/04_grid_map.html          interactive map: substations, backbone,
                                   laterals, damage sites colored by crew
  output/04_outage_curve.png  customers restored vs. time (the curve
                                   utilities publish during storms)

Scaling note (the real goal â€” 2,000 crews / 25,000 outages across CT):
  this exact model decomposes naturally by substation/feeder: assign
  crews to regions, solve each region as below, re-optimize on a rolling
  horizon as damage assessment trickles in. See README.
"""

import os
import random
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

SEED = 11
GRAPH_CACHE = "cache/hartford_drive.graphml"
PLACE = "Hartford, Connecticut, USA"

NUM_SUBSTATIONS = 4
NUM_DAMAGES = 40
NUM_CREWS = 8
DEPOT_LATLON = (41.7896, -72.6747)   # mock utility service center
BACKBONE_THRESHOLD = 250             # downstream customers => feeder trunk
REPAIR_BACKBONE_S = 90 * 60          # 90 min: broken pole / 3-phase span
REPAIR_LATERAL_S = 45 * 60           # 45 min: fuse / single-phase span
SHIFT_HORIZON_S = 24 * 3600
TIME_LIMIT_S = 30


# ----------------------------------------------------------------------
# Step 1: road network (reuses Example 3's cache)
# ----------------------------------------------------------------------
def load_graph():
    if os.path.exists(GRAPH_CACHE):
        G = ox.load_graphml(GRAPH_CACHE)
    else:
        G = ox.graph_from_place(PLACE, network_type="drive")
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
        ox.save_graphml(G, GRAPH_CACHE)
    largest = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    print(f"Road graph: {len(G.nodes):,} poles/intersections, "
          f"{len(G.edges):,} segments")
    return G


# ----------------------------------------------------------------------
# Step 2: build the mock distribution grid
# ----------------------------------------------------------------------
def build_grid(G, rng):
    Gu = nx.Graph(G.to_undirected())  # power doesn't care about one-ways

    # Substations: the node nearest each quadrant center of the city,
    # so the four feeders cover distinct territories.
    xs = [d["x"] for _, d in Gu.nodes(data=True)]
    ys = [d["y"] for _, d in Gu.nodes(data=True)]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    quadrant_centers = [
        (x0 + 0.3 * (x1 - x0), y0 + 0.3 * (y1 - y0)),
        (x0 + 0.7 * (x1 - x0), y0 + 0.3 * (y1 - y0)),
        (x0 + 0.3 * (x1 - x0), y0 + 0.7 * (y1 - y0)),
        (x0 + 0.7 * (x1 - x0), y0 + 0.7 * (y1 - y0)),
    ][:NUM_SUBSTATIONS]
    substations = [
        ox.distance.nearest_nodes(G, cx, cy) for cx, cy in quadrant_centers
    ]

    # Shortest-path forest from the substations: each pole is fed by its
    # nearest substation along the streets. This is our radial grid.
    _, paths = nx.multi_source_dijkstra(Gu, set(substations), weight="length")
    parent, root = {}, {}
    for node, path in paths.items():
        root[node] = path[0]
        parent[node] = path[-2] if len(path) >= 2 else None

    # Customers per pole: most poles serve a handful of homes; a few
    # (apartment blocks, commercial) serve many.
    customers = {
        n: rng.choices([0, 4, 8, 15, 40], weights=[25, 30, 25, 15, 5])[0]
        for n in parent
    }
    for s in substations:
        customers[s] = 0

    # Downstream customer count for every node = its own + all
    # descendants'. Process nodes farthest-first so children are summed
    # before their parents.
    dist, _ = nx.multi_source_dijkstra(Gu, set(substations), weight="length")
    downstream = dict(customers)
    for node in sorted(parent, key=lambda n: -dist[n]):
        if parent[node] is not None:
            downstream[parent[node]] += downstream[node]

    # The tree edge ABOVE node n (parent[n] -> n) carries downstream[n]
    # customers: that's what classifies it as backbone or lateral.
    edge_class = {
        n: ("backbone" if downstream[n] >= BACKBONE_THRESHOLD else "lateral")
        for n in parent if parent[n] is not None
    }

    total = sum(customers.values())
    n_backbone = sum(1 for c in edge_class.values() if c == "backbone")
    print(f"Mock grid: {NUM_SUBSTATIONS} substations, "
          f"{total:,} customers, {n_backbone} backbone spans, "
          f"{len(edge_class) - n_backbone} lateral spans")
    return Gu, substations, parent, root, customers, downstream, edge_class


# ----------------------------------------------------------------------
# Step 3: storm damage + who-restores-whom analysis
# ----------------------------------------------------------------------
def generate_damage(parent, edge_class, customers, rng):
    # Each damaged "span" is the tree edge above a node; we identify the
    # damage by that node (crews drive to that intersection). Storms hit
    # feeder trunks disproportionately (long spans on tree-lined major
    # roads), so force a realistic mix: ~15% backbone, rest laterals.
    backbone = sorted(n for n, c in edge_class.items() if c == "backbone")
    laterals = sorted(n for n, c in edge_class.items() if c == "lateral")
    n_backbone = max(1, round(0.15 * NUM_DAMAGES))
    damaged = (rng.sample(backbone, min(n_backbone, len(backbone)))
               + rng.sample(laterals, NUM_DAMAGES - n_backbone))
    damaged_set = set(damaged)

    def upstream_chain(n):
        """Damaged ancestors of damage n, nearest-first (its span's
        parent on up to the substation)."""
        chain, cur = [], parent[n]
        while cur is not None:
            if cur in damaged_set:
                chain.append(cur)
            cur = parent[cur]
        return chain

    # Credit every customer to the NEAREST damage above them. Fixing
    # that span (after its own upstream is fixed) brings them back.
    weight = defaultdict(int)
    for node, ncust in customers.items():
        if ncust == 0:
            continue
        cur = node
        while cur is not None:
            if cur in damaged_set:
                weight[cur] += ncust
                break
            cur = parent[cur]

    precedence = {d: upstream_chain(d) for d in damaged}
    n_blocked = sum(1 for c in precedence.values() if c)
    print(f"Storm: {NUM_DAMAGES} damaged spans, "
          f"{sum(weight.values()):,} customers out, "
          f"{n_blocked} damages blocked by upstream damage")
    return damaged, weight, precedence


# ----------------------------------------------------------------------
# Step 4: crew routing â€” weighted-latency VRP with precedence
# ----------------------------------------------------------------------
def travel_matrix(G, road_nodes):
    n = len(road_nodes)
    pos = {node: i for i, node in enumerate(road_nodes)}
    M = [[0] * n for _ in range(n)]
    for i, src in enumerate(road_nodes):
        times = nx.single_source_dijkstra_path_length(G, src, weight="travel_time")
        for node, t in times.items():
            j = pos.get(node)
            if j is not None:
                M[i][j] = int(round(t))
    return M


def solve_restoration(G, damaged, weight, precedence, edge_class):
    depot_node = ox.distance.nearest_nodes(G, DEPOT_LATLON[1], DEPOT_LATLON[0])
    road_nodes = [depot_node] + damaged          # model node i <-> damaged[i-1]
    M = travel_matrix(G, road_nodes)
    repair = [0] + [
        REPAIR_BACKBONE_S if edge_class[d] == "backbone" else REPAIR_LATERAL_S
        for d in damaged
    ]

    manager = pywrapcp.RoutingIndexManager(len(road_nodes), NUM_CREWS, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Transit = drive time + repair time at the node you're leaving, so
    # the Time dimension's cumul at a node = the moment work STARTS there.
    def transit_cb(fi, ti):
        i, j = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return M[i][j] + repair[i]

    transit = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    routing.AddDimension(transit, 0, SHIFT_HORIZON_S, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # THE OBJECTIVE: soft upper bound of 0 with coefficient = customers
    # adds (customers x start-time-of-repair) to the cost for every
    # damage => minimize total customer-waiting (approx. CMI). Travel
    # arc costs remain as a small tie-breaker.
    for i, d in enumerate(damaged, start=1):
        idx = manager.NodeToIndex(i)
        time_dim.SetCumulVarSoftUpperBound(idx, 0, max(weight[d], 1))

    # PRECEDENCE across crews: a damage can't be worked before its
    # nearest upstream damage is fully repaired. (Chains are transitive,
    # so constraining nearest-upstream only is sufficient.)
    solver = routing.solver()
    node_of = {d: i for i, d in enumerate(damaged, start=1)}
    for d, chain in precedence.items():
        if chain:
            up = chain[0]
            i_d = manager.NodeToIndex(node_of[d])
            i_u = manager.NodeToIndex(node_of[up])
            solver.Add(
                time_dim.CumulVar(i_d) >= time_dim.CumulVar(i_u) + repair[node_of[up]]
            )

    for i in range(1, len(road_nodes)):
        routing.AddDisjunction([manager.NodeToIndex(i)], 10**12)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(TIME_LIMIT_S)
    print(f"Solving weighted-latency VRP with precedence "
          f"(GLS, {TIME_LIMIT_S} s) ...")
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("No solution found")

    # Extract per-crew sequences and completion times of each repair.
    crews, completion = [], {}
    for v in range(NUM_CREWS):
        index = routing.Start(v)
        seq = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                d = damaged[node - 1]
                start = solution.Value(time_dim.CumulVar(index))
                completion[d] = start + repair[node]
                seq.append(d)
            index = solution.Value(routing.NextVar(index))
        crews.append(seq)
    return road_nodes, crews, completion, depot_node


# ----------------------------------------------------------------------
# Step 5: true energization times + reporting + maps
# ----------------------------------------------------------------------
def energization_times(completion, precedence):
    """A customer group on damage d is energized only when d AND every
    damaged span above it are repaired."""
    return {
        d: max([completion[d]] + [completion[u] for u in precedence[d]])
        for d in completion
    }


def restoration_curve(energized, weight, filename):
    events = sorted((t, weight[d]) for d, t in energized.items() if weight[d])
    total_out = sum(w for _, w in events)
    times = [0] + [t / 3600 for t, _ in events]
    without_power = [total_out]
    for _, w in events:
        without_power.append(without_power[-1] - w)
    cmi = sum(t / 60 * w for t, w in events)  # customer-minutes

    # Utility convention: customers WITHOUT power, decaying to zero.
    # The shaded area under this curve IS the CMI.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(times, without_power, where="post", lw=2, color="tab:red")
    ax.fill_between(times, without_power, step="post", alpha=0.15,
                    color="tab:red", label="area = CMI")
    ax.set_xlabel("hours since crews dispatched")
    ax.set_ylabel("customers without power")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Outage curve â€” {total_out:,} customers out at t=0, "
                 f"CMI = {cmi/1e3:,.0f}k customer-minutes")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    print(f"Saved restoration curve -> {filename}")
    return cmi, total_out


SUB_COLORS_MPL = ["tab:blue", "tab:green", "tab:red", "tab:purple"]
CREW_COLORS_MPL = ["red", "blue", "green", "orange", "magenta",
                   "teal", "black", "saddlebrown"]


def render_grid_png(G, Gu, substations, parent, root, edge_class, filename,
                    title, damaged=None, weight=None, crews=None,
                    depot_node=None):
    """Static map of the mock grid. Three flavors:
    - raw grid (substations + backbone + laterals)
    - + storm damage (X marks sized by customers restored when fixed)
    - + crew assignment (damage colored by assigned crew)
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0,
                            edge_color="#eeeeee", edge_linewidth=0.3,
                            bgcolor="white", figsize=(11, 11))

    sub_idx = {s: k for k, s in enumerate(substations)}
    lateral_segs, backbone_segs, backbone_cols = [], [], []
    for n, p in parent.items():
        if p is None:
            continue
        seg = [(Gu.nodes[p]["x"], Gu.nodes[p]["y"]),
               (Gu.nodes[n]["x"], Gu.nodes[n]["y"])]
        if edge_class[n] == "backbone":
            backbone_segs.append(seg)
            backbone_cols.append(SUB_COLORS_MPL[sub_idx[root[n]] % 4])
        else:
            lateral_segs.append(seg)
    ax.add_collection(LineCollection(lateral_segs, colors="#aaaaaa",
                                     linewidths=0.7, alpha=0.6, zorder=2))
    ax.add_collection(LineCollection(backbone_segs, colors=backbone_cols,
                                     linewidths=2.8, alpha=0.95, zorder=3))

    for k, s in enumerate(substations):
        ax.scatter(Gu.nodes[s]["x"], Gu.nodes[s]["y"], marker="*", s=600,
                   c=SUB_COLORS_MPL[k % 4], edgecolors="black",
                   linewidths=1.2, zorder=6)

    legend = [
        Line2D([], [], color="#888888", lw=1, label="lateral"),
        Line2D([], [], color="tab:blue", lw=3, label="feeder backbone"),
        Line2D([], [], marker="*", ls="", ms=18, markerfacecolor="tab:blue",
               markeredgecolor="black", label="substation"),
    ]

    if damaged is not None and crews is None:
        # X marks, sized by the customers that come back when it's fixed
        xs = [Gu.nodes[d]["x"] for d in damaged]
        ys = [Gu.nodes[d]["y"] for d in damaged]
        sizes = [40 + 0.5 * weight[d] for d in damaged]
        ax.scatter(xs, ys, marker="X", s=sizes, c="black",
                   edgecolors="yellow", linewidths=0.8, zorder=7)
        legend.append(Line2D([], [], marker="X", ls="", ms=12,
                             markerfacecolor="black",
                             markeredgecolor="yellow",
                             label="damaged span (size = customers)"))

    if crews is not None:
        for v, seq in enumerate(crews):
            xs = [Gu.nodes[d]["x"] for d in seq]
            ys = [Gu.nodes[d]["y"] for d in seq]
            ax.scatter(xs, ys, marker="o",
                       s=[60 + 0.4 * weight[d] for d in seq],
                       c=CREW_COLORS_MPL[v % 8], edgecolors="white",
                       linewidths=0.8, zorder=7)
            for order, d in enumerate(seq, start=1):
                ax.annotate(str(order),
                            (Gu.nodes[d]["x"], Gu.nodes[d]["y"]),
                            fontsize=6, ha="center", va="center",
                            color="white", zorder=8)
        legend.append(Line2D([], [], marker="o", ls="", ms=10,
                             markerfacecolor="gray",
                             markeredgecolor="white",
                             label="repair job (color = crew, # = order)"))

    if depot_node is not None:
        ax.scatter(G.nodes[depot_node]["x"], G.nodes[depot_node]["y"],
                   marker="s", s=200, c="dimgray", edgecolors="black",
                   zorder=6)
        legend.append(Line2D([], [], marker="s", ls="", ms=12,
                             markerfacecolor="dimgray",
                             markeredgecolor="black",
                             label="crew depot"))

    ax.set_title(title)
    ax.legend(handles=legend, loc="lower left", fontsize=9)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {filename}")


def render_map(G, Gu, substations, parent, edge_class, damaged, weight,
               crews, energized, depot_node, filename):
    import folium

    sub_colors = ["darkblue", "darkgreen", "darkred", "purple"]
    crew_colors = ["red", "blue", "green", "orange", "magenta",
                   "cadetblue", "black", "brown"]

    dy, dx = G.nodes[depot_node]["y"], G.nodes[depot_node]["x"]
    m = folium.Map(location=[dy, dx], zoom_start=13, tiles="cartodbpositron")

    # Grid: laterals faint, backbone bold (colored by feeding substation).
    sub_idx = {s: k for k, s in enumerate(substations)}
    for n, p in parent.items():
        if p is None:
            continue
        pts = [(Gu.nodes[p]["y"], Gu.nodes[p]["x"]),
               (Gu.nodes[n]["y"], Gu.nodes[n]["x"])]
        if edge_class[n] == "backbone":
            # walk to the root to find which substation feeds this span
            r = n
            while parent[r] is not None:
                r = parent[r]
            folium.PolyLine(pts, color=sub_colors[sub_idx[r] % 4],
                            weight=4, opacity=0.7).add_to(m)
        else:
            folium.PolyLine(pts, color="gray", weight=1,
                            opacity=0.35).add_to(m)

    for k, s in enumerate(substations):
        folium.Marker([Gu.nodes[s]["y"], Gu.nodes[s]["x"]],
                      tooltip=f"Substation {k}",
                      icon=folium.Icon(color=sub_colors[k % 4],
                                       icon="bolt", prefix="fa")).add_to(m)
    folium.Marker([dy, dx], tooltip="Service center (crew depot)",
                  icon=folium.Icon(color="gray", icon="wrench",
                                   prefix="fa")).add_to(m)

    for v, seq in enumerate(crews):
        for order, d in enumerate(seq, start=1):
            folium.CircleMarker(
                [Gu.nodes[d]["y"], Gu.nodes[d]["x"]],
                radius=7 if edge_class[d] == "backbone" else 5,
                color=crew_colors[v % 8], fill=True, fill_opacity=0.9,
                tooltip=(f"Crew {v}, job #{order} ({edge_class[d]}): "
                         f"{weight[d]:,} customers back at "
                         f"t={energized[d]/3600:.1f} h"),
            ).add_to(m)

    m.save(filename)
    print(f"Saved grid map -> {filename}")


def main():
    rng = random.Random(SEED)
    G = load_graph()
    Gu, substations, parent, root, customers, downstream, edge_class = \
        build_grid(G, rng)

    render_grid_png(
        G, Gu, substations, parent, root, edge_class,
        "output/04a_grid_raw.png",
        "Mock distribution grid over Hartford roads â€” 4 substations, "
        "feeder backbones, laterals")

    damaged, weight, precedence = generate_damage(parent, edge_class,
                                                  customers, rng)

    render_grid_png(
        G, Gu, substations, parent, root, edge_class,
        "output/04b_grid_outages.png",
        f"Storm damage â€” {NUM_DAMAGES} broken spans, "
        f"{sum(weight.values()):,} customers without power",
        damaged=damaged, weight=weight)

    road_nodes, crews, completion, depot_node = solve_restoration(
        G, damaged, weight, precedence, edge_class)

    energized = energization_times(completion, precedence)

    print("\n--- Restoration plan ---")
    for v, seq in enumerate(crews):
        if not seq:
            continue
        jobs = ", ".join(
            f"{edge_class[d][:4]}({weight[d]:,}c)" for d in seq)
        print(f"Crew {v}: {len(seq)} jobs -> {jobs}")

    cmi, total_out = restoration_curve(energized, weight,
                                       "output/04_outage_curve.png")
    last = max(energized.values()) / 3600
    print(f"\nAll {total_out:,} customers restored by t = {last:.1f} h")
    print(f"CMI = {cmi:,.0f} customer-minutes "
          f"(equiv. SAIDI contribution: {cmi/total_out:.0f} min/customer)")

    render_grid_png(
        G, Gu, substations, parent, root, edge_class,
        "output/04c_restoration_plan.png",
        f"Restoration plan â€” {NUM_CREWS} crews, all customers back in "
        f"{last:.1f} h",
        damaged=damaged, weight=weight, crews=crews, depot_node=depot_node)

    render_map(G, Gu, substations, parent, edge_class, damaged, weight,
               crews, energized, depot_node, "output/04_grid_map.html")


if __name__ == "__main__":
    main()

