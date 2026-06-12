"""
Example 3: Realistic CVRP on Hartford's actual road network
============================================================
10 trucks must service 100 locations spread across Hartford, CT
(think: storm-debris pickup sites, salt/sand drops, inspection points).

What makes this "realistic" compared to Examples 1-2:

1. REAL ROADS    : the drivable street network of Hartford is downloaded
                   from OpenStreetMap via OSMnx (cached locally after the
                   first run).
2. REAL COSTS    : the cost of going from A to B is the shortest-path
                   TRAVEL TIME through the road network (using per-edge
                   speed limits), not straight-line distance. The matrix
                   is ASYMMETRIC — one-way streets and highway ramps mean
                   A->B != B->A. OR-Tools handles this natively.
3. REAL DEPOT    : routes start/end at the Hartford DPW yard on
                   Jennings Road.
4. REAL OUTPUT   : each truck's route is rendered along the actual
                   streets it would drive (interactive HTML map + PNG).

Pipeline:
  OSM download -> largest strongly-connected component -> sample 100
  service nodes -> 101x101 travel-time matrix (one Dijkstra per source)
  -> OR-Tools CVRP with Guided Local Search -> snap routes back onto
  the street network for display.

NOTE ON PROBLEM CHOICE: if the job is literally "plow every street",
that is a Capacitated ARC Routing Problem (CARP) — you must traverse
edges, not visit nodes. See README. Here we model the also-real node
version: a set of locations to service using the road network.
"""

import os
import pickle
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

SEED = 7
PLACE = "Hartford, Connecticut, USA"
NUM_LOCATIONS = 100
NUM_TRUCKS = 10
TRUCK_CAPACITY = 13          # max service "loads" per truck per route
DEPOT_LATLON = (41.7896, -72.6747)  # Hartford DPW, 40 Jennings Rd
TIME_LIMIT_S = 30

GRAPH_CACHE = "cache/hartford_drive.graphml"
MATRIX_CACHE = "cache/hartford_matrix.pkl"


# ----------------------------------------------------------------------
# Step 1: get the road network
# ----------------------------------------------------------------------
def load_graph():
    if os.path.exists(GRAPH_CACHE):
        print(f"Loading cached graph: {GRAPH_CACHE}")
        G = ox.load_graphml(GRAPH_CACHE)
    else:
        print(f"Downloading road network for {PLACE} ...")
        G = ox.graph_from_place(PLACE, network_type="drive")
        # Attach speed (km/h) and travel_time (seconds) to every edge,
        # imputing from OSM speed limits / road class where missing.
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
        ox.save_graphml(G, GRAPH_CACHE)
        print(f"Saved -> {GRAPH_CACHE}")

    # Keep the largest strongly connected component so every node can
    # reach every other node (one-way streets make this non-trivial).
    largest = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    print(f"Graph: {len(G.nodes):,} intersections, {len(G.edges):,} road segments")
    return G


# ----------------------------------------------------------------------
# Step 2: pick depot + 100 service locations, build travel-time matrix
# ----------------------------------------------------------------------
def build_instance(G):
    rng = random.Random(SEED)
    depot_node = ox.distance.nearest_nodes(G, DEPOT_LATLON[1], DEPOT_LATLON[0])
    candidates = [n for n in G.nodes if n != depot_node]
    service_nodes = rng.sample(candidates, NUM_LOCATIONS)
    nodes = [depot_node] + service_nodes  # index 0 = depot

    if os.path.exists(MATRIX_CACHE):
        with open(MATRIX_CACHE, "rb") as f:
            cached = pickle.load(f)
        if cached["nodes"] == nodes:
            print(f"Loading cached travel-time matrix: {MATRIX_CACHE}")
            return nodes, cached["matrix"]

    print(f"Computing {len(nodes)}x{len(nodes)} travel-time matrix "
          f"(one Dijkstra per source) ...")
    n = len(nodes)
    node_pos = {node: i for i, node in enumerate(nodes)}
    matrix = [[0] * n for _ in range(n)]
    for i, src in enumerate(nodes):
        # Travel time (seconds) from src to every node in the graph.
        times = nx.single_source_dijkstra_path_length(G, src, weight="travel_time")
        for node, t in times.items():
            j = node_pos.get(node)
            if j is not None:
                matrix[i][j] = int(round(t))

    with open(MATRIX_CACHE, "wb") as f:
        pickle.dump({"nodes": nodes, "matrix": matrix}, f)
    print(f"Saved -> {MATRIX_CACHE}")
    return nodes, matrix


# ----------------------------------------------------------------------
# Step 3: solve the CVRP
# ----------------------------------------------------------------------
def solve_cvrp(matrix, demands, capacities):
    manager = pywrapcp.RoutingIndexManager(len(matrix), len(capacities), 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(fi, ti):
        return matrix[manager.IndexToNode(fi)][manager.IndexToNode(ti)]

    transit = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand_cb(fi):
        return demands[manager.IndexToNode(fi)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, capacities, True, "Load")

    # Optional but realistic: cap each shift at 4 hours of driving.
    routing.AddDimension(transit, 0, 4 * 3600, True, "Time")

    # Soft route-balancing: penalize the longest route a bit so one truck
    # doesn't do all the work while others sit idle.
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetGlobalSpanCostCoefficient(50)

    for node in range(1, len(matrix)):
        routing.AddDisjunction([manager.NodeToIndex(node)], 100_000_000)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(TIME_LIMIT_S)
    print(f"Solving (GLS, {TIME_LIMIT_S} s time limit) ...")
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("No solution found")

    routes = []
    for v in range(len(capacities)):
        index = routing.Start(v)
        stops = []
        while not routing.IsEnd(index):
            stops.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        stops.append(manager.IndexToNode(index))
        secs = sum(matrix[a][b] for a, b in zip(stops, stops[1:]))
        routes.append((stops, secs))
    return routes


# ----------------------------------------------------------------------
# Step 4: render routes along actual streets
# ----------------------------------------------------------------------
def route_to_street_path(G, nodes, stops):
    """Expand depot->stop->...->depot into the full node path on roads."""
    full = []
    for a, b in zip(stops, stops[1:]):
        seg = nx.shortest_path(G, nodes[a], nodes[b], weight="travel_time")
        full.extend(seg[:-1])
    full.append(nodes[stops[-1]])
    return full


def render(G, nodes, routes):
    colors = ["red", "blue", "green", "purple", "orange", "darkred",
              "cadetblue", "darkgreen", "magenta", "black"]

    # --- Interactive folium map ---
    import folium

    depot_y, depot_x = G.nodes[nodes[0]]["y"], G.nodes[nodes[0]]["x"]
    m = folium.Map(location=[depot_y, depot_x], zoom_start=13, tiles="cartodbpositron")
    folium.Marker([depot_y, depot_x], tooltip="Depot (Hartford DPW)",
                  icon=folium.Icon(color="gray", icon="home")).add_to(m)

    for v, (stops, secs) in enumerate(routes):
        if len(stops) <= 2:
            continue
        path = route_to_street_path(G, nodes, stops)
        latlons = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
        folium.PolyLine(latlons, color=colors[v % len(colors)], weight=3.5,
                        opacity=0.8,
                        tooltip=f"Truck {v}: {len(stops)-2} stops, "
                                f"{secs/60:.0f} min driving").add_to(m)
        for s in stops[1:-1]:
            y, x = G.nodes[nodes[s]]["y"], G.nodes[nodes[s]]["x"]
            folium.CircleMarker([y, x], radius=4, color=colors[v % len(colors)],
                                fill=True, fill_opacity=0.9,
                                tooltip=f"Stop {s} (truck {v})").add_to(m)

    html_path = "output/03_hartford_routes.html"
    m.save(html_path)
    print(f"Saved interactive map -> {html_path}")

    # --- Static PNG over the street grid ---
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0,
                            edge_color="#cccccc", edge_linewidth=0.4,
                            bgcolor="white", figsize=(11, 11))
    for v, (stops, secs) in enumerate(routes):
        if len(stops) <= 2:
            continue
        path = route_to_street_path(G, nodes, stops)
        xs = [G.nodes[n]["x"] for n in path]
        ys = [G.nodes[n]["y"] for n in path]
        ax.plot(xs, ys, color=colors[v % len(colors)], lw=1.6, alpha=0.85, zorder=3)
    sx = [G.nodes[nodes[i]]["x"] for i in range(1, len(nodes))]
    sy = [G.nodes[nodes[i]]["y"] for i in range(1, len(nodes))]
    ax.scatter(sx, sy, s=14, c="black", zorder=4)
    ax.scatter(G.nodes[nodes[0]]["x"], G.nodes[nodes[0]]["y"],
               s=160, c="red", marker="s", zorder=5)
    ax.set_title(f"Hartford CVRP: {NUM_TRUCKS} trucks, {NUM_LOCATIONS} locations "
                 f"(red square = DPW depot)")
    png_path = "output/03_hartford_routes.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved static map -> {png_path}")


def main():
    G = load_graph()
    nodes, matrix = build_instance(G)

    demands = [0] + [1] * NUM_LOCATIONS
    capacities = [TRUCK_CAPACITY] * NUM_TRUCKS
    routes = solve_cvrp(matrix, demands, capacities)

    print("\n--- Solution ---")
    total = 0
    for v, (stops, secs) in enumerate(routes):
        n_stops = len(stops) - 2
        print(f"Truck {v}: {n_stops:3d} stops, {secs/60:6.1f} min driving")
        total += secs
    served = sum(len(s) - 2 for s, _ in routes)
    print(f"\nLocations served: {served}/{NUM_LOCATIONS}")
    print(f"Total fleet drive time: {total/3600:.1f} hours")

    render(G, nodes, routes)


if __name__ == "__main__":
    main()
