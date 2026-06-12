"""
Example 2: A bigger, generated CVRP — and why search strategy matters
======================================================================
Steps up from Example 1 in three ways:

1. SCALE      : 60 stops + 1 depot, 6 vehicles (generated, reproducible).
2. REALISM    : demands and capacities in real units (tons of debris,
                10-ton trucks); distances in meters (Euclidean for now —
                Example 3 replaces this with the real road network).
3. SOLVER USE : compares a pure construction heuristic against
                metaheuristic improvement (Guided Local Search), showing
                that *how long and how smartly you search* changes the
                answer. This is the key practical lesson with OR-Tools.

Also introduces:
- Penalties / disjunctions: allow dropping a stop at a (large) cost, so the
  model stays feasible even if demand exceeds total fleet capacity.
- A matplotlib plot of the resulting routes.
"""

import math
import random

import matplotlib

matplotlib.use("Agg")  # no display needed; we save to a file
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

SEED = 42
NUM_STOPS = 60
NUM_VEHICLES = 6
TRUCK_CAPACITY_TONS = 25  # fleet capacity 150 vs ~116 tons demand (~77% utilization)
CITY_SIZE_M = 12_000  # stops scattered over a 12 km x 12 km square


def create_data_model():
    rng = random.Random(SEED)
    # Depot in the center of town.
    coords = [(CITY_SIZE_M / 2, CITY_SIZE_M / 2)]
    coords += [
        (rng.uniform(0, CITY_SIZE_M), rng.uniform(0, CITY_SIZE_M))
        for _ in range(NUM_STOPS)
    ]
    # Demand: most stops need 1-2 tons hauled, a few need 3-4.
    demands = [0] + [rng.choices([1, 2, 3, 4], weights=[40, 35, 15, 10])[0]
                     for _ in range(NUM_STOPS)]

    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = int(math.dist(coords[i], coords[j]))
            dist[i][j] = dist[j][i] = d

    return {
        "coords": coords,
        "distance_matrix": dist,
        "demands": demands,
        "vehicle_capacities": [TRUCK_CAPACITY_TONS] * NUM_VEHICLES,
        "num_vehicles": NUM_VEHICLES,
        "depot": 0,
    }


def build_model(data):
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return data["distance_matrix"][manager.IndexToNode(from_index)][
            manager.IndexToNode(to_index)
        ]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index):
        return data["demands"][manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, data["vehicle_capacities"], True, "Capacity"
    )

    # Disjunctions: each stop MAY be dropped, but at a huge penalty.
    # In practice none get dropped unless the instance is infeasible —
    # this keeps the solver from simply failing on hard instances.
    DROP_PENALTY = 1_000_000  # meters; far more than any detour costs
    for node in range(1, len(data["distance_matrix"])):
        routing.AddDisjunction([manager.NodeToIndex(node)], DROP_PENALTY)

    return manager, routing


def solve(data, use_gls, time_limit_s=10):
    manager, routing = build_model(data)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if use_gls:
        # Guided Local Search: escapes local optima by penalizing
        # frequently-used arcs. The workhorse metaheuristic for VRPs.
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.time_limit.FromSeconds(time_limit_s)
    else:
        # Stop at the very first constructed solution. (Without this,
        # OR-Tools quietly runs greedy-descent local search even when no
        # metaheuristic is set, which hides how rough the raw heuristic is.)
        params.solution_limit = 1
    solution = routing.SolveWithParameters(params)
    return manager, routing, solution


def extract_routes(data, manager, routing, solution):
    routes, total = [], 0
    for v in range(data["num_vehicles"]):
        index = routing.Start(v)
        nodes, dist = [], 0
        while not routing.IsEnd(index):
            nodes.append(manager.IndexToNode(index))
            prev = index
            index = solution.Value(routing.NextVar(index))
            dist += routing.GetArcCostForVehicle(prev, index, v)
        nodes.append(manager.IndexToNode(index))
        routes.append((nodes, dist))
        total += dist
    return routes, total


def plot_routes(data, routes, total, filename):
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 9))
    xs, ys = zip(*data["coords"])
    ax.scatter(xs[1:], ys[1:], c="gray", s=25, zorder=2, label="stops")
    ax.scatter(*data["coords"][0], c="red", marker="s", s=120, zorder=3, label="depot")
    for v, (nodes, dist) in enumerate(routes):
        if len(nodes) <= 2:
            continue  # unused vehicle
        pts = [data["coords"][n] for n in nodes]
        rx, ry = zip(*pts)
        ax.plot(rx, ry, "-", color=colors[v % 10], lw=1.5, zorder=1,
                label=f"truck {v} ({dist/1000:.1f} km)")
    ax.set_title(f"CVRP: {NUM_STOPS} stops, {NUM_VEHICLES} trucks — total {total/1000:.1f} km")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    print(f"Saved plot -> {filename}")


def main():
    data = create_data_model()
    print(f"Instance: {NUM_STOPS} stops, total demand "
          f"{sum(data['demands'])} tons, fleet capacity "
          f"{sum(data['vehicle_capacities'])} tons\n")

    # --- Run A: construction heuristic only (instant, mediocre) ---
    manager, routing, sol = solve(data, use_gls=False)
    routes_a, total_a = extract_routes(data, manager, routing, sol)
    print(f"PATH_CHEAPEST_ARC only:          total = {total_a/1000:8.1f} km")

    # --- Run B: + Guided Local Search for 10 seconds (much better) ---
    manager, routing, sol = solve(data, use_gls=True, time_limit_s=10)
    routes_b, total_b = extract_routes(data, manager, routing, sol)
    print(f"+ Guided Local Search (10 s):    total = {total_b/1000:8.1f} km")
    print(f"Improvement: {100 * (total_a - total_b) / total_a:.1f}%\n")

    for v, (nodes, dist) in enumerate(routes_b):
        load = sum(data["demands"][n] for n in nodes)
        print(f"Truck {v}: {len(nodes) - 2:2d} stops, {load:2d}/"
              f"{TRUCK_CAPACITY_TONS} tons, {dist/1000:.1f} km")

    plot_routes(data, routes_b, total_b, "output/02_routes.png")


if __name__ == "__main__":
    main()
