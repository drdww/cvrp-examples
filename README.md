# CVRP Examples: from textbook to Hartford's real road network

A graded sequence of Capacitated Vehicle Routing Problem (CVRP) examples,
built for teaching. Each example introduces one layer of realism.

## Setup

```
pip install ortools osmnx folium matplotlib scipy
```

## The examples

| # | Script | Scale | What it adds |
|---|--------|-------|--------------|
| 1 | `01_basic_cvrp.py` | 16 stops, 4 trucks | The core OR-Tools machinery: index manager, transit callback, capacity dimension |
| 2 | `02_cvrp_random_city.py` | 60 stops, 6 trucks | Real units, drop penalties (disjunctions), and the construction-heuristic vs. Guided Local Search comparison |
| 3 | `03_hartford_road_network.py` | 100 stops, 10 trucks | **Real Hartford streets** from OpenStreetMap, asymmetric travel-time matrix, shift-length limits, route balancing, interactive map |

Run them in order:

```
python 01_basic_cvrp.py
python 02_cvrp_random_city.py
python 03_hartford_road_network.py   # first run downloads OSM data (~1 min), then cached
```

Example 3 writes `output/03_hartford_routes.html` — open it in a browser and
hover over routes/stops.

## How the "realistic" pipeline works (Example 3)

1. **Road network**: OSMnx pulls Hartford's drivable streets from
   OpenStreetMap as a directed graph (one-way streets preserved). Each edge
   gets a speed (from OSM speed limits, imputed by road class where missing)
   and hence a travel time.
2. **Strong connectivity**: we keep the largest strongly connected component
   so every location can reach every other (otherwise one-way streets can
   strand the solver).
3. **Distance matrix**: one Dijkstra per location (101 total) gives a
   101×101 matrix of *road* travel times. It is asymmetric — A→B ≠ B→A —
   and OR-Tools consumes it without modification.
4. **Solve**: CVRP with capacity, a 4-hour shift cap per truck, a global-span
   coefficient to balance workloads, and Guided Local Search for 30 s.
5. **Render**: solver output is a sequence of stops; we re-expand each leg
   into the actual street path for display.

This is exactly the architecture of a production routing stack — in industry
you'd swap step 1–3 for a routing engine like **OSRM**, **Valhalla**, or the
Google Distance Matrix API (faster matrices, live traffic), and keep
steps 4–5 unchanged.

## ⚠️ Node routing vs. arc routing (important for "clearing roads")

If the task is "10 plows must clear *every street* in Hartford", that is
**not** a CVRP. Visiting locations = **node routing** (CVRP/TSP). Covering
every road segment = **arc routing**:

- Chinese Postman Problem (one vehicle, traverse every edge)
- **Capacitated Arc Routing Problem (CARP)** — the snow-plow / street-sweeper
  / salt-spreader problem (k vehicles, capacity = salt/fuel, deadheading
  minimized)

OR-Tools has no native CARP solver. The standard trick is to **transform CARP
into a node-routing problem** (each required edge becomes a pair of nodes)
and then feed it to OR-Tools — that's a great "advanced" lecture. The
examples here model the equally real *node* version: debris pickup sites,
salt-pile drops, inspection points, deliveries.

## Roadmap: beyond OR-Tools

### Stage 2 — specialized metaheuristic solvers (usually beat OR-Tools on pure CVRP)

| Solver | Notes |
|--------|-------|
| **PyVRP** (`pip install pyvrp`) | State-of-the-art Hybrid Genetic Search (HGS), pure Python API, actively maintained, wins DIMACS/VRP competitions. Easiest next step — same matrix in, routes out. |
| **HGS-CVRP** (Vidal) | The reference C++ HGS implementation PyVRP builds on. |
| **LKH-3** | Lin-Kernighan-Helsgaun; phenomenal on TSP/CVRP, trickier licensing & file-based I/O. |

### Stage 3 — exact / mathematical programming

| Approach | Notes |
|----------|-------|
| **MILP (Gurobi / HiGHS / CP-SAT)** | Two/three-index flow formulations with MTZ or subtour-elimination cuts. Proves optimality but only to ~50–80 customers. Great for teaching duality/bounds. |
| **Branch-cut-and-price (VRPSolver)** | The academic state of the art for exact CVRP (hundreds of customers). |

A natural lecture: run PyVRP and a MILP on the same 30-customer instance —
the metaheuristic finds the optimum in seconds, the MILP *proves* it.

### Stage 4 — quantum and quantum-inspired

The honest framing: quantum does not beat classical on VRP today, but VRP
maps cleanly to **QUBO** (Quadratic Unconstrained Binary Optimization), which
makes it a great vehicle (sorry) for teaching the formulations.

1. **QUBO formulation**: binary variable x[v,i,t] = "vehicle v is at location
   i at step t"; constraints (visit-once, capacity) become quadratic
   penalties. Watch qubit counts explode: even 20 locations → thousands of
   binary variables. This *is* the lesson.
2. **Quantum annealing — D-Wave** (`pip install dwave-ocean-sdk`): hybrid
   solvers (LeapHybridCQM) actually accept ~100-location problems by
   splitting quantum/classical work. Free trial minutes available.
3. **Gate-model — QAOA via Qiskit** (`pip install qiskit qiskit-optimization`):
   solve a 4–5 location toy VRP on a simulator; demonstrates the variational
   loop, and why NISQ devices can't touch 100 locations yet.
4. **Quantum-inspired classical**: simulated-annealing / simulated-bifurcation
   QUBO samplers (`neal`, Fujitsu Digital Annealer style) — good baseline
   showing the QUBO *formulation* can be attacked classically too.

Suggested arc for a course module:
OR-Tools (this repo) → PyVRP on the same Hartford matrix → MILP bound on a
subsample → QUBO on a 5-stop toy → discussion of where quantum could matter.

## Files

```
01_basic_cvrp.py              # textbook 17-node CVRP
02_cvrp_random_city.py        # 60 stops, GLS comparison, matplotlib plot
03_hartford_road_network.py   # 100 stops on real Hartford streets
cache/                        # OSM graph + travel-time matrix (auto-created)
output/                       # plots and interactive maps
```
