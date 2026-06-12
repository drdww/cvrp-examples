"""
Example 5: Dispatch under uncertainty — static vs. greedy vs. rolling horizon
==============================================================================
Examples 1-4 solved a SNAPSHOT: all damage known, repair times exact.
Real storms are not like that. This experiment makes the Example 4 world
dynamic and races three dispatch policies on the SAME realized storm:

  UNCERTAINTY (planner doesn't know):
  - Only ~40% of damage is known at t=0; the rest is discovered by
    damage-assessment over the first 3 hours.
  - True repair times are lognormal around the estimates (a "45-minute
    fuse" is sometimes a 2-hour broken pole).

  POLICIES:
  (a) STATIC   : solve the full VRP once at t=0 on known damage; crews
                 follow the plan, then take late-discovered jobs in
                 first-come-first-served order. (The "solve once" trap.)
  (b) GREEDY   : every time a crew frees up, it grabs the available job
                 with the best customers / (drive + est. repair) ratio.
                 Myopic, but always uses current information.
  (c) ROLLING  : every time a crew frees up, re-solve a weighted-latency
                 routing problem from its current position over all
                 revealed unfinished jobs, and take the first job of the
                 plan. (Re-optimization with current information AND
                 lookahead — the policy real systems approximate.)

  SCORE: realized customer-minutes of interruption (CMI), with
  energization computed honestly — a repair counts only once every
  damaged span upstream of it is also fixed.

Kept deliberately small (24 outages, 5 crews, 1-2 s solver calls) so the
whole experiment runs in about a minute. The point isn't scale — it's
the gap between the three curves in output/05_policy_comparison.png.

HONEST FINDING (seeds 7, 23, 42, 99): static is consistently 14-26%
worse, but greedy often TIES rolling at this scale — with 5 crews and 24
jobs there's little for lookahead to exploit. That's a real result from
the dynamic-VRP literature: the dominant value is using current
information at all; re-optimization's edge over a good myopic rule grows
with fleet size, job heterogeneity, and coupling (precedence, shifts).
Run `python 05_dynamic_dispatch.py <seed>` to try other storms.
"""

import importlib.util
import random
import sys
import time as wallclock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Reuse Example 4's grid-building machinery (filename starts with a digit,
# so import it by path).
spec = importlib.util.spec_from_file_location("ex4", "04_power_restoration.py")
ex4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex4)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7  # try other storms!
NUM_DAMAGES = 24
NUM_CREWS = 5
FRACTION_KNOWN_AT_T0 = 0.4
ASSESSMENT_WINDOW_S = 3 * 3600     # remaining damage discovered over 3 h
REPAIR_SIGMA = 0.55                # lognormal spread of true repair times
SOLVE_BUDGET_S = 1                 # per re-solve; 3 s for the static plan


# ----------------------------------------------------------------------
# Scenario generation (shared by all three policies: common random numbers)
# ----------------------------------------------------------------------
def build_scenario(rng):
    G = ex4.load_graph()
    Gu, substations, parent, root, customers, downstream, edge_class = \
        ex4.build_grid(G, rng)
    ex4.NUM_DAMAGES = NUM_DAMAGES
    damaged, weight, precedence = ex4.generate_damage(
        parent, edge_class, customers, rng)

    depot = ex4.ox.distance.nearest_nodes(
        G, ex4.DEPOT_LATLON[1], ex4.DEPOT_LATLON[0])
    road_nodes = [depot] + damaged
    M = ex4.travel_matrix(G, road_nodes)

    n_known = round(FRACTION_KNOWN_AT_T0 * NUM_DAMAGES)
    known = set(rng.sample(range(NUM_DAMAGES), n_known))
    jobs = []
    for j, d in enumerate(damaged):
        est = (ex4.REPAIR_BACKBONE_S if edge_class[d] == "backbone"
               else ex4.REPAIR_LATERAL_S)
        true = est * rng.lognormvariate(-REPAIR_SIGMA**2 / 2, REPAIR_SIGMA)
        jobs.append({
            "id": j,
            "node": d,
            "midx": j + 1,                      # row/col in M (0 = depot)
            "reveal": 0.0 if j in known else rng.uniform(0, ASSESSMENT_WINDOW_S),
            "est": est,
            "true": true,
            "weight": weight[d],
            "chain": precedence[d],             # damaged ancestors (node ids)
        })
    print(f"Scenario: {NUM_DAMAGES} damages ({n_known} known at t=0), "
          f"{sum(w['weight'] for w in jobs):,} customers out")
    return jobs, M


# ----------------------------------------------------------------------
# The optimizer used by STATIC (once, all crews) and ROLLING (per dispatch)
# ----------------------------------------------------------------------
def solve_weighted_latency(job_list, start_midxs, M, time_budget_s):
    """Min sum(weight x start-of-repair) routes from given crew positions.
    Crews need not return: routes end at a zero-cost dummy node."""
    n_crews = len(start_midxs)
    sub = start_midxs + [j["midx"] for j in job_list]
    DUMMY = len(sub)
    size = len(sub) + 1
    est = {i + n_crews: j["est"] for i, j in enumerate(job_list)}
    wgt = {i + n_crews: j["weight"] for i, j in enumerate(job_list)}

    manager = pywrapcp.RoutingIndexManager(
        size, n_crews, list(range(n_crews)), [DUMMY] * n_crews)
    routing = pywrapcp.RoutingModel(manager)

    def transit(fi, ti):
        a, b = manager.IndexToNode(fi), manager.IndexToNode(ti)
        if a == DUMMY or b == DUMMY:
            return 0
        return M[sub[a]][sub[b]] + est.get(a, 0)

    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimension(cb, 0, 10**9, True, "Time")
    tdim = routing.GetDimensionOrDie("Time")
    for node in range(n_crews, size - 1):
        tdim.SetCumulVarSoftUpperBound(
            manager.NodeToIndex(node), 0, max(wgt[node], 1))
        routing.AddDisjunction([manager.NodeToIndex(node)], 10**12)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(time_budget_s)
    sol = routing.SolveWithParameters(params)

    routes = []
    for v in range(n_crews):
        idx, seq = routing.Start(v), []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node >= n_crews and node != DUMMY:
                seq.append(job_list[node - n_crews]["id"])
            idx = sol.Value(routing.NextVar(idx))
        routes.append(seq)
    return routes


# ----------------------------------------------------------------------
# Discrete-event simulation of one policy on the shared scenario
# ----------------------------------------------------------------------
def simulate(policy, jobs, M):
    by_id = {j["id"]: j for j in jobs}
    crew_pos = [0] * NUM_CREWS                  # matrix index (0 = depot)
    crew_free = [0.0] * NUM_CREWS
    done, in_progress = set(), set()
    completion = {}
    reveals = sorted(j["reveal"] for j in jobs)

    static_queues = None
    if policy == "static":
        t0_jobs = [j for j in jobs if j["reveal"] == 0.0]
        static_queues = solve_weighted_latency(
            t0_jobs, [0] * NUM_CREWS, M, time_budget_s=3)

    def available(t):
        return [j for j in jobs
                if j["reveal"] <= t and j["id"] not in done
                and j["id"] not in in_progress]

    def pick(crew, t):
        avail = available(t)
        if not avail:
            return None
        if policy == "static":
            q = static_queues[crew]
            while q and q[0] in done | in_progress:
                q.pop(0)
            if q:
                return by_id[q.pop(0)]
            # plan exhausted -> first-come-first-served on the backlog
            return min(avail, key=lambda j: j["reveal"])
        if policy == "greedy":
            return max(avail, key=lambda j: j["weight"] /
                       (M[crew_pos[crew]][j["midx"]] + j["est"] + 1))
        # rolling: re-solve from this crew's position, take the first job
        routes = solve_weighted_latency(
            avail, [crew_pos[crew]], M, time_budget_s=SOLVE_BUDGET_S)
        return by_id[routes[0][0]] if routes[0] else None

    while len(done) < len(jobs):
        crew = min(range(NUM_CREWS), key=lambda c: crew_free[c])
        t = crew_free[crew]
        job = pick(crew, t)
        if job is None:                          # idle until next discovery
            nxt = min(r for r in reveals if r > t)
            crew_free[crew] = nxt
            continue
        in_progress.add(job["id"])
        finish = t + M[crew_pos[crew]][job["midx"]] + job["true"]
        completion[job["node"]] = finish
        crew_pos[crew], crew_free[crew] = job["midx"], finish
        done.add(job["id"])
        in_progress.discard(job["id"])

    # honest energization: all upstream damage must also be repaired
    energized = {
        j["node"]: max([completion[j["node"]]]
                       + [completion[u] for u in j["chain"]])
        for j in jobs
    }
    weight_by_node = {j["node"]: j["weight"] for j in jobs}
    cmi = sum(t / 60 * weight_by_node[d] for d, t in energized.items())
    return energized, weight_by_node, cmi


# ----------------------------------------------------------------------
# Compare and plot
# ----------------------------------------------------------------------
def outage_steps(energized, weight):
    total = sum(weight.values())
    events = sorted((t, weight[d]) for d, t in energized.items() if weight[d])
    times, out = [0.0], [total]
    for t, w in events:
        times.append(t / 3600)
        out.append(out[-1] - w)
    return times, out


def main():
    rng = random.Random(SEED)
    jobs, M = build_scenario(rng)

    styles = {"static": ("tab:gray", "(a) static plan + FCFS"),
              "greedy": ("tab:orange", "(b) greedy dispatch"),
              "rolling": ("tab:green", "(c) rolling re-optimization")}
    results = {}
    for policy in styles:
        t0 = wallclock.time()
        energized, weight, cmi = simulate(policy, jobs, M)
        results[policy] = (energized, weight, cmi)
        print(f"{policy:8s}: CMI = {cmi:10,.0f} customer-min   "
              f"all restored by {max(energized.values())/3600:4.1f} h   "
              f"(simulated in {wallclock.time()-t0:.0f} s)")

    base = results["static"][2]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for policy, (color, label) in styles.items():
        energized, weight, cmi = results[policy]
        times, out = outage_steps(energized, weight)
        ax.step(times, out, where="post", lw=2, color=color,
                label=f"{label} — CMI {cmi/1e3:,.0f}k "
                      f"({100*(cmi-base)/base:+.0f}%)")
    ax.set_xlabel("hours since crews dispatched")
    ax.set_ylabel("customers without power")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Same storm, three dispatch policies — "
                 f"{NUM_DAMAGES} outages, {NUM_CREWS} crews, "
                 f"{round(FRACTION_KNOWN_AT_T0*100)}% of damage known at t=0")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig("output/05_policy_comparison.png", dpi=130)
    print("Saved -> output/05_policy_comparison.png")


if __name__ == "__main__":
    main()
