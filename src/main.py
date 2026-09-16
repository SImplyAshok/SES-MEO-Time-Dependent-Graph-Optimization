from pathlib import Path
import pulp
import numpy as np
import heapq

from .config import REUSE_STORED_FEASIBILITY
from .network_data import build_network
from .Model import build_model, solve_model
from .reporting import extract, report
from .optimization import (
    solve_one_timestamp_route,
    run_joint_temporal_optimization,
    build_temporal_candidates,
    solve_sequential_dp,
)

def solve_one_timestamp_dijkstra(
    U_fes,
    I_fes,
    F_fes,
    L_US,
    L_ISL,
    L_F,
    C_US,
    C_ISL,
    C_F,
    demand=350.0,
):
    """
    Solve the one-timestamp routing problem using Dijkstra.

    The Stage-1 feasibility matrices define the available topology.

    A link is usable only if:
        availability = 1
        capacity >= demand

    For every feasible serving satellite s:

        1. Fix Z -> s as the service link.
        2. Run Dijkstra from s to the feasible gateways.
        3. Add the service-link latency.
        4. Keep the minimum-cost candidate.

    This is an exact shortest-path solution for the baseline
    one-timestamp problem with additive non-negative latency.

    Returns:
        U_one
        I_one
        F_one
        objective
        serving_satellite
        gateway
        path
        candidate_costs
        candidate_routes
    """

    # =====================================================================
    # CONVERT INPUTS TO NUMPY ARRAYS
    # =====================================================================

    U_fes = np.asarray(U_fes, dtype=float)
    I_fes = np.asarray(I_fes, dtype=float)
    F_fes = np.asarray(F_fes, dtype=float)

    L_US = np.asarray(L_US, dtype=float)
    L_ISL = np.asarray(L_ISL, dtype=float)
    L_F = np.asarray(L_F, dtype=float)

    C_US = np.asarray(C_US, dtype=float)
    C_ISL = np.asarray(C_ISL, dtype=float)
    C_F = np.asarray(C_F, dtype=float)

    n_sat = len(U_fes)
    n_gw = F_fes.shape[1]

    INF = float("inf")

    # =====================================================================
    # BUILD FEASIBLE SATELLITE -> SATELLITE GRAPH
    # =====================================================================

    adjacency = [
        []
        for _ in range(n_sat)
    ]

    for i in range(n_sat):

        for j in range(n_sat):

            if i == j:
                continue

            # -------------------------------------------------------------
            # ISL must be physically available and have enough capacity
            # -------------------------------------------------------------

            if (
                I_fes[i, j] > 0.5
                and C_ISL[i, j] >= demand
            ):

                adjacency[i].append(
                    (
                        j,
                        float(L_ISL[i, j]),
                    )
                )

    # =====================================================================
    # BUILD FEASIBLE FEEDER LINKS
    # =====================================================================

    feasible_feeders = [
        []
        for _ in range(n_sat)
    ]

    for i in range(n_sat):

        for g in range(n_gw):

            if (
                F_fes[i, g] > 0.5
                and C_F[i, g] >= demand
            ):

                feasible_feeders[i].append(
                    (
                        g,
                        float(L_F[i, g]),
                    )
                )

    # =====================================================================
    # DIJKSTRA FOR ONE SERVING SATELLITE
    # =====================================================================

    def shortest_route_from_serving(
        serving_idx,
    ):
        """
        Run Dijkstra from the selected serving satellite.

        The destination can be any satellite with a feasible
        feeder link to any gateway.
        """

        # -------------------------------------------------------------
        # Distance from serving satellite
        # -------------------------------------------------------------

        dist = [
            INF
            for _ in range(n_sat)
        ]

        # -------------------------------------------------------------
        # Predecessor used for path reconstruction
        # -------------------------------------------------------------

        predecessor = [
            None
            for _ in range(n_sat)
        ]

        dist[serving_idx] = 0.0

        # -------------------------------------------------------------
        # Priority queue
        #
        # (distance, node)
        # -------------------------------------------------------------

        heap = [
            (
                0.0,
                serving_idx,
            )
        ]

        best_gateway = None
        best_gateway_cost = INF

        # =============================================================
        # DIJKSTRA LOOP
        # =============================================================

        while heap:

            current_dist, u = heapq.heappop(
                heap
            )

            # ---------------------------------------------------------
            # Ignore outdated queue entry
            # ---------------------------------------------------------

            if (
                current_dist
                > dist[u] + 1e-12
            ):
                continue

            # =========================================================
            # CHECK FEEDER FROM CURRENT SATELLITE
            # =========================================================

            for (
                gateway_idx,
                feeder_latency,
            ) in feasible_feeders[u]:

                total_cost = (
                    current_dist
                    + feeder_latency
                )

                if (
                    total_cost
                    < best_gateway_cost
                ):

                    best_gateway_cost = (
                        total_cost
                    )

                    best_gateway = (
                        gateway_idx
                    )

            # =========================================================
            # RELAX ISL EDGES
            # =========================================================

            for (
                v,
                edge_latency,
            ) in adjacency[u]:

                new_dist = (
                    current_dist
                    + edge_latency
                )

                if (
                    new_dist
                    < dist[v] - 1e-12
                ):

                    dist[v] = new_dist

                    predecessor[v] = u

                    heapq.heappush(
                        heap,
                        (
                            new_dist,
                            v,
                        ),
                    )

        # =============================================================
        # NO GATEWAY REACHABLE
        # =============================================================

        if best_gateway is None:

            return (
                INF,
                None,
                None,
            )

        # =============================================================
        # FIND THE FEEDER SATELLITE
        # =============================================================

        feeder_satellite = None

        best_feeder_total = INF

        for u in range(n_sat):

            if not feasible_feeders[u]:
                continue

            for (
                gateway_idx,
                feeder_latency,
            ) in feasible_feeders[u]:

                if (
                    gateway_idx
                    == best_gateway
                ):

                    total = (
                        dist[u]
                        + feeder_latency
                    )

                    if (
                        abs(
                            total
                            - best_gateway_cost
                        )
                        < 1e-9
                    ):

                        if (
                            dist[u]
                            < best_feeder_total
                        ):

                            best_feeder_total = (
                                dist[u]
                            )

                            feeder_satellite = u

        if feeder_satellite is None:

            return (
                INF,
                None,
                None,
            )

        # =============================================================
        # RECONSTRUCT SATELLITE PATH
        # =============================================================

        satellite_path = [
            feeder_satellite
        ]

        current = feeder_satellite

        while current != serving_idx:

            current = predecessor[current]

            if current is None:

                return (
                    INF,
                    None,
                    None,
                )

            satellite_path.append(
                current
            )

        satellite_path.reverse()

        return (
            best_gateway_cost,
            satellite_path,
            best_gateway,
        )

    # =====================================================================
    # EVALUATE ALL FEASIBLE SERVING SATELLITES
    # =====================================================================

    candidate_costs = {}

    candidate_routes = {}

    feasible_serving_satellites = (
        np.where(
            U_fes > 0.5
        )[0]
    )

    for serving_idx in (
        feasible_serving_satellites
    ):

        serving_idx = int(
            serving_idx
        )

        # -------------------------------------------------------------
        # Service link must have sufficient capacity
        # -------------------------------------------------------------

        if (
            C_US[serving_idx]
            < demand
        ):

            candidate_costs[
                serving_idx
            ] = INF

            candidate_routes[
                serving_idx
            ] = None

            continue

        # -------------------------------------------------------------
        # Dijkstra from serving satellite
        # -------------------------------------------------------------

        (
            downstream_cost,
            satellite_path,
            gateway_idx,
        ) = shortest_route_from_serving(
            serving_idx
        )

        # -------------------------------------------------------------
        # No gateway reachable
        # -------------------------------------------------------------

        if satellite_path is None:

            candidate_costs[
                serving_idx
            ] = INF

            candidate_routes[
                serving_idx
            ] = None

            continue

        # -------------------------------------------------------------
        # End-to-end cost
        #
        # Z -> serving satellite
        # +
        # serving satellite -> ... -> gateway
        # -------------------------------------------------------------

        total_cost = (
            L_US[serving_idx]
            + downstream_cost
        )

        candidate_costs[
            serving_idx
        ] = float(
            total_cost
        )

        candidate_routes[
            serving_idx
        ] = {
            "satellite_path":
                satellite_path,

            "gateway":
                gateway_idx,

            "cost":
                float(total_cost),
        }

    # =====================================================================
    # FIND BEST SERVING SATELLITE
    # =====================================================================

    feasible_candidates = [

        s

        for s, cost
        in candidate_costs.items()

        if np.isfinite(cost)

    ]

    # =====================================================================
    # INFEASIBLE
    # =====================================================================

    if not feasible_candidates:

        return {

            "feasible":
                False,

            "status":
                "INFEASIBLE",

            "U_one":
                None,

            "I_one":
                None,

            "F_one":
                None,

            "objective":
                INF,

            "serving_satellite":
                None,

            "gateway":
                None,

            "path":
                None,

            "candidate_costs":
                candidate_costs,

            "candidate_routes":
                candidate_routes,
        }

    # =====================================================================
    # SELECT MINIMUM-COST SERVING SATELLITE
    # =====================================================================

    serving_idx = min(

        feasible_candidates,

        key=lambda s: (
            candidate_costs[s],
            s,
        ),

    )

    selected_route = (
        candidate_routes[
            serving_idx
        ]
    )

    satellite_path = (
        selected_route[
            "satellite_path"
        ]
    )

    gateway_idx = (
        selected_route[
            "gateway"
        ]
    )

    # =====================================================================
    # CREATE U / I / F MATRICES
    # =====================================================================

    U_one = np.zeros(
        n_sat,
        dtype=int,
    )

    I_one = np.zeros(
        (
            n_sat,
            n_sat,
        ),
        dtype=int,
    )

    F_one = np.zeros(
        (
            n_sat,
            n_gw,
        ),
        dtype=int,
    )

    # =====================================================================
    # SERVING SATELLITE
    # =====================================================================

    U_one[
        serving_idx
    ] = 1

    # =====================================================================
    # ISL PATH
    # =====================================================================

    for u, v in zip(
        satellite_path[:-1],
        satellite_path[1:],
    ):

        I_one[
            u,
            v,
        ] = 1

    # =====================================================================
    # FEEDER
    # =====================================================================

    feeder_satellite = (
        satellite_path[-1]
    )

    F_one[
        feeder_satellite,
        gateway_idx,
    ] = 1

    # =====================================================================
    # HUMAN-READABLE PATH
    # =====================================================================

    path = [

        "z"

    ]

    path.extend(

        f"S{i + 1}"

        for i
        in satellite_path

    )

    path.append(

        f"GW{gateway_idx + 1}"

    )

    # =====================================================================
    # RETURN RESULT
    # =====================================================================

    return {

        "feasible":
            True,

        "status":
            "OPTIMAL",

        "U_one":
            U_one,

        "I_one":
            I_one,

        "F_one":
            F_one,

        "objective":
            float(
                candidate_costs[
                    serving_idx
                ]
            ),

        "serving_satellite":
            serving_idx,

        "gateway":
            gateway_idx,

        "path":
            path,

        "candidate_costs":
            candidate_costs,

        "candidate_routes":
            candidate_routes,
    }

def get_serving_for_validation(U):

    if U is None:
        return None

    selected = np.where(
        np.asarray(U) > 0.5
    )[0]

    if len(selected) != 1:
        return None

    return f"S{int(selected[0]) + 1}"


def calculate_solution_latency_for_validation(
    k,
    U,
    I,
    F,
    L,
):

    if (
        U is None
        or I is None
        or F is None
    ):
        return None

    U = np.asarray(
        U,
        dtype=float,
    )

    I = np.asarray(
        I,
        dtype=float,
    )

    F = np.asarray(
        F,
        dtype=float,
    )

    return float(

        np.sum(
            L[k]["US"] * U
        )

        +

        np.sum(
            L[k]["ISL"] * I
        )

        +

        np.sum(
            L[k]["F"] * F
        )

    )

def compare_one_slot_vs_joint(
    E,
    L,
    one_timestamp_results,
    temporal_result,
    H=8.0,
):
    """
    Compare:

        Method 1:
            One-timestamp-at-a-time optimization

        Method 2:
            Joint temporal optimization

    For both methods calculate:

        - serving satellite at every time slot
        - total routing latency
        - number of handovers
        - handover penalty
        - total complete-horizon cost

    Total cost:

        J = total routing latency + H * number of handovers
    """

    # =========================================================================
    # LOCAL FUNCTION: CALCULATE ROUTING LATENCY
    # =========================================================================

    def calculate_latency(k, U, I, F):

        if U is None or I is None or F is None:
            return None

        U = np.asarray(U, dtype=float)
        I = np.asarray(I, dtype=float)
        F = np.asarray(F, dtype=float)

        latency = (
            np.sum(L[k]["US"] * U)
            + np.sum(L[k]["ISL"] * I)
            + np.sum(L[k]["F"] * F)
        )

        return float(latency)

    # =========================================================================
    # LOCAL FUNCTION: GET SERVING SATELLITE
    # =========================================================================

    def get_serving_satellite(U):

        if U is None:
            return None

        selected = np.where(
            np.asarray(U) > 0.5
        )[0]

        if len(selected) != 1:
            return None

        return f"S{int(selected[0]) + 1}"

    # =========================================================================
    # LOCAL FUNCTION: COUNT HANDOVERS
    # =========================================================================

    def count_handovers(serving_sequence):

        handovers = 0

        for k in range(1, len(serving_sequence)):

            previous_satellite = serving_sequence[k - 1]
            current_satellite = serving_sequence[k]

            if (
                previous_satellite is not None
                and current_satellite is not None
                and previous_satellite != current_satellite
            ):
                handovers += 1

        return handovers

    # =========================================================================
    # METHOD 1
    # ONE-TIMESTAMP-AT-A-TIME
    # =========================================================================

    one_serving = {}
    one_latency = {}

    for k in sorted(E):

        result = one_timestamp_results.get(
            k,
            {},
        )

        U_one = result.get("U_one")
        I_one = result.get("I_one")
        F_one = result.get("F_one")

        one_serving[k] = get_serving_satellite(
            U_one
        )

        one_latency[k] = calculate_latency(
            k,
            U_one,
            I_one,
            F_one,
        )

    # Total routing latency
    one_total_latency = sum(
        value
        for value in one_latency.values()
        if value is not None
    )

    # Serving-satellite sequence
    one_serving_sequence = [
        one_serving[k]
        for k in sorted(E)
    ]

    # Handover count
    one_handovers = count_handovers(
        one_serving_sequence
    )

    # Handover penalty
    one_handover_cost = H * one_handovers

    # Complete-horizon operational cost
    one_total_cost = (
        one_total_latency
        + one_handover_cost
    )

    # =========================================================================
    # METHOD 2
    # JOINT TEMPORAL OPTIMIZATION
    # =========================================================================

    U_temp = temporal_result.get(
        "U_temp",
        {},
    )

    I_temp = temporal_result.get(
        "I_temp",
        {},
    )

    F_temp = temporal_result.get(
        "F_temp",
        {},
    )

    joint_serving = {}
    joint_latency = {}

    for k in sorted(E):

        U_k = U_temp.get(k)
        I_k = I_temp.get(k)
        F_k = F_temp.get(k)

        joint_serving[k] = get_serving_satellite(
            U_k
        )

        joint_latency[k] = calculate_latency(
            k,
            U_k,
            I_k,
            F_k,
        )

    # Total routing latency
    joint_total_latency = sum(
        value
        for value in joint_latency.values()
        if value is not None
    )

    # Serving-satellite sequence
    joint_serving_sequence = [
        joint_serving[k]
        for k in sorted(E)
    ]

    # Handover count
    joint_handovers = count_handovers(
        joint_serving_sequence
    )

    # Handover penalty
    joint_handover_cost = H * joint_handovers

    # Complete-horizon operational cost
    joint_total_cost = (
        joint_total_latency
        + joint_handover_cost
    )

    # =========================================================================
    # DISPLAY SERVING SATELLITE SELECTION
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "SERVING SATELLITE COMPARISON"
    )
    print("=" * 100)

    print()
    print(
        f"{'Slot':<8}"
        f"{'One-slot':<20}"
        f"{'Joint temporal':<20}"
        f"{'One latency (ms)':<20}"
        f"{'Joint latency (ms)':<20}"
    )

    print("-" * 100)

    for k in sorted(E):

        one_sat = (
            one_serving[k]
            if one_serving[k] is not None
            else "N/A"
        )

        joint_sat = (
            joint_serving[k]
            if joint_serving[k] is not None
            else "N/A"
        )

        one_lat = one_latency[k]
        joint_lat = joint_latency[k]

        one_lat_text = (
            f"{one_lat:.3f}"
            if one_lat is not None
            else "N/A"
        )

        joint_lat_text = (
            f"{joint_lat:.3f}"
            if joint_lat is not None
            else "N/A"
        )

        print(
            f"{k:<8}"
            f"{one_sat:<20}"
            f"{joint_sat:<20}"
            f"{one_lat_text:<20}"
            f"{joint_lat_text:<20}"
        )

    # =========================================================================
    # DISPLAY SERVING SATELLITE SEQUENCES
    # =========================================================================

    print()
    print("SERVING SATELLITE SEQUENCES")
    print("-" * 100)

    print(
        "One-slot-at-a-time:"
    )

    print(
        " -> ".join(
            satellite if satellite is not None else "N/A"
            for satellite in one_serving_sequence
        )
    )

    print()

    print(
        "Joint temporal:"
    )

    print(
        " -> ".join(
            satellite if satellite is not None else "N/A"
            for satellite in joint_serving_sequence
        )
    )

    # =========================================================================
    # DISPLAY COST COMPARISON
    # =========================================================================

    print()
    print("=" * 100)
    print("COMPLETE-HORIZON COST COMPARISON")
    print("=" * 100)

    print()
    print(
        f"{'Metric':<40}"
        f"{'One-slot':>25}"
        f"{'Joint temporal':>25}"
    )

    print("-" * 100)

    print(
        f"{'Total routing latency (ms)':<40}"
        f"{one_total_latency:>25.3f}"
        f"{joint_total_latency:>25.3f}"
    )

    print(
        f"{'Number of handovers':<40}"
        f"{one_handovers:>25d}"
        f"{joint_handovers:>25d}"
    )

    print(
        f"{'Handover penalty':<40}"
        f"{one_handover_cost:>25.3f}"
        f"{joint_handover_cost:>25.3f}"
    )

    print("-" * 100)

    print(
        f"{'TOTAL COST':<40}"
        f"{one_total_cost:>25.3f}"
        f"{joint_total_cost:>25.3f}"
    )

    print("=" * 100)

    # =========================================================================
    # DIFFERENCE
    # =========================================================================

    cost_difference = (
        joint_total_cost
        - one_total_cost
    )

    latency_difference = (
        joint_total_latency
        - one_total_latency
    )

    handover_difference = (
        joint_handovers
        - one_handovers
    )

    print()
    print("JOINT TEMPORAL OPTIMIZATION EFFECT")
    print("-" * 100)

    print(
        f"Routing latency difference : "
        f"{latency_difference:+.3f} ms"
    )

    print(
        f"Handover difference        : "
        f"{handover_difference:+d}"
    )

    print(
        f"Total cost difference      : "
        f"{cost_difference:+.3f}"
    )

    if one_total_cost > 0:

        improvement = (
            (one_total_cost - joint_total_cost)
            / one_total_cost
            * 100.0
        )

        print(
            f"Cost improvement           : "
            f"{improvement:+.2f}%"
        )

    print()

    if joint_total_cost < one_total_cost:

        print(
            "RESULT: Joint temporal optimization "
            "achieves a LOWER total cost."
        )

    elif joint_total_cost > one_total_cost:

        print(
            "RESULT: One-slot-at-a-time optimization "
            "achieves a LOWER total cost."
        )

    else:

        print(
            "RESULT: Both methods achieve the SAME total cost."
        )

    print("=" * 100)

    # =========================================================================
    # RETURN COMPARISON RESULTS
    # =========================================================================

    return {
        "one_slot": {
            "serving": one_serving,
            "latency": one_latency,
            "total_latency": one_total_latency,
            "handovers": one_handovers,
            "handover_cost": one_handover_cost,
            "total_cost": one_total_cost,
        },
        "joint_temporal": {
            "serving": joint_serving,
            "latency": joint_latency,
            "total_latency": joint_total_latency,
            "handovers": joint_handovers,
            "handover_cost": joint_handover_cost,
            "total_cost": joint_total_cost,
        },
    }


def compare_one_slot_vs_joint_temporal(
    E,
    L,
    dijkstra_timestamp_results,
    dp_result,
    temporal_result,
    H=8.0,
):
    """
    Compare three temporal optimisation strategies:

    Method 1:
        One-timestamp-at-a-time optimisation using Dijkstra.

    Method 2:
        Sequential Dynamic Programming (DP).

    Method 3:
        Joint temporal optimisation.

    The three methods are reported using the same metrics:

        - serving-satellite sequence
        - total routing latency
        - number of handovers
        - handover penalty
        - total horizon cost

    Total cost:

        J = routing cost + H * number of handovers

    Notes
    -----
    The joint temporal solver currently returns:

        routing_cost
        handover_count
        handover_cost
        objective
        U_temp

    Therefore, if ``objective`` is not finite, the total cost is
    reconstructed from:

        routing_cost + handover_cost
    """

    time_slots = sorted(E)

    # =========================================================================
    # HELPER: CONVERT SATELLITE INDEX TO NAME
    # =========================================================================

    def satellite_name(index):

        if index is None:
            return None

        return f"S{int(index) + 1}"

    # =========================================================================
    # HELPER: COUNT HANDOVERS
    # =========================================================================

    def count_handovers(serving_sequence):

        handovers = 0

        previous_satellite = None

        for current_satellite in serving_sequence:

            if current_satellite is None:
                continue

            if (
                previous_satellite is not None
                and current_satellite != previous_satellite
            ):

                handovers += 1

            previous_satellite = current_satellite

        return handovers

    # =========================================================================
    # METHOD 1
    # ONE-TIMESTAMP-AT-A-TIME DIJKSTRA
    # =========================================================================

    one_serving = {}
    one_latency = {}

    for k in time_slots:

        result = dijkstra_timestamp_results.get(
            k,
            {},
        )

        if not result.get(
            "feasible",
            False,
        ):

            one_serving[k] = None
            one_latency[k] = None

            continue

        serving_idx = result.get(
            "serving_satellite"
        )

        one_serving[k] = satellite_name(
            serving_idx
        )

        # Dijkstra objective already contains:
        #
        # Z -> serving satellite
        # + ISL path
        # + feeder link
        #
        objective = result.get(
            "objective"
        )

        if (
            objective is not None
            and np.isfinite(objective)
        ):

            one_latency[k] = float(
                objective
            )

        else:

            one_latency[k] = None

    # -------------------------------------------------------------------------
    # Method 1 serving sequence
    # -------------------------------------------------------------------------

    one_serving_sequence = [

        one_serving[k]

        for k in time_slots

    ]

    # -------------------------------------------------------------------------
    # Method 1 handovers
    # -------------------------------------------------------------------------

    one_handover_count = count_handovers(
        one_serving_sequence
    )

    # -------------------------------------------------------------------------
    # Method 1 routing cost
    # -------------------------------------------------------------------------

    one_total_latency = sum(

        value

        for value in one_latency.values()

        if value is not None

    )

    # -------------------------------------------------------------------------
    # Method 1 handover cost
    # -------------------------------------------------------------------------

    one_handover_cost = (

        H
        * one_handover_count

    )

    # -------------------------------------------------------------------------
    # Method 1 total cost
    # -------------------------------------------------------------------------

    one_total_cost = (

        one_total_latency
        + one_handover_cost

    )

    # =========================================================================
    # METHOD 2
    # SEQUENTIAL DYNAMIC PROGRAMMING
    # =========================================================================

    dp_feasible = dp_result.get(
        "feasible",
        False,
    )

    dp_serving_raw = dp_result.get(
        "serving_sequence",
        {},
    )

    dp_routing_cost = dp_result.get(
        "total_routing_cost",
        np.inf,
    )

    dp_handover_count = dp_result.get(
        "handover_count",
        0,
    )

    dp_handover_cost = dp_result.get(
        "handover_cost",
        H * dp_handover_count,
    )

    dp_total_cost = dp_result.get(
        "total_cost",
        np.inf,
    )

    # -------------------------------------------------------------------------
    # Convert DP serving sequence to:
    #
    #     S2 -> S5 -> S4 -> ...
    #
    # DP currently returns a dictionary:
    #
    #     {0: 1, 1: 4, 2: 3, ...}
    #
    # where the values are zero-based satellite indices.
    # -------------------------------------------------------------------------

    dp_serving_sequence = []

    if isinstance(
        dp_serving_raw,
        dict,
    ):

        for k in time_slots:

            satellite_idx = dp_serving_raw.get(
                k
            )

            if satellite_idx is None:

                dp_serving_sequence.append(
                    None
                )

            else:

                dp_serving_sequence.append(
                    satellite_name(
                        satellite_idx
                    )
                )

    elif isinstance(
        dp_serving_raw,
        (list, tuple),
    ):

        for satellite_idx in dp_serving_raw:

            if satellite_idx is None:

                dp_serving_sequence.append(
                    None
                )

            else:

                dp_serving_sequence.append(
                    satellite_name(
                        satellite_idx
                    )
                )

    else:

        dp_serving_sequence = [
            None
            for _ in time_slots
        ]

    # -------------------------------------------------------------------------
    # Make sure DP handover count is consistent with sequence
    # -------------------------------------------------------------------------

    if dp_feasible:

        calculated_dp_handovers = count_handovers(
            dp_serving_sequence
        )

        # Use the actual sequence as the authoritative value.
        dp_handover_count = calculated_dp_handovers

        dp_handover_cost = (
            H
            * dp_handover_count
        )

    # -------------------------------------------------------------------------
    # If DP returned an invalid/non-finite total cost, reconstruct it.
    # -------------------------------------------------------------------------

    if not np.isfinite(
        dp_routing_cost
    ):

        dp_routing_cost = float(
            dp_result.get(
                "routing_cost",
                np.inf,
            )
        )

    if (
        dp_feasible
        and np.isfinite(dp_routing_cost)
    ):

        dp_total_cost = (

            float(dp_routing_cost)
            + float(dp_handover_cost)

        )

    # =========================================================================
    # METHOD 3
    # JOINT TEMPORAL OPTIMISATION
    # =========================================================================

    temporal_feasible = temporal_result.get(
        "feasible",
        False,
    )

    # -------------------------------------------------------------------------
    # Retrieve U_temp
    #
    # The joint solver returns:
    #
    #     U_temp[k] = binary serving-satellite vector
    #
    # We use this directly to reconstruct:
    #
    #     S2 -> S5 -> S4 -> ...
    # -------------------------------------------------------------------------

    U_temp = temporal_result.get(
        "U_temp",
        {},
    )

    # -------------------------------------------------------------------------
    # Reconstruct joint serving sequence
    # -------------------------------------------------------------------------

    temporal_serving_sequence = []

    if isinstance(
        U_temp,
        dict,
    ):

        for k in time_slots:

            U_k = U_temp.get(
                k
            )

            if U_k is None:

                temporal_serving_sequence.append(
                    None
                )

                continue

            selected = np.where(
                np.asarray(U_k) > 0.5
            )[0]

            if len(selected) == 1:

                temporal_serving_sequence.append(
                    satellite_name(
                        selected[0]
                    )
                )

            else:

                temporal_serving_sequence.append(
                    None
                )

    else:

        temporal_serving_sequence = [

            None

            for _ in time_slots

        ]

    # -------------------------------------------------------------------------
    # Routing cost
    #
    # IMPORTANT:
    #
    # solve_joint_temporal_optimization()
    # returns this field as "routing_cost".
    # -------------------------------------------------------------------------

    temporal_routing_cost = temporal_result.get(
        "routing_cost",
        np.inf,
    )

    if temporal_routing_cost is None:

        temporal_routing_cost = np.inf

    temporal_routing_cost = float(
        temporal_routing_cost
    )

    # -------------------------------------------------------------------------
    # Calculate handovers directly from the recovered sequence
    # -------------------------------------------------------------------------

    temporal_handover_count = count_handovers(
        temporal_serving_sequence
    )

    # -------------------------------------------------------------------------
    # Handover cost
    # -------------------------------------------------------------------------

    temporal_handover_cost = (

        H
        * temporal_handover_count

    )

    # -------------------------------------------------------------------------
    # IMPORTANT FIX:
    #
    # The joint solver's "objective" may currently be reported as inf,
    # even though:
    #
    #     routing_cost
    #
    # and
    #
    #     handover_cost
    #
    # are valid.
    #
    # Therefore, for comparison, reconstruct the complete objective
    # explicitly.
    # -------------------------------------------------------------------------

    if (
        temporal_feasible
        and np.isfinite(
            temporal_routing_cost
        )
    ):

        temporal_total_cost = (

            temporal_routing_cost
            + temporal_handover_cost

        )

    else:

        temporal_total_cost = np.inf

    # =========================================================================
    # PRINT SERVING SEQUENCES
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "TEMPORAL OPTIMISATION COMPARISON"
    )
    print("=" * 80)

    # =========================================================================
    # METHOD 1 OUTPUT
    # =========================================================================

    print()
    print(
        "Method 1: One-Snapshot Dijkstra"
    )

    print("-" * 80)

    print(
        f"Total routing latency : "
        f"{one_total_latency:.3f}"
    )

    print(
        f"Handover count        : "
        f"{one_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{one_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{one_total_cost:.3f}"
    )

    print(
        "Serving sequence      : "
        + " -> ".join(
            satellite
            if satellite is not None
            else "N/A"
            for satellite in one_serving_sequence
        )
    )

    # =========================================================================
    # METHOD 2 OUTPUT
    # =========================================================================

    print()
    print(
        "Method 2: Sequential Dynamic Programming"
    )

    print("-" * 80)

    print(
        f"Feasible              : "
        f"{dp_feasible}"
    )

    print(
        f"Total routing latency : "
        f"{dp_routing_cost:.3f}"
    )

    print(
        f"Handover count        : "
        f"{dp_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{dp_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{dp_total_cost:.3f}"
    )

    print(
        "Serving sequence      : "
        + " -> ".join(
            satellite
            if satellite is not None
            else "N/A"
            for satellite in dp_serving_sequence
        )
    )

    # =========================================================================
    # METHOD 3 OUTPUT
    # =========================================================================

    print()
    print(
        "Method 3: Joint Temporal Optimisation"
    )

    print("-" * 80)

    print(
        f"Feasible              : "
        f"{temporal_feasible}"
    )

    print(
        f"Total routing latency : "
        f"{temporal_routing_cost:.3f}"
    )

    print(
        f"Handover count        : "
        f"{temporal_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{temporal_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{temporal_total_cost:.3f}"
    )

    print(
        "Serving sequence      : "
        + " -> ".join(
            satellite
            if satellite is not None
            else "N/A"
            for satellite in temporal_serving_sequence
        )
    )

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "SUMMARY TABLE"
    )
    print("=" * 80)

    print(
        f"{'Method':<35}"
        f"{'Routing':>15}"
        f"{'Handovers':>12}"
        f"{'Handover Cost':>17}"
        f"{'Total':>15}"
    )

    print("-" * 94)

    print(
        f"{'One-shot / Dijkstra':<35}"
        f"{one_total_latency:>15.3f}"
        f"{one_handover_count:>12}"
        f"{one_handover_cost:>17.3f}"
        f"{one_total_cost:>15.3f}"
    )

    print(
        f"{'Sequential DP':<35}"
        f"{dp_routing_cost:>15.3f}"
        f"{dp_handover_count:>12}"
        f"{dp_handover_cost:>17.3f}"
        f"{dp_total_cost:>15.3f}"
    )

    print(
        f"{'Joint temporal optimisation':<35}"
        f"{temporal_routing_cost:>15.3f}"
        f"{temporal_handover_count:>12}"
        f"{temporal_handover_cost:>17.3f}"
        f"{temporal_total_cost:>15.3f}"
    )

    print("=" * 80)

    # =========================================================================
    # RETURN COMMON COMPARISON STRUCTURE
    # =========================================================================

    comparison = {

        "method_1_dijkstra": {

            "serving_sequence":
                one_serving_sequence,

            "serving_by_snapshot":
                one_serving,

            "latency_by_snapshot":
                one_latency,

            "total_routing_cost":
                float(
                    one_total_latency
                ),

            "handover_count":
                int(
                    one_handover_count
                ),

            "handover_cost":
                float(
                    one_handover_cost
                ),

            "total_cost":
                float(
                    one_total_cost
                ),
        },

        "method_2_dp": {

            "feasible":
                bool(
                    dp_feasible
                ),

            "serving_sequence":
                dp_serving_sequence,

            "serving_sequence_indices":
                dp_serving_raw,

            "total_routing_cost":
                float(
                    dp_routing_cost
                ),

            "handover_count":
                int(
                    dp_handover_count
                ),

            "handover_cost":
                float(
                    dp_handover_cost
                ),

            "total_cost":
                float(
                    dp_total_cost
                ),
        },

        "method_3_joint_temporal": {

            "feasible":
                bool(
                    temporal_feasible
                ),

            "serving_sequence":
                temporal_serving_sequence,

            "total_routing_cost":
                float(
                    temporal_routing_cost
                ),

            "handover_count":
                int(
                    temporal_handover_count
                ),

            "handover_cost":
                float(
                    temporal_handover_cost
                ),

            "total_cost":
                float(
                    temporal_total_cost
                ),
        },
    }

    return comparison

'''
def compare_one_slot_vs_joint_temporal(
    E,
    L,
    dijkstra_timestamp_results,
    dp_result,
    temporal_result,
    H=8.0,
):
    """
    Compare three temporal optimisation strategies:

    Method 1:
        One-timestamp-at-a-time optimisation using Dijkstra.
        Each snapshot is solved independently.

    Method 2:
        Sequential dynamic programming (DP).
        Per-snapshot candidate routes are generated first and
        then the serving-satellite sequence is optimised jointly
        with a handover penalty.

    Method 3:
        Joint temporal optimisation.
        All snapshots are optimised simultaneously with a
        handover penalty.
    """

    # ------------------------------------------------------------------
    # Method 1: independent one-snapshot Dijkstra optimisation
    # ------------------------------------------------------------------

    one_serving = {}
    one_latency = {}

    for k in sorted(E):

        result = dijkstra_timestamp_results.get(k, {})

        if not result.get("feasible", False):
            one_serving[k] = None
            one_latency[k] = None
            continue

        serving_idx = result.get("serving_satellite")

        one_serving[k] = (
            f"S{int(serving_idx) + 1}"
            if serving_idx is not None
            else None
        )

        objective = result.get("objective")

        one_latency[k] = (
            float(objective)
            if objective is not None and np.isfinite(objective)
            else None
        )

    # ------------------------------------------------------------------
    # Calculate Method 1 handovers
    # ------------------------------------------------------------------

    one_handover_count = 0
    previous_serving = None

    for k in sorted(one_serving):

        current_serving = one_serving[k]

        if current_serving is None:
            continue

        if (
            previous_serving is not None
            and current_serving != previous_serving
        ):
            one_handover_count += 1

        previous_serving = current_serving

    one_total_latency = sum(
        value
        for value in one_latency.values()
        if value is not None
    )

    one_handover_cost = H * one_handover_count

    one_total_cost = (
        one_total_latency
        + one_handover_cost
    )

    # ------------------------------------------------------------------
    # Method 2: Sequential Dynamic Programming
    # ------------------------------------------------------------------

    dp_feasible = dp_result.get(
        "feasible",
        False,
    )

    dp_serving = dp_result.get(
        "serving_sequence",
        [],
    )

    dp_routing_cost = dp_result.get(
        "total_routing_cost",
        np.inf,
    )

    dp_handover_count = dp_result.get(
        "handover_count",
        0,
    )

    dp_handover_cost = dp_result.get(
        "handover_cost",
        H * dp_handover_count,
    )

    dp_total_cost = dp_result.get(
        "total_cost",
        np.inf,
    )

    # ------------------------------------------------------------------
    # Method 3: Joint Temporal Optimisation
    # ------------------------------------------------------------------
    #
    # solve_joint_temporal_optimization() currently returns:
    #
    #   routing_cost
    #   handover_count
    #   handover_cost
    #   objective
    #   U_temp
    #
    # It does NOT return:
    #
    #   total_routing_cost
    #   total_cost
    #   serving_sequence
    #
    # Therefore read the actual returned fields here.

    temporal_feasible = temporal_result.get(
        "feasible",
        False,
    )

    # ------------------------------------------------------------------
    # Reconstruct serving sequence from U_temp
    # ------------------------------------------------------------------

    temporal_serving = temporal_result.get(
        "serving_sequence",
        None,
    )

    if temporal_serving is None:

        temporal_serving = []

        U_temp = temporal_result.get(
            "U_temp",
            {},
        )

        for k in sorted(E):

            if not isinstance(U_temp, dict):
                temporal_serving.append(None)
                continue

            U_k = U_temp.get(k)

            if U_k is None:
                temporal_serving.append(None)
                continue

            selected = np.where(
                np.asarray(U_k) > 0.5
            )[0]

            if len(selected) == 1:

                temporal_serving.append(
                    int(selected[0])
                )

            else:

                temporal_serving.append(None)

    # ------------------------------------------------------------------
    # Read the actual fields returned by the joint solver
    # ------------------------------------------------------------------

    temporal_routing_cost = temporal_result.get(
        "routing_cost",
        np.inf,
    )

    temporal_handover_count = temporal_result.get(
        "handover_count",
        0,
    )

    temporal_handover_cost = temporal_result.get(
        "handover_cost",
        H * temporal_handover_count,
    )

    temporal_total_cost = temporal_result.get(
        "objective",
        np.inf,
    )

    # ------------------------------------------------------------------
    # Common comparison structure
    # ------------------------------------------------------------------

    comparison = {

        "method_1_dijkstra": {

            "serving_sequence": one_serving,

            "latency_by_snapshot": one_latency,

            "total_routing_cost": float(
                one_total_latency
            ),

            "handover_count": int(
                one_handover_count
            ),

            "handover_cost": float(
                one_handover_cost
            ),

            "total_cost": float(
                one_total_cost
            ),
        },

        "method_2_dp": {

            "feasible": bool(
                dp_feasible
            ),

            "serving_sequence": dp_serving,

            "total_routing_cost": float(
                dp_routing_cost
            ),

            "handover_count": int(
                dp_handover_count
            ),

            "handover_cost": float(
                dp_handover_cost
            ),

            "total_cost": float(
                dp_total_cost
            ),
        },

        "method_3_joint_temporal": {

            "feasible": bool(
                temporal_feasible
            ),

            "serving_sequence": temporal_serving,

            "total_routing_cost": float(
                temporal_routing_cost
            ),

            "handover_count": int(
                temporal_handover_count
            ),

            "handover_cost": float(
                temporal_handover_cost
            ),

            "total_cost": float(
                temporal_total_cost
            ),
        },
    }

    # ------------------------------------------------------------------
    # Print comparison
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEMPORAL OPTIMISATION COMPARISON")
    print("=" * 80)

    print("\nMethod 1: One-Snapshot Dijkstra")
    print("-" * 80)

    print(
        f"Total routing latency : "
        f"{one_total_latency:.3f}"
    )

    print(
        f"Handover count        : "
        f"{one_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{one_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{one_total_cost:.3f}"
    )

    print(
        f"Serving sequence      : "
        f"{one_serving}"
    )

    print("\nMethod 2: Sequential Dynamic Programming")
    print("-" * 80)

    print(
        f"Feasible              : "
        f"{dp_feasible}"
    )

    print(
        f"Total routing latency : "
        f"{dp_routing_cost:.3f}"
    )

    print(
        f"Handover count        : "
        f"{dp_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{dp_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{dp_total_cost:.3f}"
    )

    print(
        f"Serving sequence      : "
        f"{dp_serving}"
    )

    print("\nMethod 3: Joint Temporal Optimisation")
    print("-" * 80)

    print(
        f"Feasible              : "
        f"{temporal_feasible}"
    )

    print(
        f"Total routing latency : "
        f"{temporal_routing_cost:.3f}"
    )

    print(
        f"Handover count        : "
        f"{temporal_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{temporal_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{temporal_total_cost:.3f}"
    )

    print(
        f"Serving sequence      : "
        f"{temporal_serving}"
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(
        f"{'Method':<35}"
        f"{'Routing':>15}"
        f"{'Handovers':>12}"
        f"{'Handover Cost':>17}"
        f"{'Total':>15}"
    )

    print("-" * 94)

    print(
        f"{'One-shot / Dijkstra':<35}"
        f"{one_total_latency:>15.3f}"
        f"{one_handover_count:>12}"
        f"{one_handover_cost:>17.3f}"
        f"{one_total_cost:>15.3f}"
    )

    print(
        f"{'Sequential DP':<35}"
        f"{dp_routing_cost:>15.3f}"
        f"{dp_handover_count:>12}"
        f"{dp_handover_cost:>17.3f}"
        f"{dp_total_cost:>15.3f}"
    )

    print(
        f"{'Joint temporal optimisation':<35}"
        f"{temporal_routing_cost:>15.3f}"
        f"{temporal_handover_count:>12}"
        f"{temporal_handover_cost:>17.3f}"
        f"{temporal_total_cost:>15.3f}"
    )

    print("=" * 80)

    return comparison

def compare_one_slot_vs_joint_temporal(
    E,
    L,
    dijkstra_timestamp_results,
    dp_result,
    temporal_result,
    H=8.0,
):
    """
    Compare three temporal optimisation strategies:

    Method 1:
        One-timestamp-at-a-time optimisation using Dijkstra.
        Each snapshot is solved independently.

    Method 2:
        Sequential dynamic programming (DP).
        Per-snapshot candidate routes are generated first and
        then the serving-satellite sequence is optimised jointly
        with a handover penalty.

    Method 3:
        Joint temporal optimisation.
        All snapshots are optimised simultaneously with a
        handover penalty.
    """

    # =========================================================================
    # METHOD 1
    # ONE-TIMESTAMP-AT-A-TIME (DIJKSTRA)
    # =========================================================================

    one_serving = {}
    one_latency = {}

    for k in sorted(E):

        result = dijkstra_timestamp_results.get(
            k,
            {},
        )

        if not result.get("feasible", False):

            one_serving[k] = None
            one_latency[k] = None
            continue

        serving_idx = result.get("serving_satellite")

        one_serving[k] = (
            f"S{int(serving_idx) + 1}"
            if serving_idx is not None
            else None
        )

        # Dijkstra already returns the complete end-to-end
        # one-timestamp routing cost:
        #
        #   Z -> serving satellite -> ... -> gateway
        #
        # Therefore use its objective directly instead of
        # reconstructing latency from U/I/F.

        objective = result.get("objective")

        one_latency[k] = (
            float(objective)
            if objective is not None and np.isfinite(objective)
            else None
        )

    # -------------------------------------------------------------------------
    # Calculate Method 1 handovers
    # -------------------------------------------------------------------------

    one_handover_count = 0

    previous_serving = None

    for k in sorted(one_serving):

        current_serving = one_serving[k]

        if current_serving is None:
            continue

        if previous_serving is not None:
            if current_serving != previous_serving:
                one_handover_count += 1

        previous_serving = current_serving

    one_total_latency = sum(
        value
        for value in one_latency.values()
        if value is not None
    )

    one_handover_cost = H * one_handover_count

    one_total_cost = (
        one_total_latency
        + one_handover_cost
    )

    # =========================================================================
    # METHOD 2
    # SEQUENTIAL DYNAMIC PROGRAMMING
    # =========================================================================

    dp_feasible = dp_result.get(
        "feasible",
        False,
    )

    dp_serving = dp_result.get(
        "serving_sequence",
        [],
    )

    dp_routing_cost = dp_result.get(
        "total_routing_cost",
        np.inf,
    )

    dp_handover_count = dp_result.get(
        "handover_count",
        0,
    )

    dp_handover_cost = dp_result.get(
        "handover_cost",
        H * dp_handover_count,
    )

    dp_total_cost = dp_result.get(
        "total_cost",
        np.inf,
    )

    # =========================================================================
    # METHOD 3
    # JOINT TEMPORAL OPTIMISATION
    # =========================================================================

    temporal_feasible = temporal_result.get(
        "feasible",
        False,
    )

    temporal_serving = temporal_result.get(
        "serving_sequence",
        [],
    )

    temporal_routing_cost = temporal_result.get(
        "total_routing_cost",
        np.inf,
    )

    temporal_handover_count = temporal_result.get(
        "handover_count",
        0,
    )

    temporal_handover_cost = temporal_result.get(
        "handover_cost",
        H * temporal_handover_count,
    )

    temporal_total_cost = temporal_result.get(
        "total_cost",
        np.inf,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    comparison = {

        "method_1_dijkstra": {
            "serving_sequence": one_serving,
            "latency_by_snapshot": one_latency,
            "total_routing_cost": float(one_total_latency),
            "handover_count": int(one_handover_count),
            "handover_cost": float(one_handover_cost),
            "total_cost": float(one_total_cost),
        },

        "method_2_dp": {
            "feasible": bool(dp_feasible),
            "serving_sequence": dp_serving,
            "total_routing_cost": float(dp_routing_cost),
            "handover_count": int(dp_handover_count),
            "handover_cost": float(dp_handover_cost),
            "total_cost": float(dp_total_cost),
        },

        "method_3_joint_temporal": {
            "feasible": bool(temporal_feasible),
            "serving_sequence": temporal_serving,
            "total_routing_cost": float(temporal_routing_cost),
            "handover_count": int(temporal_handover_count),
            "handover_cost": float(temporal_handover_cost),
            "total_cost": float(temporal_total_cost),
        },
    }

    # =========================================================================
    # PRINT COMPARISON
    # =========================================================================

    print("\n" + "=" * 80)
    print("TEMPORAL OPTIMISATION COMPARISON")
    print("=" * 80)

    print("\nMethod 1: One-timestamp-at-a-time (Dijkstra)")
    print("-" * 80)

    print(
        f"Total routing latency : "
        f"{one_total_latency:.3f}"
    )

    print(
        f"Handover count        : "
        f"{one_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{one_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{one_total_cost:.3f}"
    )

    print(
        f"Serving sequence      : "
        f"{one_serving}"
    )

    print("\nMethod 2: Sequential Dynamic Programming")
    print("-" * 80)

    print(
        f"Feasible              : "
        f"{dp_feasible}"
    )

    print(
        f"Total routing latency : "
        f"{dp_routing_cost:.3f}"
    )

    print(
        f"Handover count        : "
        f"{dp_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{dp_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{dp_total_cost:.3f}"
    )

    print(
        f"Serving sequence      : "
        f"{dp_serving}"
    )

    print("\nMethod 3: Joint Temporal Optimisation")
    print("-" * 80)

    print(
        f"Feasible              : "
        f"{temporal_feasible}"
    )

    print(
        f"Total routing latency : "
        f"{temporal_routing_cost:.3f}"
    )

    print(
        f"Handover count        : "
        f"{temporal_handover_count}"
    )

    print(
        f"Handover cost         : "
        f"{temporal_handover_cost:.3f}"
    )

    print(
        f"Total objective       : "
        f"{temporal_total_cost:.3f}"
    )

    print(
        f"Serving sequence      : "
        f"{temporal_serving}"
    )

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(
        f"{'Method':<35}"
        f"{'Routing':>15}"
        f"{'Handovers':>12}"
        f"{'Handover Cost':>17}"
        f"{'Total':>15}"
    )

    print("-" * 94)

    print(
        f"{'One-shot / Dijkstra':<35}"
        f"{one_total_latency:>15.3f}"
        f"{one_handover_count:>12}"
        f"{one_handover_cost:>17.3f}"
        f"{one_total_cost:>15.3f}"
    )

    print(
        f"{'Sequential DP':<35}"
        f"{dp_routing_cost:>15.3f}"
        f"{dp_handover_count:>12}"
        f"{dp_handover_cost:>17.3f}"
        f"{dp_total_cost:>15.3f}"
    )

    print(
        f"{'Joint temporal optimisation':<35}"
        f"{temporal_routing_cost:>15.3f}"
        f"{temporal_handover_count:>12}"
        f"{temporal_handover_cost:>17.3f}"
        f"{temporal_total_cost:>15.3f}"
    )

    print("=" * 80)

    return comparison
'''
# =============================================================================

# DISPLAY STAGE-1 VS STAGE-2 RESULTS

# =============================================================================

def display_stage1_vs_stage2(

    E,

    C,

    L,

    feasibility_results,

    one_timestamp_results,

):

    """

    Display Stage-1 feasible paths and Stage-2 one-timestamp

    optimal path for every time snapshot.

    Stage 1 provides:

        U_fes[k]

        I_fes[k]

        F_fes[k]

    Stage 2 provides:

        U_one[k]

        I_one[k]

        F_one[k]

    Paths are reconstructed directly from the U/I/F matrices.

    No SATELLITES or GATEWAYS configuration variables are required.

    Satellite names are generated from matrix indices:

        index 0 -> S1

        index 1 -> S2

        ...

        index 17 -> S18

    Gateway names are generated from F-matrix columns:

        column 0 -> GW1

        column 1 -> GW2

        ...

    """

    # =========================================================================

    # LOCAL HELPER: SATELLITE NAME

    # =========================================================================

    def satellite_name(index):

        return f"S{index + 1}"

    # =========================================================================

    # LOCAL HELPER: GATEWAY NAME

    # =========================================================================

    def gateway_name(index):

        return f"GW{index + 1}"

    # =========================================================================

    # LOCAL HELPER: PATH LATENCY

    # =========================================================================

    def path_latency(k, path):

        """

        Calculate the total latency of a reconstructed path.

        Path structure:

            z -> S_i -> S_j -> ... -> GW_g

        Latency components are obtained from:

            L[k]["US"]

            L[k]["ISL"]

            L[k]["F"]

        Units: ms

        """

        latency = 0.0

        for u, v in zip(

            path[:-1],

            path[1:],

        ):

            # -----------------------------------------------------------------

            # SOURCE -> SATELLITE

            # -----------------------------------------------------------------

            if u == "z":

                sat_idx = int(

                    v[1:]

                ) - 1

                latency += L[k]["US"][

                    sat_idx

                ]

            # -----------------------------------------------------------------

            # SATELLITE -> SATELLITE

            # -----------------------------------------------------------------

            elif (

                u.startswith("S")

                and v.startswith("S")

            ):

                i = int(

                    u[1:]

                ) - 1

                j = int(

                    v[1:]

                ) - 1

                latency += L[k]["ISL"][

                    i,

                    j,

                ]

            # -----------------------------------------------------------------

            # SATELLITE -> GATEWAY

            # -----------------------------------------------------------------

            elif (

                u.startswith("S")

                and v.startswith("GW")

            ):

                i = int(

                    u[1:]

                ) - 1

                g = int(

                    v[2:]

                ) - 1

                latency += L[k]["F"][

                    i,

                    g,

                ]

        return float(latency)

    # =========================================================================

    # LOCAL HELPER: RECONSTRUCT PATHS FROM U/I/F

    # =========================================================================

    def reconstruct_paths(

        U,

        I,

        F,

    ):

        """

        Reconstruct source-to-gateway paths from selected U/I/F variables.

        U:

            source -> satellite selection

        I:

            satellite -> satellite ISL selection

        F:

            satellite -> gateway feeder selection

        Assumes the Stage-1 solution is non-branching:

            one selected outgoing ISL

            OR

            one selected feeder

        for every active satellite.

        """

        U = np.asarray(

            U,

            dtype=int,

        )

        I = np.asarray(

            I,

            dtype=int,

        )

        F = np.asarray(

            F,

            dtype=int,

        )

        # ---------------------------------------------------------------------

        # Selected service satellites

        # ---------------------------------------------------------------------

        service_indices = np.where(

            U > 0.5

        )[0]

        paths = []

        # ---------------------------------------------------------------------

        # Reconstruct one path for every selected service satellite

        # ---------------------------------------------------------------------

        for service_idx in service_indices:

            current = int(

                service_idx

            )

            path = [

                "z",

                satellite_name(current),

            ]

            visited = {

                current,

            }

            valid = True

            # =================================================================

            # FOLLOW THE SELECTED ROUTE

            # =================================================================

            while True:

                # -------------------------------------------------------------

                # Check whether the current satellite has a selected feeder.

                # -------------------------------------------------------------

                gateway_indices = np.where(

                    F[current, :] > 0.5

                )[0]

                if len(gateway_indices) == 1:

                    gateway_idx = int(

                        gateway_indices[0]

                    )

                    path.append(

                        gateway_name(

                            gateway_idx

                        )

                    )

                    break

                # -------------------------------------------------------------

                # Otherwise the current satellite must have exactly one

                # selected outgoing ISL.

                # -------------------------------------------------------------

                next_indices = np.where(

                    I[current, :] > 0.5

                )[0]

                if len(next_indices) != 1:

                    valid = False

                    break

                next_idx = int(

                    next_indices[0]

                )

                # -------------------------------------------------------------

                # Prevent cycles.

                # -------------------------------------------------------------

                if next_idx in visited:

                    valid = False

                    break

                visited.add(

                    next_idx

                )

                path.append(

                    satellite_name(

                        next_idx

                    )

                )

                current = next_idx

            # -----------------------------------------------------------------

            # Store only valid paths.

            # -----------------------------------------------------------------

            if valid:

                paths.append(

                    path

                )

        return paths

    # =========================================================================

    # HEADER

    # =========================================================================

    print()

    print("=" * 80)

    print(

        "STAGE-1 FEASIBLE PATHS "

        "vs STAGE-2 ONE-SHOT OPTIMAL PATH"

    )

    print("=" * 80)

    # =========================================================================

    # LOOP THROUGH ALL TIME SNAPSHOTS

    # =========================================================================

    for k in sorted(E):

        print()

        print(

            f"TIME SNAPSHOT k = {k}"

        )

        print("-" * 80)

        # =====================================================================

        # STAGE 1: FEASIBLE PATHS

        # =====================================================================

        print()

        print(

            "STAGE-1 FEASIBLE PATHS"

        )

        print("-" * 80)

        result_fes = feasibility_results.get(

            k,

            {},

        )

        # ---------------------------------------------------------------------

        # Retrieve Stage-1 variables.

        #

        # Prefer the explicit Stage-1 names.

        # The fallback supports the older result dictionary.

        # ---------------------------------------------------------------------

        U_fes = result_fes.get(

            "U_fes",

            result_fes.get("U"),

        )

        I_fes = result_fes.get(

            "I_fes",

            result_fes.get("I"),

        )

        F_fes = result_fes.get(

            "F_fes",

            result_fes.get("F"),

        )

        # ---------------------------------------------------------------------

        # Check Stage-1 result.

        # ---------------------------------------------------------------------

        if (

            U_fes is None

            or I_fes is None

            or F_fes is None

        ):

            print(

                "No Stage-1 feasibility solution."

            )

        else:

            # -----------------------------------------------------------------

            # Reconstruct Stage-1 paths.

            # -----------------------------------------------------------------

            stage1_paths = reconstruct_paths(

                U_fes,

                I_fes,

                F_fes,

            )

            # -----------------------------------------------------------------

            # Display Stage-1 paths.

            # -----------------------------------------------------------------

            if stage1_paths:

                for path_idx, path in enumerate(

                    stage1_paths,

                    start=1,

                ):

                    latency = path_latency(

                        k,

                        path,

                    )

                    print(

                        f"Path {path_idx}: "

                        + " -> ".join(path)

                    )

                    print(

                        f"         Latency: "

                        f"{latency:.3f} ms"

                    )

            else:

                print(

                    "No valid Stage-1 paths reconstructed."

                )

        # =====================================================================

        # STAGE 2: ONE-TIMESTAMP OPTIMAL PATH

        # =====================================================================

        print()

        print(

            "STAGE-2 ONE-TIMESTAMP OPTIMAL PATH"

        )

        print("-" * 80)

        result_one = one_timestamp_results.get(

            k,

            {},

        )

        # ---------------------------------------------------------------------

        # Retrieve Stage-2 variables.

        # ---------------------------------------------------------------------

        U_one = result_one.get(

            "U_one"

        )

        I_one = result_one.get(

            "I_one"

        )

        F_one = result_one.get(

            "F_one"

        )

        # ---------------------------------------------------------------------

        # Check Stage-2 solution.

        # ---------------------------------------------------------------------

        if (

            U_one is None

            or I_one is None

            or F_one is None

        ):

            print(

                "No Stage-2 solution."

            )

            print()

            print(

                f"Solver status: "

                f"{result_one.get('status', 'N/A')}"

            )

            print(

                f"Feasible: "

                f"{result_one.get('feasible', False)}"

            )

            continue

        # ---------------------------------------------------------------------

        # Reconstruct Stage-2 path.

        # ---------------------------------------------------------------------

        stage2_paths = reconstruct_paths(

            U_one,

            I_one,

            F_one,

        )

        # ---------------------------------------------------------------------

        # A one-timestamp solution must contain exactly one route.

        # ---------------------------------------------------------------------

        if len(stage2_paths) == 1:

            optimal_path = stage2_paths[0]

            optimal_latency = path_latency(

                k,

                optimal_path,

            )

            print(

                "Optimal path: "

                + " -> ".join(optimal_path)

            )

            print(

                f"Latency: "

                f"{optimal_latency:.3f} ms"

            )

        elif len(stage2_paths) == 0:

            print(

                "Unable to reconstruct a valid "

                "Stage-2 path."

            )

        else:

            print(

                "Invalid Stage-2 solution: "

                f"{len(stage2_paths)} paths reconstructed."

            )

        # =====================================================================

        # STAGE-2 SOLVER INFORMATION

        # =====================================================================

        print()

        print(

            f"Solver status: "

            f"{result_one.get('status', 'N/A')}"

        )

        print()

def run_one_timestamp_optimization(
    E,
    C,
    L,
    feasibility_results,
    demand=350.0,
):
    """
    Run the independent one-timestamp optimization for every time snapshot.

    Stage 1 provides:
        U_fes[k]
        I_fes[k]
        F_fes[k]

    Stage 2 produces:
        U_one[k]
        I_one[k]
        F_one[k]

    There is no temporal coupling in this function.
    Each timestamp is optimized independently.
    """

    U_one = {}
    I_one = {}
    F_one = {}

    one_timestamp_results = {}

    print()
    print("=" * 72)
    print("STAGE-2 ONE-TIMESTAMP OPTIMIZATION")
    print("=" * 72)

    for k in sorted(E):

        print()
        print(f"TIME SNAPSHOT k = {k}")
        print("-" * 72)

        # =====================================================================
        # RETRIEVE STAGE-1 FEASIBILITY SOLUTION
        # =====================================================================

        result = feasibility_results.get(k, {})

        U_fes = result.get("U")
        I_fes = result.get("I")
        F_fes = result.get("F")

        # =====================================================================
        # CHECK STAGE-1 FEASIBILITY
        # =====================================================================

        if U_fes is None or I_fes is None or F_fes is None:

            print("Stage-1 feasibility : NOT FOUND")
            print("Stage-2 optimization: SKIPPED")

            U_one[k] = None
            I_one[k] = None
            F_one[k] = None

            one_timestamp_results[k] = {
                "U_one": None,
                "I_one": None,
                "F_one": None,
                "status": "NO_STAGE1_FEASIBILITY",
            }

            continue

        print("Stage-1 feasibility : FOUND")

        # =====================================================================
        # STAGE-2 ONE-TIMESTAMP OPTIMIZATION
        # =====================================================================

        #try:

        OneTimeStamp_result = solve_one_timestamp_route(

            # Stage-1 feasible connectivity
            U_fes=U_fes,
            I_fes=I_fes,
            F_fes=F_fes,

            # Capacity data
            C_US=C[k]["US"],
            C_ISL=C[k]["ISL"],
            C_F=C[k]["F"],

            # Demand
            demand=demand,

            # Latency data
            L_US=L[k]["US"],
            L_ISL=L[k]["ISL"],
            L_F=L[k]["F"],
        )
        U_one_k = OneTimeStamp_result["U_one"]
        I_one_k = OneTimeStamp_result["I_one"]
        F_one_k = OneTimeStamp_result["F_one"]

    
        # =================================================================
        # STORE STAGE-2 RESULT
        # =================================================================

        U_one[k] = U_one_k
        I_one[k] = I_one_k
        F_one[k] = F_one_k

        one_timestamp_results[k] = {
            "U_one": U_one_k,
            "I_one": I_one_k,
            "F_one": F_one_k,
            "status": "OPTIMAL",
        }

        print("Stage-2 optimization: SOLVED")

        '''except Exception as exc:

            print("Stage-2 optimization: FAILED")
            print(f"Reason: {exc}")

            U_one[k] = None
            I_one[k] = None
            F_one[k] = None

            one_timestamp_results[k] = {
                "U_one": None,
                "I_one": None,
                "F_one": None,
                "status": "FAILED",
                "error": str(exc),
            }'''

    return (
        U_one,
        I_one,
        F_one,
        one_timestamp_results,
    )

def main():

    print("=" * 72)
    print("MEO ONE-SHOT TEMPORAL OPTIMIZATION")
    print("=" * 72)
    print()

    print("Building time-dependent MEO network...")
    print()

    # =========================================================================
    # STAGE 1: BUILD / LOAD TIME-DEPENDENT NETWORK AND FEASIBILITY
    # =========================================================================
    #
    # build_network() performs:
    #
    #   1. Geometry-dependent physical availability generation
    #   2. Stage-1 feasibility optimization
    #   3. Optional storage/retrieval of Stage-1 feasibility results
    #
    # The important Stage-1 outputs are:
    #   U_fes[k]
    #   I_fes[k]
    #   F_fes[k]
    # for every timestamp k.
    # =========================================================================
    # ``build_network`` returns the network data and Stage-1 results.
    network, feasibility_results = build_network(
        reuse_stored_feasibility=REUSE_STORED_FEASIBILITY,
    )

    E = network["E"]
    C = network["C"]
    L = network["L"]


    # =========================================================================
    # STAGE 2: ONE-TIMESTAMP OPTIMIZATION
    # =========================================================================

    '''U_one, I_one, F_one, one_timestamp_results = (
        run_one_timestamp_optimization(
            E=E,
            C=C,
            L=L,
            feasibility_results=feasibility_results,
            demand=350.0,
        )
    )'''

    # =========================================================================
    # STAGE 2A: ONE-TIMESTAMP OPTIMIZATION USING DIJKSTRA
    # =========================================================================

    dijkstra_timestamp_results = {}

    print()
    print("=" * 80)
    print("STAGE-2A ONE-TIMESTAMP OPTIMIZATION USING DIJKSTRA")
    print("=" * 80)

    for k in sorted(E):

        result = feasibility_results.get(k, {})

        # Stage-1 feasible topology
        U_fes = result.get("U")
        I_fes = result.get("I")
        F_fes = result.get("F")

        if (
            U_fes is None
            or I_fes is None
            or F_fes is None
        ):

            dijkstra_timestamp_results[k] = {
                "feasible": False,
                "status": "NO_STAGE1_FEASIBILITY",
                "U_one": None,
                "I_one": None,
                "F_one": None,
            }

            print(
                f"k={k:2d}: "
                "NO STAGE-1 FEASIBILITY"
            )

            continue

        # ---------------------------------------------------------------------
        # Run Dijkstra
        # ---------------------------------------------------------------------

        dijkstra_result = solve_one_timestamp_dijkstra(

            U_fes=U_fes,

            I_fes=I_fes,

            F_fes=F_fes,

            L_US=L[k]["US"],

            L_ISL=L[k]["ISL"],

            L_F=L[k]["F"],

            C_US=C[k]["US"],

            C_ISL=C[k]["ISL"],

            C_F=C[k]["F"],

            demand=350.0,
        )

        # Store result
        dijkstra_timestamp_results[k] = dijkstra_result

        # ---------------------------------------------------------------------
        # Print result
        # ---------------------------------------------------------------------

        if dijkstra_result["feasible"]:

            print(
                f"k={k:2d}: "
                f"{' -> '.join(dijkstra_result['path'])} "
                f"cost={dijkstra_result['objective']:.3f}"
            )

        else:

            print(
                f"k={k:2d}: INFEASIBLE"
            )
    # =========================================================================
    # DISPLAY STAGE 1 VS STAGE 2
    # =========================================================================

    '''display_stage1_vs_stage2(
        E=E,
        C=C,
        L=L,
        feasibility_results=(
            feasibility_results
        ),
        one_timestamp_results=(
            one_timestamp_results
        ),
    )'''

    # =========================================================================
    # JOINT TEMPORAL OPTIMIZATION
    # =========================================================================

    temporal_result = run_joint_temporal_optimization(
        E=E,
        C=C,
        L=L,
        feasibility_results=feasibility_results,
        demand=350.0,
        H=8.0,
    )

    # =========================================================================
    # COMPARE ONE-SLOT VS JOINT TEMPORAL OPTIMIZATION
    # =========================================================================

    '''comparison_result = compare_one_slot_vs_joint(
        E=E,
        L=L,
        one_timestamp_results=one_timestamp_results,
        temporal_result=temporal_result,
        H=8.0,
    )'''

    # =============================================================================
    # STAGE 3: SEQUENTIAL TEMPORAL OPTIMIZATION USING DYNAMIC PROGRAMMING
    # =============================================================================

    candidate_costs, candidate_routes = build_temporal_candidates(
        E=E,
        C=C,
        L=L,
        feasibility_results=feasibility_results,
        demand=350.0,
        solver=None,
    )
    
    dp_result = solve_sequential_dp(
        candidate_costs=candidate_costs,
        candidate_routes=candidate_routes,
        H=8.0,
    )

    if dp_result["feasible"]:

        print("\n" + "=" * 80)
        print("SEQUENTIAL DP TEMPORAL OPTIMIZATION")
        print("=" * 80)

        print(
            f"Total routing cost : "
            f"{dp_result['total_routing_cost']:.3f}"
        )

        print(
            f"Handover count     : "
            f"{dp_result['handover_count']}"
        )

        print(
            f"Handover cost      : "
            f"{dp_result['handover_cost']:.3f}"
        )

        print(
            f"Total cost         : "
            f"{dp_result['total_cost']:.3f}"
        )

        print("\nServing satellite sequence:")

        for k, satellite_index in dp_result["serving_sequence"].items():

            print(
                f"  k={k:2d} : "
                f"S{satellite_index + 1}"
            )

    else:

        print("\nSequential DP optimization is infeasible.")

        print(
            "Reason:",
            dp_result.get("reason", "unknown"),
    )
    comparison = compare_one_slot_vs_joint_temporal(
    E=E,
    L=L,
    dijkstra_timestamp_results=dijkstra_timestamp_results,
    dp_result=dp_result,
    temporal_result=temporal_result,
    H=8.0,
    )
    

# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()