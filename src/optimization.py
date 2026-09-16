"""
MEO One-Shot Temporal Optimization
===================================

Optimization module containing the independent optimization stages.

Stage 1
-------
Geometry-derived physical availability:

    U_bar, I_bar, F_bar

        |
        v

Feasibility optimization:

    U_fes, I_fes, F_fes

The feasibility stage determines whether the geometry-supported
network can provide the required number of demand-feasible
source-to-gateway paths.

Stage 2
-------
For each individual time snapshot:

    U_fes, I_fes, F_fes

        |
        v

One-timestamp shortest-route optimization:

    U_one, I_one, F_one

The Stage-2 optimization selects exactly ONE source-to-gateway
route for the given timestamp.

Stage 3
-------
The temporal optimization is intentionally NOT mixed into the
Stage-1 or Stage-2 formulations.

It will operate separately on the per-timestamp candidate
solutions and introduce the handover decision.
"""

import cvxpy as cp
import numpy as np
from typing import cast


# =============================================================================
# MIXED-INTEGER SOLVER SELECTION
# =============================================================================


def _select_milp_solver():
    """
    Select an installed CVXPY mixed-integer solver.

    Preferred order:

        1. SCIP
        2. GLPK_MI
        3. ECOS_BB

    Returns
    -------
    solver
        CVXPY solver constant.

    Raises
    ------
    RuntimeError
        If no suitable mixed-integer solver is installed.
    """

    installed = cp.installed_solvers()

    if "SCIP" in installed:
        return cp.SCIP

    if "GLPK_MI" in installed:
        return cp.GLPK_MI

    if "ECOS_BB" in installed:
        return cp.ECOS_BB

    raise RuntimeError(
        "No suitable mixed-integer CVXPY solver found. "
        f"Installed solvers: {installed}. "
        "Install SCIP, GLPK_MI, or ECOS_BB."
    )


# =============================================================================
# STAGE 1
# END-TO-END CONNECTIVITY FEASIBILITY OPTIMIZATION
# =============================================================================


def solve_connectivity_feasibility(
    U_bar,
    I_bar,
    F_bar,
    C_US,
    C_ISL,
    C_F,
    demand=350.0,
    num_paths=1,
    min_isl_hops=1,
    max_isl_hops=6,
    solver=None,
    L_US=None,
    L_ISL=None,
    L_F=None,
    fixed_serving_satellite=None,
):
    """
    Stage 1: End-to-end connectivity feasibility optimization.

    Geometry provides:

        U_bar[i]
        I_bar[i,j]
        F_bar[i,g]

    The optimization selects:

        U[i]
        I[i,j]
        F[i,g]

    subject to:

        U <= U_bar
        I <= I_bar
        F <= F_bar

    demand-capacity constraints,

    source-to-gateway flow conservation,

    non-branching path structure,

    and minimum/maximum ISL-hop constraints.

    The objective minimizes total selected-link latency.

    Parameters
    ----------
    U_bar : ndarray, shape (S,)
        Geometry-derived source-to-satellite availability.

    I_bar : ndarray, shape (S,S)
        Geometry-derived ISL availability.

    F_bar : ndarray, shape (S,G)
        Geometry-derived satellite-to-gateway availability.

    C_US : ndarray, shape (S,)
        Source-to-satellite capacities in Mbps.

    C_ISL : ndarray, shape (S,S)
        ISL capacities in Mbps.

    C_F : ndarray, shape (S,G)
        Feeder-link capacities in Mbps.

    demand : float
        Required traffic demand in Mbps.

    num_paths : int
        Number of non-branching source-to-gateway paths required.

    min_isl_hops : int
        Minimum number of ISLs in each selected path.

    max_isl_hops : int
        Maximum number of ISLs in each selected path.

    solver : optional
        CVXPY mixed-integer solver.

    L_US : ndarray, shape (S,)
        Source-to-satellite latency.

    L_ISL : ndarray, shape (S,S)
        ISL latency.

    L_F : ndarray, shape (S,G)
        Feeder-link latency.

    Returns
    -------
    result : dict
        Stage-1 optimization result.
    """

    # =========================================================================
    # 1. CONVERT INPUTS
    # =========================================================================

    U_bar = np.asarray(U_bar, dtype=float)
    I_bar = np.asarray(I_bar, dtype=float)
    F_bar = np.asarray(F_bar, dtype=float)

    C_US = np.asarray(C_US, dtype=float)
    C_ISL = np.asarray(C_ISL, dtype=float)
    C_F = np.asarray(C_F, dtype=float)

    if L_US is None or L_ISL is None or L_F is None:
        raise ValueError(
            "L_US, L_ISL and L_F must be provided "
            "for latency minimization."
        )

    L_US = np.asarray(L_US, dtype=float)
    L_ISL = np.asarray(L_ISL, dtype=float)
    L_F = np.asarray(L_F, dtype=float)

    # =========================================================================
    # 2. DIMENSIONS
    # =========================================================================

    if U_bar.ndim != 1:
        raise ValueError(
            f"U_bar must be one-dimensional. "
            f"Received shape={U_bar.shape}."
        )

    S = U_bar.shape[0]

    if I_bar.shape != (S, S):
        raise ValueError(
            f"I_bar must have shape {(S, S)}. "
            f"Received shape={I_bar.shape}."
        )

    if F_bar.ndim != 2:
        raise ValueError(
            f"F_bar must be two-dimensional. "
            f"Received shape={F_bar.shape}."
        )

    G = F_bar.shape[1]

    if C_US.shape != (S,):
        raise ValueError(
            f"C_US must have shape {(S,)}. "
            f"Received shape={C_US.shape}."
        )

    if C_ISL.shape != (S, S):
        raise ValueError(
            f"C_ISL must have shape {(S, S)}. "
            f"Received shape={C_ISL.shape}."
        )

    if C_F.shape != (S, G):
        raise ValueError(
            f"C_F must have shape {(S, G)}. "
            f"Received shape={C_F.shape}."
        )

    if L_US.shape != (S,):
        raise ValueError(
            f"L_US must have shape {(S,)}. "
            f"Received shape={L_US.shape}."
        )

    if L_ISL.shape != (S, S):
        raise ValueError(
            f"L_ISL must have shape {(S, S)}. "
            f"Received shape={L_ISL.shape}."
        )

    if L_F.shape != (S, G):
        raise ValueError(
            f"L_F must have shape {(S, G)}. "
            f"Received shape={L_F.shape}."
        )

    # =========================================================================
    # 3. PARAMETER VALIDATION
    # =========================================================================

    if not isinstance(num_paths, (int, np.integer)):
        raise ValueError(
            "num_paths must be an integer."
        )

    if num_paths < 1:
        raise ValueError(
            "num_paths must be >= 1."
        )

    if num_paths > S:
        raise ValueError(
            f"num_paths={num_paths} cannot exceed "
            f"the number of satellites S={S}."
        )

    if min_isl_hops < 0:
        raise ValueError(
            "min_isl_hops must be >= 0."
        )

    if max_isl_hops < min_isl_hops:
        raise ValueError(
            "max_isl_hops must be >= min_isl_hops."
        )

    if max_isl_hops > S - 1:
        raise ValueError(
            f"max_isl_hops={max_isl_hops} cannot exceed "
            f"S-1={S - 1} for a simple path."
        )

    # =========================================================================
    # 4. DECISION VARIABLES
    # =========================================================================
    #
    # U[i] = 1:
    #     source -> satellite i is selected.
    #
    # I[i,j] = 1:
    #     satellite i -> satellite j ISL is selected.
    #
    # F[i,g] = 1:
    #     satellite i -> gateway g feeder is selected.
    #
    # H[i]:
    #     ISL-hop position of satellite i.
    #
    # For a selected service satellite:
    #
    #     H[i] = 0
    #
    # For every selected ISL i -> j:
    #
    #     H[j] = H[i] + 1
    # =========================================================================

    U = cp.Variable(
        S,
        boolean=True,
    )

    I = cp.Variable(
        (S, S),
        boolean=True,
    )

    F = cp.Variable(
        (S, G),
        boolean=True,
    )

    H = cp.Variable(
        S,
        integer=True,
    )

    constraints = []

    # =========================================================================
    # 5. NO ISL SELF-LOOPS
    # =========================================================================

    for i in range(S):

        constraints.append(
            I[i, i] == 0
        )

    # =========================================================================
    # 6. NUMBER OF PATHS
    # =========================================================================
    #
    # Exactly num_paths service links:
    #
    #     sum_i U_i = num_paths
    #
    # Exactly num_paths feeder links:
    #
    #     sum_i,g F_ig = num_paths
    # =========================================================================

    constraints.append(
        cp.sum(U) == num_paths
    )

    constraints.append(
        cp.sum(F) == num_paths
    )

    # =========================================================================
    # 7. FLOW CONSERVATION + NON-BRANCHING
    # =========================================================================
    #
    # For every satellite i:
    #
    #     U_i + sum_j I_ji
    #
    #         =
    #
    #     sum_j I_ij + sum_g F_ig
    #
    # This represents flow entering and leaving the satellite.
    #
    # Non-branching constraints:
    #
    #     U_i + sum_j I_ji <= 1
    #
    #     sum_j I_ij + sum_g F_ig <= 1
    #
    # Therefore every selected satellite belongs to at most one
    # source-to-gateway route.
    # =========================================================================

    for i in range(S):

        incoming = (
            U[i]
            +
            cp.sum(I[:, i])
        )

        outgoing = (
            cp.sum(I[i, :])
            +
            cp.sum(F[i, :])
        )

        # Flow conservation
        constraints.append(
            incoming == outgoing
        )

        # At most one incoming connection
        constraints.append(
            incoming <= num_paths
        )

        # At most one outgoing connection
        constraints.append(
            outgoing <= num_paths
        )

        # Every selected service satellite must continue through
        # at least one ISL before reaching a gateway.
        constraints.append(
            U[i] <= cp.sum(I[i, :])
        )

    # =========================================================================
    # 8. HOP-POSITION CONSTRAINTS
    # =========================================================================
    #
    # H[i] is meaningful only for selected satellites.
    #
    # Service satellite:
    #
    #     U_i = 1 -> H_i = 0
    #
    # Feeder terminal:
    #
    #     F_ig = 1 -> min_isl_hops <= H_i <= max_isl_hops
    #
    # Selected ISL:
    #
    #     I_ij = 1 -> H_j = H_i + 1
    #
    # Since every route is non-branching, H defines the ordering
    # of satellites along each selected route.
    # =========================================================================

    M = S + 1

    constraints.append(
        H >= 0
    )

    constraints.append(
        H <= max_isl_hops
    )

    for i in range(S):

        # ---------------------------------------------------------------------
        # Selected service satellite has hop position zero.
        # ---------------------------------------------------------------------

        constraints.append(
            H[i] <= M * (1 - U[i])
        )

        # ---------------------------------------------------------------------
        # Feeder terminal defines the final hop position.
        # ---------------------------------------------------------------------

        feeder_selected = cp.sum(F[i, :])

        constraints.append(
            H[i] >= (
                min_isl_hops
                * feeder_selected
            )
        )

        constraints.append(
            H[i] <= (
                max_isl_hops * feeder_selected
                +
                M * (1 - feeder_selected)
            )
        )

        # ---------------------------------------------------------------------
        # Every selected ISL advances the hop position by exactly one.
        # ---------------------------------------------------------------------

        for j in range(S):

            if i == j:
                continue

            constraints.append(
                H[j]
                >=
                H[i] + 1
                -
                M * (1 - I[i, j])
            )

            constraints.append(
                H[j]
                <=
                H[i] + 1
                +
                M * (1 - I[i, j])
            )

    # =========================================================================
    # 9. PHYSICAL AVAILABILITY
    # =========================================================================
    #
    # Selected routing links must exist physically:
    #
    #     U <= U_bar
    #     I <= I_bar
    #     F <= F_bar
    # =========================================================================

    constraints.append(
        U <= U_bar
    )

    constraints.append(
        I <= I_bar
    )

    constraints.append(
        F <= F_bar
    )

    # =========================================================================
    # 10. DEMAND-CAPACITY CONSTRAINTS
    # =========================================================================
    #
    # Because U/I/F are binary:
    #
    #     demand * U_i <= C_US_i
    #
    #     demand * I_ij <= C_ISL_ij
    #
    #     demand * F_ig <= C_F_ig
    #
    # means every selected link must support the complete demand.
    # =========================================================================

    constraints.append(
        demand * U <= C_US
    )

    constraints.append(
        demand * I <= C_ISL
    )

    constraints.append(
        demand * F <= C_F
    )

    # =========================================================================
    # 11. Fixed serving satellite enforcing
    # =========================================================================
    if fixed_serving_satellite is not None:
        constraints.append(
            U[fixed_serving_satellite] == 1
        )

    # =========================================================================
    # 11. LATENCY OBJECTIVE
    # =========================================================================
    #
    # Minimize total latency of all selected links:
    #
    # min
    #
    #     sum_i     L_US_i U_i
    #
    #   + sum_i,j   L_ISL_ij I_ij
    #
    #   + sum_i,g   L_F_ig F_ig
    # =========================================================================

    objective = cp.Minimize(
        cp.sum(
            cp.multiply(
                L_US,
                U,
            )
        )
        +
        cp.sum(
            cp.multiply(
                L_ISL,
                I,
            )
        )
        +
        cp.sum(
            cp.multiply(
                L_F,
                F,
            )
        )
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    # =========================================================================
    # 12. SOLVER
    # =========================================================================

    if solver is None:
        solver = _select_milp_solver()

    problem.solve(
        solver=solver,
        verbose=False,
    )

    # =========================================================================
    # 13. STATUS
    # =========================================================================

    feasible = problem.status in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    )

    result = {
        "status": problem.status,
        "feasible": feasible,
        "objective_value": None,

        "U": None,
        "I": None,
        "F": None,
        "H": None,

        "min_isl_hops": min_isl_hops,
        "max_isl_hops": max_isl_hops,

        "U_available": np.array(
            U_bar,
            copy=True,
        ),

        "I_available": np.array(
            I_bar,
            copy=True,
        ),

        "F_available": np.array(
            F_bar,
            copy=True,
        ),
    }

    # =========================================================================
    # 14. RETURN IF INFEASIBLE
    # =========================================================================

    if not feasible:
        return result

    # =========================================================================
    # 15. EXTRACT SOLUTION
    # =========================================================================

    if (
        U.value is None
        or I.value is None
        or F.value is None
        or H.value is None
    ):
        result["feasible"] = False
        result["status"] = "NO_SOLUTION_VALUES"
        return result

    result["U"] = np.rint(
        U.value
    ).astype(int)

    result["I"] = np.rint(
        I.value
    ).astype(int)

    result["F"] = np.rint(
        F.value
    ).astype(int)

    result["H"] = np.rint(
        H.value
    ).astype(int)

    if problem.value is not None:
        result["objective_value"] = float(
            cast(float, problem.value)
        )

    return result


# =============================================================================
# ROUTE RECONSTRUCTION
# =============================================================================


def reconstruct_route(
    result,
    satellites=None,
    gateways=None,
    num_paths=None,
):
    """
    Reconstruct ordered source-to-gateway routes from Stage-1 U/I/F values.

    Route order is determined entirely by the selected directed ISLs.

    Satellite numbering is NOT used to determine route order.

    Example:

        Z -> S1 -> S6 -> S12 -> S18 -> GW1

    Parameters
    ----------
    result : dict
        Result returned by solve_connectivity_feasibility().

    satellites : list, optional
        Satellite names.

    gateways : list, optional
        Gateway names.

    num_paths : int, optional
        Expected number of routes.

    Returns
    -------
    list or None
        Ordered source-to-gateway routes.
    """

    # =========================================================================
    # 1. FEASIBILITY CHECK
    # =========================================================================

    if not result.get(
        "feasible",
        False,
    ):
        return None

    U = result["U"]
    I = result["I"]
    F = result["F"]

    min_isl_hops = result.get(
        "min_isl_hops",
        0,
    )

    max_isl_hops = result.get(
        "max_isl_hops",
        len(U) - 1,
    )

    # =========================================================================
    # 2. DIMENSIONS
    # =========================================================================

    S = len(U)
    G = F.shape[1]

    # =========================================================================
    # 3. DEFAULT NAMES
    # =========================================================================

    if satellites is None:
        satellites = [
            f"S{i + 1}"
            for i in range(S)
        ]

    if gateways is None:
        gateways = [
            f"GW{i + 1}"
            for i in range(G)
        ]

    # =========================================================================
    # 4. SERVING SATELLITES
    # =========================================================================

    serving = np.where(
        U == 1
    )[0]

    if num_paths is None:
        num_paths = len(serving)

    if len(serving) != num_paths:
        raise RuntimeError(
            f"Expected {num_paths} serving satellites, "
            f"but found {len(serving)}."
        )

    # =========================================================================
    # 5. SELECTED EDGE MAPS
    # =========================================================================

    isl_successors = {
        i: list(
            np.where(
                I[i, :] == 1
            )[0]
        )
        for i in range(S)
    }

    feeder_gateways = {
        i: list(
            np.where(
                F[i, :] == 1
            )[0]
        )
        for i in range(S)
    }

    # =========================================================================
    # 6. RECONSTRUCT EACH ROUTE
    # =========================================================================

    routes = []

    for start in serving:

        start = int(start)

        route = [
            "Z",
            satellites[start],
        ]

        visited = {
            start,
        }

        current = start
        isl_hops = 0

        while True:

            # =================================================================
            # FEEDER TERMINATION
            # =================================================================

            selected_gateways = feeder_gateways[current]

            if len(selected_gateways) == 1:

                gateway = int(
                    selected_gateways[0]
                )

                route.append(
                    gateways[gateway]
                )

                break

            if len(selected_gateways) > 1:

                raise RuntimeError(
                    f"Multiple feeder links selected from "
                    f"{satellites[current]}."
                )

            # =================================================================
            # ISL SUCCESSOR
            # =================================================================

            next_satellites = [
                j
                for j in isl_successors[current]
                if j != current
            ]

            if len(next_satellites) == 0:

                raise RuntimeError(
                    f"No outgoing ISL or feeder from "
                    f"{satellites[current]} while reconstructing "
                    f"route starting from {satellites[start]}."
                )

            if len(next_satellites) > 1:

                raise RuntimeError(
                    f"Ambiguous branching: multiple outgoing ISLs "
                    f"from {satellites[current]}."
                )

            next_satellite = int(
                next_satellites[0]
            )

            # =================================================================
            # CYCLE DETECTION
            # =================================================================

            if next_satellite in visited:

                raise RuntimeError(
                    f"Cycle detected while reconstructing route "
                    f"starting from {satellites[start]}."
                )

            visited.add(
                next_satellite
            )

            # =================================================================
            # ADD ISL HOP
            # =================================================================

            isl_hops += 1

            route.append(
                satellites[next_satellite]
            )

            current = next_satellite

        # =====================================================================
        # HOP VALIDATION
        # =====================================================================

        if isl_hops < min_isl_hops:

            raise RuntimeError(
                f"Route starting from {satellites[start]} contains "
                f"{isl_hops} ISL hops, below the required minimum "
                f"of {min_isl_hops}."
            )

        if isl_hops > max_isl_hops:

            raise RuntimeError(
                f"Route starting from {satellites[start]} contains "
                f"{isl_hops} ISL hops, exceeding the allowed maximum "
                f"of {max_isl_hops}."
            )

        routes.append(
            route
        )

    # =========================================================================
    # 7. FINAL VALIDATION
    # =========================================================================

    if len(routes) != num_paths:

        raise RuntimeError(
            f"Number of reconstructed routes ({len(routes)}) "
            f"does not match num_paths ({num_paths})."
        )

    return routes


# =============================================================================
# STAGE 2
# ONE-TIMESTAMP SHORTEST-ROUTE OPTIMIZATION
# =============================================================================


def solve_one_timestamp_route(
    U_fes,
    I_fes,
    F_fes,
    C_US,
    C_ISL,
    C_F,
    demand,
    L_US,
    L_ISL,
    L_F,
    solver=None,
    fixed_serving_satellite=None,
):
    """
    Stage 2: One-timestamp shortest-route optimization.

    INPUT:

        U_fes
        I_fes
        F_fes

    These are fixed values obtained from Stage 1.

    OUTPUT:

        U_one
        I_one
        F_one

    These are new binary decision variables representing exactly
    one source-to-gateway route.

    No temporal coupling is introduced here.

    Parameters
    ----------
    U_fes : ndarray, shape (S,)
        Fixed Stage-1 service-link feasibility values.

    I_fes : ndarray, shape (S,S)
        Fixed Stage-1 ISL feasibility values.

    F_fes : ndarray, shape (S,G)
        Fixed Stage-1 feeder-link feasibility values.

    C_US : ndarray, shape (S,)
        Source-to-satellite capacities.

    C_ISL : ndarray, shape (S,S)
        ISL capacities.

    C_F : ndarray, shape (S,G)
        Feeder capacities.

    demand : float
        Required traffic demand in Mbps.

    L_US : ndarray, shape (S,)
        Source-to-satellite latency.

    L_ISL : ndarray, shape (S,S)
        ISL latency.

    L_F : ndarray, shape (S,G)
        Feeder latency.

    solver : optional
        CVXPY mixed-integer solver.

    Returns
    -------
    dict
        Stage-2 solution containing:

            U_one
            I_one
            F_one
            H_one
            path
            objective_value
            status
            feasible
    """

    # =========================================================================
    # 1. CONVERT INPUTS
    # =========================================================================

    U_fes = np.asarray(
        U_fes,
        dtype=float,
    )

    I_fes = np.asarray(
        I_fes,
        dtype=float,
    )

    F_fes = np.asarray(
        F_fes,
        dtype=float,
    )

    C_US = np.asarray(
        C_US,
        dtype=float,
    )

    C_ISL = np.asarray(
        C_ISL,
        dtype=float,
    )

    C_F = np.asarray(
        C_F,
        dtype=float,
    )

    L_US = np.asarray(
        L_US,
        dtype=float,
    )

    L_ISL = np.asarray(
        L_ISL,
        dtype=float,
    )

    L_F = np.asarray(
        L_F,
        dtype=float,
    )

    # =========================================================================
    # 2. DIMENSIONS
    # =========================================================================

    if U_fes.ndim != 1:
        raise ValueError(
            "U_fes must be one-dimensional."
        )

    S = U_fes.shape[0]

    if I_fes.shape != (S, S):
        raise ValueError(
            f"I_fes must have shape {(S, S)}. "
            f"Received {I_fes.shape}."
        )

    if F_fes.ndim != 2:
        raise ValueError(
            "F_fes must be two-dimensional."
        )

    G = F_fes.shape[1]

    if C_US.shape != (S,):
        raise ValueError(
            f"C_US must have shape {(S,)}."
        )

    if C_ISL.shape != (S, S):
        raise ValueError(
            f"C_ISL must have shape {(S, S)}."
        )

    if C_F.shape != (S, G):
        raise ValueError(
            f"C_F must have shape {(S, G)}."
        )

    if L_US.shape != (S,):
        raise ValueError(
            f"L_US must have shape {(S,)}."
        )

    if L_ISL.shape != (S, S):
        raise ValueError(
            f"L_ISL must have shape {(S, S)}."
        )

    if L_F.shape != (S, G):
        raise ValueError(
            f"L_F must have shape {(S, G)}."
        )

    # =========================================================================
    # 4. FINAL ROUTE VARIABLES
    # =========================================================================
    #
    # IMPORTANT:
    #
    # U_fes/I_fes/F_fes are FIXED Stage-1 values.
    #
    # U_one/I_one/F_one are NEW Stage-2 optimization variables.
    # =========================================================================

    U_one = cp.Variable(
        S,
        boolean=True,
    )

    I_one = cp.Variable(
        (S, S),
        boolean=True,
    )

    F_one = cp.Variable(
        (S, G),
        boolean=True,
    )


    constraints = []

    # =========================================================================
    # 5. ONLY STAGE-1 FEASIBLE LINKS MAY BE USED
    # =========================================================================

    constraints.append(
        U_one <= U_fes
    )

    constraints.append(
        I_one <= I_fes
    )

    constraints.append(
        F_one <= F_fes
    )

    # =========================================================================
    # 6. NO SELF-LOOP ISLs
    # =========================================================================

    for i in range(S):

        constraints.append(
            I_one[i, i] == 0
        )

    # =========================================================================
    # 7. EXACTLY ONE SERVING SATELLITE
    # =========================================================================

    constraints.append(
        cp.sum(U_one) == 1
    )

    # =========================================================================
    # 8. EXACTLY ONE FEEDER LINK
    # =========================================================================

    constraints.append(
        cp.sum(F_one) == 1
    )

    # =========================================================================
    # 9. DEMAND-CAPACITY CONSTRAINTS
    # =========================================================================

    constraints.append(
        demand * U_one <= C_US
    )

    constraints.append(
        demand * I_one <= C_ISL
    )

    constraints.append(
        demand * F_one <= C_F
    )

    # =========================================================================
    # 10. FLOW CONSERVATION
    # =========================================================================
    #
    # For every satellite i:
    #
    #     U_one[i] + sum_j I_one[j,i]
    #
    #          =
    #
    #     sum_j I_one[i,j] + sum_g F_one[i,g]
    # =========================================================================

    for i in range(S):

        incoming = (
            U_one[i]
            +
            cp.sum(I_one[:, i])
        )

        outgoing = (
            cp.sum(I_one[i, :])
            +
            cp.sum(F_one[i, :])
        )

        constraints.append(
            incoming == outgoing
        )

        # Non-branching
        constraints.append(
            cp.sum(I_one[:, i]) <= 1
        )

        constraints.append(
            cp.sum(I_one[i, :]) <= 1
        )

        constraints.append(
            cp.sum(F_one[i, :]) <= 1
        )

    # =========================================================================
    # 11. Fixed serving satellite enforcing
    # =========================================================================
    if fixed_serving_satellite is not None:
        constraints.append(
            U_one[fixed_serving_satellite] == 1
        )

    # =========================================================================
    # 12. OBJECTIVE
    # =========================================================================
    #
    # Minimize latency of exactly ONE route.
    #
    # Since:
    #
    #     U_one <= U_fes
    #     I_one <= I_fes
    #     F_one <= F_fes
    #
    # the Stage-1 masks are fixed.
    # =========================================================================

    objective = cp.Minimize(

        cp.sum(
            cp.multiply(
                L_US * U_fes,
                U_one,
            )
        )

        +

        cp.sum(
            cp.multiply(
                L_ISL * I_fes,
                I_one,
            )
        )

        +

        cp.sum(
            cp.multiply(
                L_F * F_fes,
                F_one,
            )
        )
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    # =========================================================================
    # 13. SELECT MIP SOLVER
    # =========================================================================

    if solver is None:
        solver = _select_milp_solver()

    # IMPORTANT:
    #
    # Do NOT use CLARABEL here.
    #
    # U_one/I_one/F_one are Boolean variables, so a mixed-integer
    # capable solver is required.
    # =========================================================================

    problem.solve(
        solver=solver,
        verbose=False,
    )

    # =========================================================================
    # 14. CHECK SOLVER STATUS
    # =========================================================================

    feasible = problem.status in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    )

    # =========================================================================
    # 15. DEFAULT RESULT
    # =========================================================================

    result = {
        "feasible": feasible,
        "status": problem.status,

        "U_one": None,
        "I_one": None,
        "F_one": None,

        "objective_value": None,

        "path": None,
    }

    # =========================================================================
    # 16. NO FEASIBLE SOLUTION
    # =========================================================================

    if not feasible:

        result["U_one"] = np.zeros(
            S,
            dtype=int,
        )

        result["I_one"] = np.zeros(
            (S, S),
            dtype=int,
        )

        result["F_one"] = np.zeros(
            (S, G),
            dtype=int,
        )

        result["H_one"] = np.zeros(
            S,
            dtype=int,
        )

        return result

    # =========================================================================
    # 17. CHECK VARIABLE VALUES
    # =========================================================================

    if (
        U_one.value is None
        or I_one.value is None
        or F_one.value is None
        ):

        return {
            "feasible": False,
            "status": "NO_SOLUTION_VALUES",

            "U_one": np.zeros(
                S,
                dtype=int,
            ),

            "I_one": np.zeros(
                (S, S),
                dtype=int,
            ),

            "F_one": np.zeros(
                (S, G),
                dtype=int,
            ),

            "objective_value": None,
            "path": None,
        }

    # =========================================================================
    # 18. CONVERT SOLUTION TO INTEGER ARRAYS
    # =========================================================================

    U_value = np.rint(
        U_one.value
    ).astype(int)

    I_value = np.rint(
        I_one.value
    ).astype(int)

    F_value = np.rint(
        F_one.value
    ).astype(int)

    

    # =========================================================================
    # 19. RECONSTRUCT SINGLE ROUTE
    # =========================================================================

    source_satellites = [
        i
        for i in range(S)
        if U_value[i] == 1
    ]

    if len(source_satellites) != 1:

        return {
            "feasible": False,
            "status": "INVALID_FINAL_SOLUTION",

            "U_one": U_value,
            "I_one": I_value,
            "F_one": F_value,

            "objective_value": (
                float(cast(float, problem.value))
                if problem.value is not None
                else None
            ),

            "path": None,
        }

    # =========================================================================
    # 21. FOLLOW SELECTED ISLs
    # =========================================================================

    current = int(
        source_satellites[0]
    )

    path_indices = [
        current
    ]

    visited = {
        current
    }

    terminal_gateway = None
    isl_hops = 0

    for _ in range(S + 1):

        # ---------------------------------------------------------------------
        # Feeder termination
        # ---------------------------------------------------------------------

        feeder_targets = [
            g
            for g in range(G)
            if F_value[current, g] == 1
        ]

        if len(feeder_targets) == 1:

            terminal_gateway = int(
                feeder_targets[0]
            )

            break

        if len(feeder_targets) > 1:

            terminal_gateway = None

            break

        # ---------------------------------------------------------------------
        # ISL successor
        # ---------------------------------------------------------------------

        next_satellites = [
            j
            for j in range(S)
            if I_value[current, j] == 1
        ]

        if len(next_satellites) != 1:

            terminal_gateway = None

            break

        next_satellite = int(
            next_satellites[0]
        )

        # ---------------------------------------------------------------------
        # Cycle detection
        # ---------------------------------------------------------------------

        if next_satellite in visited:

            terminal_gateway = None

            break

        visited.add(
            next_satellite
        )

        path_indices.append(
            next_satellite
        )

        isl_hops += 1

        current = next_satellite

    # =========================================================================
    # 21. VALIDATE HOP COUNT
    # =========================================================================

    if terminal_gateway is None:

        path = None

    else:

        path = [
            "Z"
        ]

        path.extend(
            f"S{i + 1}"
            for i in path_indices
        )

        path.append(
            f"GW{terminal_gateway + 1}"
        )

    # =========================================================================
    # 22. OBJECTIVE VALUE
    # =========================================================================

    objective_value = (
        float(cast(float, problem.value))
        if problem.value is not None
        else None
    )

    # =========================================================================
    # 23. RETURN STAGE-2 RESULT
    # =========================================================================

    return {
        "feasible": (
            path is not None
        ),

        "status": problem.status,

        "U_one": U_value,

        "I_one": I_value,

        "F_one": F_value,

        "objective_value": objective_value,

        "path": path,
    }

# =============================================================================
# RUN JOINT TEMPORAL OPTIMIZATION
# =============================================================================

def run_joint_temporal_optimization(
    E,
    C,
    L,
    feasibility_results,
    demand=350.0,
    H=8.0,
    fixed_serving_satellite=None,

):
    """
    Run the joint multi-slot temporal optimization.

    Stage 1:
        U_fes[k]
        I_fes[k]
        F_fes[k]

    Joint temporal optimization:
        U_temp[k]
        I_temp[k]
        F_temp[k]

    The optimization is performed jointly over all time slots.

    The objective is:

        total latency
        +
        handover penalty

    No explicit h_k variable is used.
    """

    print()
    print("=" * 80)
    print("JOINT MULTI-SLOT TEMPORAL OPTIMIZATION")
    print("=" * 80)

    # =========================================================================
    # COLLECT STAGE-1 FEASIBILITY
    # =========================================================================

    U_fes = {}
    I_fes = {}
    F_fes = {}

    C_US = {}
    C_ISL = {}
    C_F = {}

    L_US = {}
    L_ISL = {}
    L_F = {}

    for k in sorted(E):

        result = feasibility_results.get(
            k,
            {},
        )

        U_fes[k] = result.get(
            "U_fes",
            result.get("U"),
        )

        I_fes[k] = result.get(
            "I_fes",
            result.get("I"),
        )

        F_fes[k] = result.get(
            "F_fes",
            result.get("F"),
        )

        if (
            U_fes[k] is None
            or I_fes[k] is None
            or F_fes[k] is None
        ):
            raise ValueError(
                f"Stage-1 feasibility missing "
                f"for time slot k={k}."
            )

        C_US[k] = C[k]["US"]
        C_ISL[k] = C[k]["ISL"]
        C_F[k] = C[k]["F"]

        L_US[k] = L[k]["US"]
        L_ISL[k] = L[k]["ISL"]
        L_F[k] = L[k]["F"]

    # =========================================================================
    # JOINT OPTIMIZATION
    # =========================================================================

    result = solve_joint_temporal_optimization(

        U_fes=U_fes,
        I_fes=I_fes,
        F_fes=F_fes,

        C_US=C_US,
        C_ISL=C_ISL,
        C_F=C_F,

        L_US=L_US,
        L_ISL=L_ISL,
        L_F=L_F,

        demand=demand,
        fixed_serving_satellite=fixed_serving_satellite,

    )

    # =========================================================================
    # DISPLAY SOLVER RESULT
    # =========================================================================

    print()
    print(
        f"Solver status: "
        f"{result['status']}"
    )

    print(
        f"Total objective: "
        f"{result['objective_value']}"
    )

    print(
        f"Routing latency cost: "
        f"{result['routing_cost']}"
    )

    print(
        f"Handover cost: "
        f"{result['handover_cost']}"
    )

    # =========================================================================
    # DISPLAY SELECTED SERVING SATELLITE
    # =========================================================================

    print()
    print("JOINT TEMPORAL SERVING SATELLITES")
    print("-" * 80)

    U_temp = result["U_temp"]

    for k in sorted(U_temp):

        if U_temp[k] is None:

            print(
                f"k={k}: no solution"
            )

            continue

        serving = np.where(
            U_temp[k] > 0.5
        )[0]

        if len(serving) == 1:

            sat = (
                f"S{serving[0] + 1}"
            )

            print(
                f"k={k:2d}: {sat}"
            )

        else:

            print(
                f"k={k:2d}: invalid "
                f"serving-satellite selection"
            )

    # =========================================================================
    # DISPLAY HANDOVERS
    # =========================================================================

    print()
    print("HANDOVERS")
    print("-" * 80)

    previous_serving = None
    number_handovers = 0

    for k in sorted(U_temp):

        if U_temp[k] is None:
            continue

        serving = np.where(
            U_temp[k] > 0.5
        )[0]

        if len(serving) != 1:
            continue

        current_serving = int(
            serving[0]
        )

        if previous_serving is not None:

            if current_serving != previous_serving:

                number_handovers += 1

                print(
                    f"k={k:2d}: "
                    f"S{previous_serving + 1}"
                    f" -> "
                    f"S{current_serving + 1}"
                    f" | penalty = {H:.3f}"
                )

        previous_serving = current_serving

    print()
    print(
        f"Total handovers: "
        f"{number_handovers}"
    )

    print(
        f"Total handover penalty: "
        f"{number_handovers * H:.3f}"
    )

    return result

# =============================================================================
# JOINT MULTI-SLOT TEMPORAL OPTIMIZATION
# =============================================================================


def solve_joint_temporal_optimization(
    U_fes,
    I_fes,
    F_fes,
    C_US,
    C_ISL,
    C_F,
    L_US,
    L_ISL,
    L_F,
    demand=350.0,
    H=8.0,
    solver=None,
    fixed_serving_satellite=None,
):
    """
    Joint multi-slot temporal optimization.

    Stage 1 provides fixed feasible-link masks:

        U_fes[k]
        I_fes[k]
        F_fes[k]

    The joint optimization determines:

        U_temp[k]
        I_temp[k]
        F_temp[k]

    simultaneously for all time slots.

    The objective is:

        minimize

            sum_k C_k

            +
            
            (H / 2)
            * sum_{k=1}^K sum_i
                (U_i^k - U_i^(k-1))^2

    where C_k is the routing latency at time k.

    Because exactly one serving satellite is selected at every
    timestamp, the quadratic term is exactly equal to:

        H

    when the serving satellite changes, and:

        0

    when the serving satellite remains unchanged.

    Therefore:

        (H / 2)
        * sum_i (U_i^k - U_i^(k-1))^2

    represents exactly one handover penalty.

    No explicit handover variable h_k is introduced.

    Parameters
    ----------
    U_fes : dict
        Stage-1 feasible source-to-satellite links indexed by time.

    I_fes : dict
        Stage-1 feasible ISL links indexed by time.

    F_fes : dict
        Stage-1 feasible feeder links indexed by time.

    C_US : dict
        Source-to-satellite capacities indexed by time.

    C_ISL : dict
        ISL capacities indexed by time.

    C_F : dict
        Feeder-link capacities indexed by time.

    L_US : dict
        Source-to-satellite latencies indexed by time.

    L_ISL : dict
        ISL latencies indexed by time.

    L_F : dict
        Feeder-link latencies indexed by time.

    demand : float
        Required traffic demand in Mbps.

    H : float
        Handover penalty per serving-satellite change.
    solver : optional
        Mixed-integer quadratic solver supported by CVXPY.

    Returns
    -------
    result : dict
        Dictionary containing:

            feasible
            status
            objective_value
            routing_cost
            handover_cost
            handover_count
            U_temp
            I_temp
            F_temp
            serving_satellite
    """

    # =========================================================================
    # 1. TIME-SLOT VALIDATION
    # =========================================================================

    time_slots = sorted(U_fes.keys())

    if not time_slots:
        raise ValueError(
            "No time slots were provided to joint temporal optimization."
        )

    # All input dictionaries must contain the same time slots.
    required_data = {
        "I_fes": I_fes,
        "F_fes": F_fes,
        "C_US": C_US,
        "C_ISL": C_ISL,
        "C_F": C_F,
        "L_US": L_US,
        "L_ISL": L_ISL,
        "L_F": L_F,
    }

    for name, data in required_data.items():

        missing = [
            k
            for k in time_slots
            if k not in data
        ]

        if missing:

            raise ValueError(
                f"Missing time-slot data for {name}: {missing}"
            )

    # =========================================================================
    # 2. CONVERT INPUT DATA TO NUMPY ARRAYS
    # =========================================================================

    U_fes = {
        k: np.asarray(
            U_fes[k],
            dtype=float,
        )
        for k in time_slots
    }

    I_fes = {
        k: np.asarray(
            I_fes[k],
            dtype=float,
        )
        for k in time_slots
    }

    F_fes = {
        k: np.asarray(
            F_fes[k],
            dtype=float,
        )
        for k in time_slots
    }

    C_US = {
        k: np.asarray(
            C_US[k],
            dtype=float,
        )
        for k in time_slots
    }

    C_ISL = {
        k: np.asarray(
            C_ISL[k],
            dtype=float,
        )
        for k in time_slots
    }

    C_F = {
        k: np.asarray(
            C_F[k],
            dtype=float,
        )
        for k in time_slots
    }

    L_US = {
        k: np.asarray(
            L_US[k],
            dtype=float,
        )
        for k in time_slots
    }

    L_ISL = {
        k: np.asarray(
            L_ISL[k],
            dtype=float,
        )
        for k in time_slots
    }

    L_F = {
        k: np.asarray(
            L_F[k],
            dtype=float,
        )
        for k in time_slots
    }

    # =========================================================================
    # 3. DETERMINE NETWORK DIMENSIONS
    # =========================================================================

    first_k = time_slots[0]

    if U_fes[first_k].ndim != 1:

        raise ValueError(
            f"U_fes[{first_k}] must be one-dimensional. "
            f"Received shape={U_fes[first_k].shape}."
        )

    S = U_fes[first_k].shape[0]

    if I_fes[first_k].shape != (S, S):

        raise ValueError(
            f"I_fes[{first_k}] must have shape {(S, S)}. "
            f"Received shape={I_fes[first_k].shape}."
        )

    if F_fes[first_k].ndim != 2:

        raise ValueError(
            f"F_fes[{first_k}] must be two-dimensional."
        )

    G = F_fes[first_k].shape[1]

    # =========================================================================
    # 4. PARAMETER VALIDATION
    # =========================================================================

    if H < 0:

        raise ValueError(
            "H must be >= 0."
        )

    if demand < 0:

        raise ValueError(
            "demand must be >= 0."
        )

    # =========================================================================
    # 5. VALIDATE ALL TIME-SLOT DIMENSIONS
    # =========================================================================

    for k in time_slots:

        if U_fes[k].shape != (S,):

            raise ValueError(
                f"U_fes[{k}] has shape {U_fes[k].shape}; "
                f"expected {(S,)}."
            )

        if I_fes[k].shape != (S, S):

            raise ValueError(
                f"I_fes[{k}] has shape {I_fes[k].shape}; "
                f"expected {(S, S)}."
            )

        if F_fes[k].shape != (S, G):

            raise ValueError(
                f"F_fes[{k}] has shape {F_fes[k].shape}; "
                f"expected {(S, G)}."
            )

        if C_US[k].shape != (S,):

            raise ValueError(
                f"C_US[{k}] has shape {C_US[k].shape}; "
                f"expected {(S,)}."
            )

        if C_ISL[k].shape != (S, S):

            raise ValueError(
                f"C_ISL[{k}] has shape {C_ISL[k].shape}; "
                f"expected {(S, S)}."
            )

        if C_F[k].shape != (S, G):

            raise ValueError(
                f"C_F[{k}] has shape {C_F[k].shape}; "
                f"expected {(S, G)}."
            )

        if L_US[k].shape != (S,):

            raise ValueError(
                f"L_US[{k}] has shape {L_US[k].shape}; "
                f"expected {(S,)}."
            )

        if L_ISL[k].shape != (S, S):

            raise ValueError(
                f"L_ISL[{k}] has shape {L_ISL[k].shape}; "
                f"expected {(S, S)}."
            )

        if L_F[k].shape != (S, G):

            raise ValueError(
                f"L_F[{k}] has shape {L_F[k].shape}; "
                f"expected {(S, G)}."
            )

    # =========================================================================
    # 6. DECISION VARIABLES
    # =========================================================================
    #
    # For every timestamp k:
    #
    # U_temp[k][i]
    #
    #     = 1 if satellite i is selected as serving satellite.
    #
    # I_temp[k][i,j]
    #
    #     = 1 if ISL i -> j is selected.
    #
    # F_temp[k][i,g]
    #
    #     = 1 if satellite i -> gateway g is selected.
    #
    # =========================================================================

    U_temp = {
        k: cp.Variable(
            S,
            boolean=True,
        )
        for k in time_slots
    }

    I_temp = {
        k: cp.Variable(
            (S, S),
            boolean=True,
        )
        for k in time_slots
    }

    F_temp = {
        k: cp.Variable(
            (S, G),
            boolean=True,
        )
        for k in time_slots
    }

    constraints = []


    # =========================================================================
    # 7. PER-TIMESTAMP ROUTING CONSTRAINTS
    # =========================================================================

    for k in time_slots:

        U_k = U_temp[k]
        I_k = I_temp[k]
        F_k = F_temp[k]

        # ---------------------------------------------------------------------
# Optional fixed serving satellite at timestamp k=1 ONLY.
#
#     U_1[i*] = 1
#
# No such constraint is imposed for k != 1.
# ---------------------------------------------------------------------

        if (
            k == 1
            and
            fixed_serving_satellite is not None
        ):

            constraints.append(
                U_k[fixed_serving_satellite] == 1
            )

        # ---------------------------------------------------------------------
        # 7.1 Stage-1 feasibility masks
        # ---------------------------------------------------------------------
        #
        # Stage 1 determines which links are feasible.
        #
        # Stage 3 can only select links from those feasible sets.
        # ---------------------------------------------------------------------

        constraints.append(
            U_k <= U_fes[k]
        )

        constraints.append(
            I_k <= I_fes[k]
        )

        constraints.append(
            F_k <= F_fes[k]
        )

        # ---------------------------------------------------------------------
        # 7.2 No ISL self-loops
        # ---------------------------------------------------------------------

        for i in range(S):

            constraints.append(
                I_k[i, i] == 0
            )

        # ---------------------------------------------------------------------
        # 7.3 Exactly one serving satellite
        # ---------------------------------------------------------------------

        constraints.append(
            cp.sum(U_k) == 1
        )

        # ---------------------------------------------------------------------
        # 7.4 Exactly one feeder link
        # ---------------------------------------------------------------------

        constraints.append(
            cp.sum(F_k) == 1
        )

        # ---------------------------------------------------------------------
        # 7.5 Demand-capacity constraints
        # ---------------------------------------------------------------------

        constraints.append(
            demand * U_k <= C_US[k]
        )

        constraints.append(
            demand * I_k <= C_ISL[k]
        )

        constraints.append(
            demand * F_k <= C_F[k]
        )

        # ---------------------------------------------------------------------
        # 7.6 Flow conservation
        # ---------------------------------------------------------------------
        #
        # For every satellite i:
        #
        #     U_i
        #     +
        #     sum_j I_ji
        #
        #     =
        #
        #     sum_j I_ij
        #     +
        #     sum_g F_ig
        #
        # ---------------------------------------------------------------------

        for i in range(S):

            incoming = (
                U_k[i]
                +
                cp.sum(
                    I_k[:, i]
                )
            )

            outgoing = (
                cp.sum(
                    I_k[i, :]
                )
                +
                cp.sum(
                    F_k[i, :]
                )
            )

            constraints.append(
                incoming == outgoing
            )

            # -----------------------------------------------------------------
            # Non-branching:
            #
            # At most one incoming connection.
            # -----------------------------------------------------------------

            constraints.append(
                incoming <= 1
            )

            # -----------------------------------------------------------------
            # Non-branching:
            #
            # At most one outgoing connection.
            # -----------------------------------------------------------------

            constraints.append(
                outgoing <= 1
            )

            # -----------------------------------------------------------------
            # The serving satellite must use at least one ISL.
            #
            # This prevents:
            #
            #     Z -> S_i -> GW
            #
            # and therefore makes min_isl_hops meaningful.
            # -----------------------------------------------------------------

            constraints.append(
                U_k[i] <= cp.sum(
                    I_k[i, :]
                )
            )

        
    # =========================================================================
    # 8. ROUTING LATENCY OBJECTIVE
    # =========================================================================
    #
    # For every timestamp:
    #
    # C_k =
    #
    #     sum_i
    #         L_US[i,k] U_i^k
    #
    #   + sum_i,j
    #         L_ISL[i,j,k] I_ij^k
    #
    #   + sum_i,g
    #         L_F[i,g,k] F_ig^k
    #
    # =========================================================================

    routing_cost = 0

    for k in time_slots:

        routing_cost += (

            cp.sum(
                cp.multiply(
                    L_US[k],
                    U_temp[k],
                )
            )

            +

            cp.sum(
                cp.multiply(
                    L_ISL[k],
                    I_temp[k],
                )
            )

            +

            cp.sum(
                cp.multiply(
                    L_F[k],
                    F_temp[k],
                )
            )
        )

    # =========================================================================
    # 9. HANDOVER PENALTY
    # =========================================================================
    #
    # Between two consecutive timestamps:
    #
    #     k-1 -> k
    #
    # calculate:
    #
    #     H/2 * sum_i
    #
    #         (U_i^k - U_i^(k-1))^2
    #
    # Since each U vector contains exactly one '1':
    #
    # SAME SATELLITE:
    #
    #     U^k = U^(k-1)
    #
    #     sum_i (...)^2 = 0
    #
    # DIFFERENT SATELLITE:
    #
    #     one satellite changes 1 -> 0
    #     another changes 0 -> 1
    #
    #     sum_i (...)^2 = 2
    #
    # Therefore:
    #
    #     H/2 * 2 = H
    #
    # exactly one handover penalty.
    # =========================================================================

    handover_cost = 0

    for previous_k, current_k in zip(
        time_slots[:-1],
        time_slots[1:],
    ):

        handover_cost += (

            H / 2.0

        ) * cp.sum(

            cp.square(

                U_temp[current_k]
                -
                U_temp[previous_k]

            )

        )

    # =========================================================================
    # 10. COMPLETE JOINT OBJECTIVE
    # =========================================================================
    #
    #     minimize
    #
    #         routing latency
    #
    #         +
    #
    #         handover penalty
    # =========================================================================

    objective = cp.Minimize(
        routing_cost
        +
        handover_cost
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    # =========================================================================
    # 11. SOLVER
    # =========================================================================
    #
    # IMPORTANT:
    #
    # This is a mixed-integer quadratic program (MIQP), not a simple MILP,
    # because the handover objective contains:
    #
    #     cp.square(...)
    #
    # Therefore the solver must support mixed-integer convex quadratic
    # optimization.
    # =========================================================================

    if solver is None:

        solver = _select_milp_solver()

    try:

        problem.solve(
            solver=solver,
            verbose=False,
        )

    except Exception as exc:

        raise RuntimeError(

            "Joint temporal optimization failed. "

            f"Selected solver={solver}. "

            "The joint formulation contains the quadratic handover "
            "term and therefore requires an MIQP-capable solver. "

            f"Original solver error: {exc}"

        ) from exc

    # =========================================================================
    # 12. CHECK STATUS
    # =========================================================================

    feasible = problem.status in (

        cp.OPTIMAL,

        cp.OPTIMAL_INACCURATE,

    )

    # =========================================================================
    # 13. INITIAL RESULT STRUCTURE
    # =========================================================================

    result = {

        "feasible":
            feasible,

        "status":
            problem.status,

        "objective_value":
            None,

        "routing_cost":
            None,

        "handover_cost":
            None,

        "handover_count":
            None,

        "U_temp":
            {},

        "I_temp":
            {},

        "F_temp":
            {},

        "serving_satellite":
            {},
        "handover_penalty_per_change":
            H,
    }

    # =========================================================================
    # 14. HANDLE INFEASIBLE SOLUTION
    # =========================================================================

    if not feasible:

        for k in time_slots:

            result["U_temp"][k] = np.zeros(
                S,
                dtype=int,
            )

            result["I_temp"][k] = np.zeros(
                (S, S),
                dtype=int,
            )

            result["F_temp"][k] = np.zeros(
                (S, G),
                dtype=int,
            )

            result["serving_satellite"][k] = None

        return result

    # =========================================================================
    # 15. EXTRACT ALL TIME-SLOT SOLUTIONS
    # =========================================================================

    for k in time_slots:

        if (

            U_temp[k].value is None

            or

            I_temp[k].value is None

            or

            F_temp[k].value is None

        ):

            result["feasible"] = False

            result["status"] = (
                "NO_SOLUTION_VALUES"
            )

            for kk in time_slots:

                result["U_temp"][kk] = np.zeros(
                    S,
                    dtype=int,
                )

                result["I_temp"][kk] = np.zeros(
                    (S, S),
                    dtype=int,
                )

                result["F_temp"][kk] = np.zeros(
                    (S, G),
                    dtype=int,
                )

                result["serving_satellite"][kk] = None

            return result

                # ---------------------------------------------------------------------
        # Convert CVXPY solution to integer arrays.
        # ---------------------------------------------------------------------

        U_solution = U_temp[k].value
        I_solution = I_temp[k].value
        F_solution = F_temp[k].value

        if (
            U_solution is None
            or I_solution is None
            or F_solution is None
        ):
            result["feasible"] = False
            result["status"] = "NO_SOLUTION_VALUES"
            return result

        result["U_temp"][k] = np.rint(
            np.asarray(U_solution, dtype=float)
        ).astype(int)

        result["I_temp"][k] = np.rint(
            np.asarray(I_solution, dtype=float)
        ).astype(int)

        result["F_temp"][k] = np.rint(
            np.asarray(F_solution, dtype=float)
        ).astype(int)

        # ---------------------------------------------------------------------
        # Identify serving satellite.
        # ---------------------------------------------------------------------

        serving = np.where(
            result["U_temp"][k] == 1
        )[0]

        if len(serving) == 1:

            result["serving_satellite"][k] = int(
                serving[0]
            )

        else:

            result["serving_satellite"][k] = None

    # =========================================================================
    # 16. RE-CALCULATE ROUTING LATENCY FROM INTEGER SOLUTION
    # =========================================================================
    #
    # We calculate the cost again from the extracted binary arrays rather
    # than relying only on solver floating-point output.
    # =========================================================================

    actual_routing_cost = 0.0

    for k in time_slots:

        actual_routing_cost += float(

            np.sum(
                L_US[k]
                *
                result["U_temp"][k]
            )

            +

            np.sum(
                L_ISL[k]
                *
                result["I_temp"][k]
            )

            +

            np.sum(
                L_F[k]
                *
                result["F_temp"][k]
            )

        )

    # =========================================================================
    # 17. COUNT HANDOVERS
    # =========================================================================

    actual_handover_count = 0

    for previous_k, current_k in zip(
        time_slots[:-1],
        time_slots[1:],
    ):

        previous_serving = (
            result["serving_satellite"][previous_k]
        )

        current_serving = (
            result["serving_satellite"][current_k]
        )

        if (

            previous_serving is not None

            and

            current_serving is not None

            and

            previous_serving != current_serving

        ):

            actual_handover_count += 1

    # =========================================================================
    # 18. HANDOVER COST
    # =========================================================================

    actual_handover_cost = (

        H
        *
        actual_handover_count

    )

    # =========================================================================
    # 19. FINAL OBJECTIVE
    # =========================================================================

    actual_objective = (

        actual_routing_cost
        +
        actual_handover_cost

    )

    # =========================================================================
    # 20. STORE FINAL METRICS
    # =========================================================================

    result["routing_cost"] = (
        actual_routing_cost
    )

    result["handover_count"] = (
        actual_handover_count
    )

    result["handover_cost"] = (
        actual_handover_cost
    )

    result["objective_value"] = (
        actual_objective
    )

    # =========================================================================
    # 21. RETURN
    # =========================================================================

    return result

# =============================================================================
# SEQUENTIAL TEMPORAL OPTIMIZATION USING DYNAMIC PROGRAMMING
# =============================================================================

def solve_sequential_dp(
    candidate_costs,
    candidate_routes,
    H=8.0,
):
    """
    Sequential temporal optimization using Dynamic Programming.

    Objective:

        min
            sum_k c_k(s_k)
            +
            H * sum_{k=1}^K 1{s_k != s_{k-1}}

    where:

        c_k(s_i)

    is the minimum routing latency when satellite S_i serves at
    timestamp k.

    DP state:

        J_k(i)

    = minimum total cost from timestamp 0 through k when S_i
      serves at timestamp k.

    Recursion:

        J_k(j)
        =
        c_k(j)
        +
        min_i [
            J_{k-1}(i)
            +
            H * 1{i != j}
        ]

    The predecessor table is used to recover the optimal serving
    satellite sequence.
    """

    # =========================================================================
    # CHECK INPUT
    # =========================================================================

    if not candidate_costs:

        return {
            "feasible": False,
            "reason": "No candidate costs were provided.",
        }

    timestamps = sorted(
        candidate_costs.keys()
    )

    if len(timestamps) == 0:

        return {
            "feasible": False,
            "reason": "No timestamps were provided.",
        }

    # =========================================================================
    # SATELLITE STATE SPACE
    # =========================================================================

    satellites = sorted(
        {
            satellite
            for k in timestamps
            for satellite
            in candidate_costs[k].keys()
        }
    )

    if not satellites:

        return {
            "feasible": False,
            "reason": (
                "No serving-satellite candidates "
                "were provided."
            ),
        }

    # =========================================================================
    # DP ARRAYS
    #
    # J[k][i] = minimum cost up to k ending at satellite i
    #
    # P[k][i] = predecessor satellite at k-1
    # =========================================================================

    J = {

        k: {
            i: np.inf
            for i in satellites
        }

        for k in timestamps
    }

    P = {

        k: {
            i: None
            for i in satellites
        }

        for k in timestamps
    }

    # =========================================================================
    # INITIALIZATION
    #
    # J_0(i) = c_0(i)
    # =========================================================================

    k0 = timestamps[0]

    for i in satellites:

        cost = candidate_costs[
            k0
        ].get(
            i,
            np.inf,
        )

        if np.isfinite(cost):

            J[k0][i] = float(
                cost
            )

    # =========================================================================
    # CHECK INITIAL FEASIBILITY
    # =========================================================================

    if all(
        not np.isfinite(
            J[k0][i]
        )
        for i in satellites
    ):

        return {

            "feasible": False,

            "reason": (
                f"No feasible serving satellite "
                f"at timestamp {k0}."
            ),

            "cost_to_go": J,

            "predecessor": P,
        }

    # =========================================================================
    # DP RECURSION
    # =========================================================================

    for position in range(
        1,
        len(timestamps),
    ):

        k = timestamps[position]

        k_prev = timestamps[
            position - 1
        ]

        # ---------------------------------------------------------------------
        # Evaluate every possible current serving satellite.
        # ---------------------------------------------------------------------

        for j in satellites:

            current_cost = candidate_costs[
                k
            ].get(
                j,
                np.inf,
            )

            # ---------------------------------------------------------------
            # Current satellite is infeasible.
            # ---------------------------------------------------------------

            if not np.isfinite(
                current_cost
            ):

                continue

            # ---------------------------------------------------------------
            # Find best predecessor.
            # ---------------------------------------------------------------

            best_previous_cost = np.inf

            best_previous_satellite = None

            for i in satellites:

                previous_cost = J[
                    k_prev
                ][i]

                if not np.isfinite(
                    previous_cost
                ):

                    continue

                # =============================================================
                # NO HANDOVER
                # =============================================================

                if i == j:

                    transition_cost = (
                        previous_cost
                    )

                # =============================================================
                # HANDOVER
                # =============================================================

                else:

                    transition_cost = (
                        previous_cost
                        + H
                    )

                # =============================================================
                # UPDATE BEST PREDECESSOR
                # =============================================================

                if (
                    transition_cost
                    < best_previous_cost
                ):

                    best_previous_cost = (
                        transition_cost
                    )

                    best_previous_satellite = i

            # -----------------------------------------------------------------
            # No feasible predecessor.
            # -----------------------------------------------------------------

            if best_previous_satellite is None:

                continue

            # -----------------------------------------------------------------
            # DP recursion.
            # -----------------------------------------------------------------

            J[k][j] = (

                current_cost

                +

                best_previous_cost
            )

            P[k][j] = (
                best_previous_satellite
            )

    # =========================================================================
    # TERMINAL STATE
    #
    # s_K = argmin_i J_K(i)
    # =========================================================================

    k_final = timestamps[-1]

    final_satellite = min(

        satellites,

        key=lambda i:
        J[k_final][i],
    )

    final_cost = J[
        k_final
    ][final_satellite]

    # =========================================================================
    # CHECK FINAL FEASIBILITY
    # =========================================================================

    if not np.isfinite(
        final_cost
    ):

        return {

            "feasible": False,

            "reason": (
                "No feasible temporal serving "
                "sequence exists."
            ),

            "cost_to_go": J,

            "predecessor": P,
        }

    # =========================================================================
    # BACKTRACKING
    # =========================================================================

    serving_sequence = {}

    current_satellite = (
        final_satellite
    )

    for position in range(
        len(timestamps) - 1,
        -1,
        -1,
    ):

        k = timestamps[
            position
        ]

        serving_sequence[k] = (
            current_satellite
        )

        predecessor = P[k][
            current_satellite
        ]

        if position > 0:

            if predecessor is None:

                return {

                    "feasible": False,

                    "reason": (
                        "DP backtracking failed."
                    ),

                    "cost_to_go": J,

                    "predecessor": P,
                }

            current_satellite = (
                predecessor
            )

    # =========================================================================
    # REORDER CHRONOLOGICALLY
    # =========================================================================

    serving_sequence = {

        k: serving_sequence[k]

        for k in timestamps
    }

    # =========================================================================
    # COUNT HANDOVERS
    # =========================================================================

    handover_count = 0

    for position in range(
        1,
        len(timestamps),
    ):

        k_prev = timestamps[
            position - 1
        ]

        k = timestamps[
            position
        ]

        if (
            serving_sequence[k]
            !=
            serving_sequence[k_prev]
        ):

            handover_count += 1

    # =========================================================================
    # HANDOVER COST
    # =========================================================================

    handover_cost = (
        H
        * handover_count
    )

    # =========================================================================
    # TOTAL ROUTING COST
    # =========================================================================

    total_routing_cost = 0.0

    for k in timestamps:

        satellite = (
            serving_sequence[k]
        )

        total_routing_cost += float(
            candidate_costs[k][
                satellite
            ]
        )

    # =========================================================================
    # TOTAL COST
    # =========================================================================

    total_cost = (

        total_routing_cost

        +

        handover_cost
    )

    # =========================================================================
    # RETRIEVE ACTUAL ROUTES
    # =========================================================================

    selected_routes = {}

    U_temp = {}

    I_temp = {}

    F_temp = {}

    for k in timestamps:

        satellite = (
            serving_sequence[k]
        )

        route = candidate_routes[
            k
        ].get(
            satellite
        )

        if route is None:

            return {

                "feasible": False,

                "reason": (
                    f"Route missing for "
                    f"timestamp {k}, "
                    f"satellite S"
                    f"{satellite + 1}."
                ),

                "cost_to_go": J,

                "predecessor": P,
            }

        selected_routes[k] = route

        U_temp[k] = route.get(
            "U_one"
        )

        I_temp[k] = route.get(
            "I_one"
        )

        F_temp[k] = route.get(
            "F_one"
        )

    # =========================================================================
    # RETURN
    # =========================================================================

    return {

        "feasible": True,

        # ---------------------------------------------------------------------
        # Cost components
        # ---------------------------------------------------------------------

        "total_routing_cost":
            total_routing_cost,

        "handover_count":
            handover_count,

        "handover_cost":
            handover_cost,

        "total_cost":
            total_cost,

        # ---------------------------------------------------------------------
        # Optimal serving sequence
        # ---------------------------------------------------------------------

        "serving_sequence":
            serving_sequence,

        # ---------------------------------------------------------------------
        # Selected routes
        # ---------------------------------------------------------------------

        "selected_routes":
            selected_routes,

        "U_temp":
            U_temp,

        "I_temp":
            I_temp,

        "F_temp":
            F_temp,

        # ---------------------------------------------------------------------
        # DP internal information
        # ---------------------------------------------------------------------

        "cost_to_go":
            J,

        "predecessor":
            P,

        # ---------------------------------------------------------------------
        # Terminal state
        # ---------------------------------------------------------------------

        "final_serving_satellite":
            final_satellite,

        "final_cost":
            final_cost,
    }
# 
# # =============================================================================
# GENERATE SERVING-SATELLITE CANDIDATES
# =============================================================================

def generate_serving_satellite_candidates(
    U_fes,
    I_fes,
    F_fes,
    C_US,
    C_ISL,
    C_F,
    L_US,
    L_ISL,
    L_F,
    demand=350.0,
    solver=None,
):
    """
    Generate the best feasible route for every possible serving satellite
    at one timestamp.

    Stage-1 provides the feasible U/I/F connectivity.

    For every Stage-1 feasible serving satellite S_i, the existing
    one-timestamp optimizer is solved with

        U_one[i] = 1

    so that S_i is forced to be the serving satellite.

    Therefore:

        c_k(S_i)
        =
        minimum routing latency at timestamp k
        when S_i is the serving satellite.

    Infeasible serving satellites receive:

        c_k(S_i) = infinity

    Parameters
    ----------
    U_fes : array
        Stage-1 feasible source-to-satellite mask.

    I_fes : array
        Stage-1 feasible satellite-to-satellite mask.

    F_fes : array
        Stage-1 feasible satellite-to-gateway mask.

    C_US : array
        Source-to-satellite capacities.

    C_ISL : array
        ISL capacities.

    C_F : array
        Feeder capacities.

    L_US : array
        Source-to-satellite latencies.

    L_ISL : array
        ISL latencies.

    L_F : array
        Satellite-to-gateway latencies.

    demand : float
        Traffic demand in Mbps.

    solver : str or None
        CVXPY MILP solver.

    Returns
    -------
    candidate_costs : dict
        candidate_costs[i] = best routing cost for serving satellite i.

    candidate_routes : dict
        candidate_routes[i] = complete U/I/F solution for serving satellite i.
    """

    # =========================================================================
    # CONVERT INPUTS TO NUMPY ARRAYS
    # =========================================================================

    U_fes = np.asarray(
        U_fes,
        dtype=int,
    )

    I_fes = np.asarray(
        I_fes,
        dtype=int,
    )

    F_fes = np.asarray(
        F_fes,
        dtype=int,
    )

    C_US = np.asarray(
        C_US,
        dtype=float,
    )

    C_ISL = np.asarray(
        C_ISL,
        dtype=float,
    )

    C_F = np.asarray(
        C_F,
        dtype=float,
    )

    L_US = np.asarray(
        L_US,
        dtype=float,
    )

    L_ISL = np.asarray(
        L_ISL,
        dtype=float,
    )

    L_F = np.asarray(
        L_F,
        dtype=float,
    )

    # =========================================================================
    # NUMBER OF SATELLITES
    # =========================================================================

    S = len(U_fes)

    # =========================================================================
    # CHECK DIMENSIONS
    # =========================================================================

    if I_fes.shape != (S, S):

        raise ValueError(
            "I_fes must have shape (S, S). "
            f"Received {I_fes.shape}."
        )

    if C_ISL.shape != (S, S):

        raise ValueError(
            "C_ISL must have shape (S, S). "
            f"Received {C_ISL.shape}."
        )

    if L_ISL.shape != (S, S):

        raise ValueError(
            "L_ISL must have shape (S, S). "
            f"Received {L_ISL.shape}."
        )

    if F_fes.ndim != 2:

        raise ValueError(
            "F_fes must be a two-dimensional matrix."
        )

    # =========================================================================
    # INITIALIZE CANDIDATE STORAGE
    # =========================================================================

    candidate_costs = {
        i: np.inf
        for i in range(S)
    }

    candidate_routes = {
        i: None
        for i in range(S)
    }

    # =========================================================================
    # STAGE-1 FEASIBLE SERVING SATELLITES
    # =========================================================================

    feasible_serving_satellites = np.where(
        U_fes > 0.5
    )[0]

    print()
    print(
        "SERVING-SATELLITE CANDIDATE GENERATION"
    )
    print(
        "-" * 72
    )

    print(
        "Stage-1 feasible serving satellites: "
        + ", ".join(
            f"S{i + 1}"
            for i in feasible_serving_satellites
        )
    )

    # =========================================================================
    # SOLVE ONE-TIMESTAMP PROBLEM FOR EACH POSSIBLE SERVING SATELLITE
    # =========================================================================

    for serving_idx in feasible_serving_satellites:

        print()
        print(
            f"Candidate serving satellite: "
            f"S{serving_idx + 1}"
        )

        # ---------------------------------------------------------------------
        # Force this satellite to be the serving satellite.
        # ---------------------------------------------------------------------

        result = solve_one_timestamp_route(

            U_fes=U_fes,

            I_fes=I_fes,

            F_fes=F_fes,

            C_US=C_US,

            C_ISL=C_ISL,

            C_F=C_F,

            demand=demand,

            L_US=L_US,

            L_ISL=L_ISL,

            L_F=L_F,

            solver=solver,

            fixed_serving_satellite=serving_idx,
        )

        # =====================================================================
        # CHECK FEASIBILITY
        # =====================================================================

        if not result.get(
            "feasible",
            False,
        ):

            print(
                "  Result: INFEASIBLE"
            )

            continue

        # =====================================================================
        # RETRIEVE OBJECTIVE VALUE
        # =====================================================================

        objective_value = result.get(
            "objective_value"
        )

        # ---------------------------------------------------------------------
        # Safety fallback: calculate routing latency directly if the optimizer
        # did not return an objective value.
        # ---------------------------------------------------------------------

        if objective_value is None:

            U_candidate = result.get(
                "U_one"
            )

            I_candidate = result.get(
                "I_one"
            )

            F_candidate = result.get(
                "F_one"
            )

            if (
                U_candidate is not None
                and I_candidate is not None
                and F_candidate is not None
            ):

                objective_value = (

                    np.sum(
                        L_US
                        * np.asarray(
                            U_candidate,
                            dtype=float,
                        )
                    )

                    +

                    np.sum(
                        L_ISL
                        * np.asarray(
                            I_candidate,
                            dtype=float,
                        )
                    )

                    +

                    np.sum(
                        L_F
                        * np.asarray(
                            F_candidate,
                            dtype=float,
                        )
                    )
                )

        # =====================================================================
        # STORE CANDIDATE
        # =====================================================================

        if objective_value is None:

            print(
                "  Result: FEASIBLE "
                "but objective value is unavailable."
            )

            continue

        candidate_costs[serving_idx] = float(
            objective_value
        )

        candidate_routes[serving_idx] = result

        print(
            "  Result: FEASIBLE"
        )

        print(
            f"  Routing cost: "
            f"{candidate_costs[serving_idx]:.3f} ms"
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    feasible_candidates = [

        i

        for i in range(S)

        if np.isfinite(
            candidate_costs[i]
        )
    ]

    print()
    print(
        f"Feasible candidates: "
        f"{len(feasible_candidates)}/{S}"
    )

    return (
        candidate_costs,
        candidate_routes,
    )

# =============================================================================
# BUILD CANDIDATES FOR THE COMPLETE TIME HORIZON
# =============================================================================

def build_temporal_candidates(
    E,
    C,
    L,
    feasibility_results,
    demand=350.0,
    solver=None,
):
    """
    Generate serving-satellite candidates for every timestamp.

    For every timestamp k:

        Stage-1:
            U_fes[k], I_fes[k], F_fes[k]

        Candidate generation:
            c_k(S_i)
            route_k(S_i)

    Returns
    -------
    candidate_costs : dict
        candidate_costs[k][i]

    candidate_routes : dict
        candidate_routes[k][i]
    """

    candidate_costs = {}

    candidate_routes = {}

    print()
    print("=" * 80)
    print(
        "TEMPORAL SERVING-SATELLITE CANDIDATE GENERATION"
    )
    print("=" * 80)

    # =========================================================================
    # LOOP THROUGH ALL TIMESTAMPS
    # =========================================================================

    for k in sorted(E):

        print()
        print("=" * 80)
        print(
            f"CANDIDATE GENERATION | TIME SNAPSHOT k = {k}"
        )
        print("=" * 80)

        # =====================================================================
        # RETRIEVE STAGE-1 FEASIBILITY
        # =====================================================================

        result_fes = feasibility_results.get(
            k,
            {},
        )

        # ---------------------------------------------------------------------
        # Prefer explicit Stage-1 names.
        # Fallback to older names.
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

        # =====================================================================
        # CHECK STAGE-1 FEASIBILITY
        # =====================================================================

        if (
            U_fes is None
            or I_fes is None
            or F_fes is None
        ):

            print(
                "Stage-1 feasibility not available."
            )

            candidate_costs[k] = {}

            candidate_routes[k] = {}

            continue

        # =====================================================================
        # GENERATE CANDIDATES
        # =====================================================================

        costs_k, routes_k = (
            generate_serving_satellite_candidates(

                U_fes=U_fes,

                I_fes=I_fes,

                F_fes=F_fes,

                C_US=C[k]["US"],

                C_ISL=C[k]["ISL"],

                C_F=C[k]["F"],

                L_US=L[k]["US"],

                L_ISL=L[k]["ISL"],

                L_F=L[k]["F"],

                demand=demand,

                solver=solver,
            )
        )

        candidate_costs[k] = costs_k

        candidate_routes[k] = routes_k

    # =========================================================================
    # COMPLETE-HORIZON SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "TEMPORAL CANDIDATE GENERATION COMPLETE"
    )
    print("=" * 80)

    for k in sorted(candidate_costs):

        feasible = [

            i

            for i, cost
            in candidate_costs[k].items()

            if np.isfinite(cost)
        ]

        print(
            f"k={k:2d} | "
            f"{len(feasible)} feasible serving candidates"
        )

    return (
        candidate_costs,
        candidate_routes,
    )