import pulp

from .config import (
    SOURCE,
    SATELLITES,
    GATEWAYS,
    TIME,
    K,
    DEMAND,
    HANDOVER_COST,
)


# =============================================================================
# EDGE ATTRIBUTE HELPERS
# =============================================================================
#
# E[k] contains edges as ordinary node pairs:
#
#     ('z', 'S1')
#     ('S1', 'S2')
#     ('S1', 'GW1')
#
# However, C and L are stored in network_data.py by time and link type:
#
#     C[k]["US"]   -> User/Satellite capacities
#     C[k]["ISL"]  -> Inter-satellite-link capacities
#     C[k]["F"]    -> Feeder-link capacities
#
#     L[k]["US"]   -> User/Satellite latencies
#     L[k]["ISL"]  -> Inter-satellite-link latencies
#     L[k]["F"]    -> Feeder-link latencies
#
# Therefore, the optimization model needs an explicit mapping from
# (u, v, k) to the appropriate matrix/vector entry.
# =============================================================================


def edge_capacity(C, u, v, k):
    """
    Return the capacity of edge (u, v) at time snapshot k.

    Capacity data structure:
        C[k]["US"]  -> shape (S,)
        C[k]["ISL"] -> shape (S, S)
        C[k]["F"]   -> shape (S, G)

    Parameters
    ----------
    C : dict
        Time-indexed capacity data.
    u : str
        Source node of the edge.
    v : str
        Destination node of the edge.
    k : int
        Time snapshot index.

    Returns
    -------
    float
        Capacity of the requested edge in Mbps.
    """

    # -------------------------------------------------------------------------
    # User -> Satellite service link
    # -------------------------------------------------------------------------
    if u == SOURCE and v in SATELLITES:
        i = SATELLITES.index(v)
        return float(C[k]["US"][i])

    # -------------------------------------------------------------------------
    # Satellite -> Satellite ISL
    # -------------------------------------------------------------------------
    if u in SATELLITES and v in SATELLITES:
        i = SATELLITES.index(u)
        j = SATELLITES.index(v)
        return float(C[k]["ISL"][i, j])

    # -------------------------------------------------------------------------
    # Satellite -> Gateway feeder link
    # -------------------------------------------------------------------------
    if u in SATELLITES and v in GATEWAYS:
        i = SATELLITES.index(u)
        g = GATEWAYS.index(v)
        return float(C[k]["F"][i, g])

    raise ValueError(
        f"Unknown edge type while retrieving capacity: "
        f"edge=({u}, {v}), time={k}"
    )


def edge_latency(L, u, v, k):
    """
    Return the latency of edge (u, v) at time snapshot k.

    Latency data structure:
        L[k]["US"]  -> shape (S,)
        L[k]["ISL"] -> shape (S, S)
        L[k]["F"]   -> shape (S, G)

    Parameters
    ----------
    L : dict
        Time-indexed latency data.
    u : str
        Source node of the edge.
    v : str
        Destination node of the edge.
    k : int
        Time snapshot index.

    Returns
    -------
    float
        Latency of the requested edge.
    """

    # -------------------------------------------------------------------------
    # User -> Satellite service link
    # -------------------------------------------------------------------------
    if u == SOURCE and v in SATELLITES:
        i = SATELLITES.index(v)
        return float(L[k]["US"][i])

    # -------------------------------------------------------------------------
    # Satellite -> Satellite ISL
    # -------------------------------------------------------------------------
    if u in SATELLITES and v in SATELLITES:
        i = SATELLITES.index(u)
        j = SATELLITES.index(v)
        return float(L[k]["ISL"][i, j])

    # -------------------------------------------------------------------------
    # Satellite -> Gateway feeder link
    # -------------------------------------------------------------------------
    if u in SATELLITES and v in GATEWAYS:
        i = SATELLITES.index(u)
        g = GATEWAYS.index(v)
        return float(L[k]["F"][i, g])

    raise ValueError(
        f"Unknown edge type while retrieving latency: "
        f"edge=({u}, {v}), time={k}"
    )


# =============================================================================
# TEMPORAL MILP MODEL
# =============================================================================

def build_model(E, C, L):
    """
    Build the one-shot temporal routing MILP.

    Parameters
    ----------
    E : dict
        Time-indexed feasible graph.

        E[k] = [
            (u, v),
            ...
        ]

    C : dict
        Time-indexed edge capacities.

        C[k]["US"]
        C[k]["ISL"]
        C[k]["F"]

    L : dict
        Time-indexed edge latencies.

        L[k]["US"]
        L[k]["ISL"]
        L[k]["F"]

    Returns
    -------
    model : pulp.LpProblem
        MILP model.

    x : dict
        Binary edge-selection variables:
            x[(u, v, k)] = 1
        if edge (u,v) is selected at snapshot k.

    y : dict
        Binary node-activation variables:
            y[(v,k)] = 1
        if node v belongs to the selected route at snapshot k.

    h : dict
        Binary handover variables:
            h[k] = 1
        if the serving satellite changes between k-1 and k.
    """

    # =========================================================================
    # MODEL
    # =========================================================================

    model = pulp.LpProblem(
        "MEO_One_Shot_Temporal_Optimization",
        pulp.LpMinimize,
    )

    # =========================================================================
    # EDGE DECISION VARIABLES
    # =========================================================================
    #
    # x[(u,v,k)] = 1 if edge (u,v) is selected at snapshot k.
    #
    # Only edges contained in E[k] are represented.
    # Thus E[k] already encodes the physically/feasibility-filtered topology.
    # =========================================================================

    x = {}

    for k in TIME:

        for u, v in E[k]:

            x[(u, v, k)] = pulp.LpVariable(
                f"x_{u}_{v}_{k}",
                cat=pulp.LpBinary,
            )

            # ---------------------------------------------------------------
            # Capacity feasibility
            # ---------------------------------------------------------------
            #
            # An edge cannot carry the requested demand if:
            #
            #       C_uv^k < DEMAND
            #
            # Such an edge is fixed to zero.
            # ---------------------------------------------------------------

            capacity = edge_capacity(C, u, v, k)

            if capacity < DEMAND:
                x[(u, v, k)].upBound = 0

    # =========================================================================
    # NODE-ACTIVATION VARIABLES
    # =========================================================================
    #
    # y[(v,k)] = 1 if node v is part of the selected route at snapshot k.
    #
    # Nodes:
    #     SOURCE
    #     SATELLITES
    #     GATEWAYS
    # =========================================================================

    y = {
        (v, k): pulp.LpVariable(
            f"y_{v}_{k}",
            cat=pulp.LpBinary,
        )
        for k in TIME
        for v in [SOURCE] + SATELLITES + GATEWAYS
    }

    # =========================================================================
    # HANDOVER VARIABLES
    # =========================================================================
    #
    # h[k] = 1 if the serving satellite changes between snapshots k-1 and k.
    #
    # h[k] >= q_i^k - q_i^(k-1)
    # h[k] >= q_i^(k-1) - q_i^k
    #
    # where
    #
    # q_i^k = x_(z,S_i)^k
    #
    # because exactly one source-to-satellite service edge is selected.
    # =========================================================================

    h = {
        k: pulp.LpVariable(
            f"h_{k}",
            cat=pulp.LpBinary,
        )
        for k in range(1, K + 1)
    }

    # =========================================================================
    # OBJECTIVE
    # =========================================================================
    #
    # Minimize:
    #
    #   sum_k sum_(u,v) L_uv^k x_uv^k
    #
    #       + H sum_k h_k
    #
    # First term:
    #     total selected-route latency over all snapshots.
    #
    # Second term:
    #     total handover penalty.
    # =========================================================================

    route_latency = pulp.lpSum(
        edge_latency(L, u, v, k) * x[(u, v, k)]
        for k in TIME
        for u, v in E[k]
    )

    handover_cost = HANDOVER_COST * pulp.lpSum(
        h[k]
        for k in range(1, K + 1)
    )

    model += route_latency + handover_cost

    # =========================================================================
    # PER-SNAPSHOT ROUTING CONSTRAINTS
    # =========================================================================

    for k in TIME:

        # ---------------------------------------------------------------------
        # SOURCE
        # ---------------------------------------------------------------------
        #
        # Source must always be active:
        #
        #     y_z^k = 1
        # ---------------------------------------------------------------------

        model += (
            y[(SOURCE, k)] == 1,
            f"source_active_{k}",
        )

        # ---------------------------------------------------------------------
        # EXACTLY ONE SERVING SATELLITE
        # ---------------------------------------------------------------------
        #
        #     sum_i x_(z,S_i)^k = 1
        #
        # The selected source edge identifies the serving satellite.
        # ---------------------------------------------------------------------

        service_edges = [
            x[(SOURCE, s, k)]
            for s in SATELLITES
            if (SOURCE, s) in E[k]
        ]

        if not service_edges:
            raise ValueError(
                f"No source-to-satellite service edge exists at "
                f"time snapshot k={k}."
            )

        model += (
            pulp.lpSum(service_edges) == 1,
            f"one_serving_satellite_{k}",
        )

        # ---------------------------------------------------------------------
        # EXACTLY ONE GATEWAY
        # ---------------------------------------------------------------------
        #
        #     sum_g y_g^k = 1
        #
        # The gateway is the termination point of the route.
        # ---------------------------------------------------------------------

        model += (
            pulp.lpSum(
                y[(g, k)]
                for g in GATEWAYS
            ) == 1,
            f"one_gateway_{k}",
        )

        # ---------------------------------------------------------------------
        # SATELLITE FLOW CONSERVATION
        # ---------------------------------------------------------------------
        #
        # For every satellite S_i:
        #
        #     sum_u x_(u,i)^k = y_i^k
        #
        #     sum_v x_(i,v)^k = y_i^k
        #
        # Therefore an active satellite has exactly:
        #
        #     one incoming edge
        #     one outgoing edge
        #
        # while an inactive satellite has:
        #
        #     zero incoming edges
        #     zero outgoing edges
        #
        # This produces a single source-to-gateway route under the
        # positive-latency objective.
        # ---------------------------------------------------------------------

        for s in SATELLITES:

            incoming = [
                x[(u, v, k)]
                for u, v in E[k]
                if v == s
            ]

            outgoing = [
                x[(u, v, k)]
                for u, v in E[k]
                if u == s
            ]

            model += (
                pulp.lpSum(incoming) == y[(s, k)],
                f"satellite_inflow_{s}_{k}",
            )

            model += (
                pulp.lpSum(outgoing) == y[(s, k)],
                f"satellite_outflow_{s}_{k}",
            )

        # ---------------------------------------------------------------------
        # GATEWAY TERMINATION
        # ---------------------------------------------------------------------
        #
        # For each gateway GW_g:
        #
        #     sum_i x_(S_i,GW_g)^k = y_GWg^k
        #
        # and
        #
        #     sum_v x_(GW_g,v)^k = 0
        #
        # Therefore the selected gateway terminates the route.
        # ---------------------------------------------------------------------

        for g in GATEWAYS:

            incoming = [
                x[(u, v, k)]
                for u, v in E[k]
                if v == g
            ]

            outgoing = [
                x[(u, v, k)]
                for u, v in E[k]
                if u == g
            ]

            model += (
                pulp.lpSum(incoming) == y[(g, k)],
                f"gateway_inflow_{g}_{k}",
            )

            model += (
                pulp.lpSum(outgoing) == 0,
                f"gateway_no_outflow_{g}_{k}",
            )

    # =========================================================================
    # HANDOVER CONSTRAINTS
    # =========================================================================
    #
    # Define:
    #
    #     q_i^k = x_(z,S_i)^k
    #
    # Since exactly one serving satellite is selected at every snapshot,
    # the following absolute-difference formulation gives:
    #
    #     h_k = 0  -> same serving satellite
    #     h_k = 1  -> serving satellite changed
    #
    # Constraints:
    #
    #     h_k >= q_i^k - q_i^(k-1)
    #
    #     h_k >= q_i^(k-1) - q_i^k
    #
    # for every satellite i.
    # =========================================================================

    for k in range(1, K + 1):

        for s in SATELLITES:

            now = x.get(
                (SOURCE, s, k),
                0,
            )

            previous = x.get(
                (SOURCE, s, k - 1),
                0,
            )

            model += (
                h[k] >= now - previous,
                f"handover_positive_{s}_{k}",
            )

            model += (
                h[k] >= previous - now,
                f"handover_negative_{s}_{k}",
            )

    return model, x, y, h


# =============================================================================
# SOLVER
# =============================================================================

def solve_model(model):
    """
    Solve the temporal MILP using CBC.

    Parameters
    ----------
    model : pulp.LpProblem
        MILP model generated by build_model().

    Raises
    ------
    RuntimeError
        If CBC does not return an optimal solution.
    """

    status = model.solve(
        pulp.PULP_CBC_CMD(msg=True)
    )

    if status != pulp.LpStatusOptimal:
        raise RuntimeError(
            f"Solver status: {pulp.LpStatus[status]}"
        )
