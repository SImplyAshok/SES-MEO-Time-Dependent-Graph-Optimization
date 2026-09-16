from pathlib import Path
import pickle

import numpy as np

# Support both:
#   1. package execution: python -m src.main
#   2. direct execution from this directory
try:
    from .config import (
        SOURCE,
        SATELLITES,
        GATEWAYS,
        TIME,
        DEMAND,
        NUM_PATHS,
        FIXED_SERVING_SATELLITE,
    )
    from .optimization import (
        solve_connectivity_feasibility,
    )
except ImportError:
    from config import (
        SOURCE,
        SATELLITES,
        GATEWAYS,
        TIME,
        DEMAND,
        NUM_PATHS,
        FIXED_SERVING_SATELLITE,
    )
    from optimization import (
        solve_connectivity_feasibility,
    )


# ============================================================
# NETWORK / GEOMETRY PARAMETERS
# ============================================================

NUM_SATELLITES = len(SATELLITES)
NUM_GATEWAYS = len(GATEWAYS)

# Number of satellites per orbital plane
SATS_PER_PLANE = 6


# ============================================================
# SATELLITE INDEX / ORBITAL PLANE HELPERS
# ============================================================

def satellite_plane_index(sat_index):
    """
    Return the orbital plane index of a satellite.

    Satellite numbering:
        S1-S6   -> Plane 1
        S7-S12  -> Plane 2
        S13-S18 -> Plane 3
    """
    return sat_index // SATS_PER_PLANE


def satellite_local_index(sat_index):
    """
    Return the satellite's local position within its orbital plane.

    Example:
        S1 -> 0
        S6 -> 5
        S7 -> 0
        S12 -> 5
    """
    return sat_index % SATS_PER_PLANE


# ============================================================
# CANDIDATE ISL TOPOLOGY
# ============================================================

def build_candidate_isl_topology():
    """
    Construct the candidate satellite-to-satellite ISL topology.

    The topology is based on the orbital-plane structure:

    1. Same-plane neighbours:
         local index difference = 1 or 5

       This creates the circular intra-plane connectivity:
         S1 <-> S2
         S2 <-> S3
         ...
         S5 <-> S6
         S6 <-> S1

    2. Inter-plane connectivity:
         satellites having the same local index are connected.

       Example:
         S1 <-> S7
         S7 <-> S13
         S1 <-> S13

    IMPORTANT:
    This function generates CANDIDATE physical links.

    It does NOT decide whether a link is available at a
    particular time snapshot. Time-dependent availability
    is represented separately by I_bar.
    """

    isl = []
    isl_set = set()

    for i in range(NUM_SATELLITES):

        pi = satellite_plane_index(i)
        ii = satellite_local_index(i)

        for j in range(NUM_SATELLITES):

            if i == j:
                continue

            pj = satellite_plane_index(j)
            jj = satellite_local_index(j)

            # ------------------------------------------------
            # Same-plane neighbour links
            # ------------------------------------------------
            if (
                pi == pj
                and abs(ii - jj) in (1, 5)
            ):
                edge = (SATELLITES[i], SATELLITES[j])

                if edge not in isl_set:
                    isl.append(edge)
                    isl_set.add(edge)

            # ------------------------------------------------
            # Inter-plane links
            # ------------------------------------------------
            elif (
                pi != pj
                and ii == jj
            ):
                edge = (SATELLITES[i], SATELLITES[j])

                if edge not in isl_set:
                    isl.append(edge)
                    isl_set.add(edge)

    return isl


# ============================================================
# GEOMETRY / VISIBILITY
# ============================================================
def compute_geometry_availability(k):
    """
    Compute physical availability for snapshot k.

    TEST / DEBUG MODE:

    1. The fixed serving satellite is ALWAYS service-visible.
    2. Additional service satellites follow a deterministic
       time-varying pattern.
    3. Every satellite can establish an ISL with every other
       satellite.
    4. Feeder links follow a deterministic time-varying pattern,
       but at least one gateway is ALWAYS reachable from the
       fixed serving satellite.

    This guarantees that the fixed serving satellite is not
    excluded by the physical availability model while still
    allowing the network topology to change with time.
    """

    # --------------------------------------------------------
    # Allocate availability matrices
    # --------------------------------------------------------

    U_bar = np.zeros(
        NUM_SATELLITES,
        dtype=int,
    )

    I_bar = np.zeros(
        (NUM_SATELLITES, NUM_SATELLITES),
        dtype=int,
    )

    F_bar = np.zeros(
        (NUM_SATELLITES, NUM_GATEWAYS),
        dtype=int,
    )

    # --------------------------------------------------------
    # Fixed serving satellite
    # --------------------------------------------------------
    #
    # FIXED_SERVING_SATELLITE is the satellite forced by the
    # Stage-1 optimization at k=0.
    #
    # We make it service-visible at EVERY timestamp.
    # --------------------------------------------------------

    fixed_service_index = FIXED_SERVING_SATELLITE
    
    if k == 0:
        U_bar[fixed_service_index] = 1

    # --------------------------------------------------------
    # Additional service-satellite pattern
    # --------------------------------------------------------
    #
    # These satellites change with time.
    #
    # The fixed serving satellite is added independently above,
    # so it remains available even when it is not in this pattern.
    # --------------------------------------------------------

    service_pattern = [
        0,
        4,
        8,
        17,
        5,
        10,
        14,
        2,
        7,
        12,
        16,
        1,
        6,
        11,
        15,
        3,
        9,
        13,
        0,
        4,
        8,
    ]

    primary_service = service_pattern[
        k % len(service_pattern)
    ]

    # Primary service satellite
    U_bar[primary_service] = 1

    # Additional service-visible satellite
    if NUM_SATELLITES > 1:
        U_bar[
            (primary_service + 4) % NUM_SATELLITES
        ] = 1

    # Additional service-visible satellite
    if NUM_SATELLITES > 2:
        U_bar[
            (primary_service + 13) % NUM_SATELLITES
        ] = 1
    '''
    # --------------------------------------------------------
    # ISL availability
    # --------------------------------------------------------
    #
    # TEST MODE:
    #
    # Every satellite can establish an ISL with every other
    # satellite at every timestamp.
    #
    # Therefore:
    #
    #     I_bar[i,j] = 1,  i != j
    #
    # The optimization variables I[i,j] still decide which
    # available ISLs are actually selected.
    # --------------------------------------------------------

    for i in range(NUM_SATELLITES):

        for j in range(NUM_SATELLITES):

            if i == j:
                continue

            I_bar[i, j] = 1
    '''
        # --------------------------------------------------------
    # ISL availability
    # --------------------------------------------------------
    #
    # TEST MODE:
    #
    # Each satellite is connected to:
    #
    #   1. Two neighbours in the SAME orbital plane.
    #      The satellites form a ring within each plane.
    #
    #   2. One satellite at the SAME local orbital position
    #      in each of the OTHER orbital planes.
    #
    # With 3 planes and 6 satellites per plane, each satellite
    # has:
    #
    #       2 same-plane neighbours
    #       2 inter-plane neighbours
    #
    # Therefore, each satellite has 4 available ISLs.
    #
    # Example:
    #
    #   S1 <-> S2
    #   S1 <-> S6
    #   S1 <-> S7
    #   S1 <-> S13
    #
    # The optimization variables I[i,j] still decide which
    # available ISLs are actually selected.
    # --------------------------------------------------------

    for i in range(NUM_SATELLITES):

        plane_i = satellite_plane_index(i)
        local_i = satellite_local_index(i)

        for j in range(NUM_SATELLITES):

            if i == j:
                continue

            plane_j = satellite_plane_index(j)
            local_j = satellite_local_index(j)

            # ------------------------------------------------
            # Same-plane neighbours
            # ------------------------------------------------
            #
            # Connect to the previous and next satellite in
            # the circular orbital plane.
            #
            if plane_i == plane_j:

                if (local_j - local_i) % SATS_PER_PLANE in (
                    1,
                    SATS_PER_PLANE - 1,
                ):

                    I_bar[i, j] = 1

            # ------------------------------------------------
            # Inter-plane connectivity
            # ------------------------------------------------
            #
            # Connect satellites having the same local
            # position in different orbital planes.
            #
            elif local_i == local_j:

                I_bar[i, j] = 1

    # --------------------------------------------------------
    # Feeder availability
    # --------------------------------------------------------
    #
    # Keep a deterministic time-varying feeder pattern.
    # --------------------------------------------------------

    for i in range(NUM_SATELLITES):

        for g in range(NUM_GATEWAYS):

            if ((i + 2 * g + k) % 5) != 0:
                F_bar[i, g] = 1

    # --------------------------------------------------------
    # Guarantee feeder connectivity for the fixed satellite
    # --------------------------------------------------------
    #
    # At least one gateway is always reachable directly from
    # the fixed serving satellite.
    # --------------------------------------------------------

    if NUM_GATEWAYS > 0:

        F_bar[fixed_service_index, 0] = 1

    return U_bar, I_bar, F_bar

# ============================================================
# EDGE ATTRIBUTES
# ============================================================

def build_edge_attributes(k):
    """
    Build capacity and latency attributes for snapshot k.

    Returns
    -------
    C_US :
        Source-to-satellite capacities.

    C_ISL :
        Satellite-to-satellite capacities.

    C_F :
        Satellite-to-gateway capacities.

    L_US :
        Source-to-satellite latency.

    L_ISL :
        ISL latency.

    L_F :
        Feeder latency.

    NOTE:
    Availability and capacity are intentionally kept
    separate.

        availability = physical existence / visibility

        capacity     = whether the link can carry D Mbps

        latency      = routing cost
    """

    # --------------------------------------------------------
    # Capacities
    # --------------------------------------------------------

    C_US = np.full(
        NUM_SATELLITES,
        500.0,
        dtype=float,
    )

    C_ISL = np.zeros(
        (NUM_SATELLITES, NUM_SATELLITES),
        dtype=float,
    )

    C_F = np.full(
        (NUM_SATELLITES, NUM_GATEWAYS),
        500.0,
        dtype=float,
    )

    # ISL capacity
    for i in range(NUM_SATELLITES):

        for j in range(NUM_SATELLITES):

            if i != j:
                C_ISL[i, j] = 500.0

        # --------------------------------------------------------
    # Latencies
    # --------------------------------------------------------
    #
    # Toy latency model with physically meaningful ranges.
    #
    # The values are generated pseudo-randomly so that:
    #
    #   - different edges have different latencies
    #   - latency changes from one time snapshot to another
    #   - results remain reproducible
    #
    # Units: milliseconds (ms)
    #
    # Approximate ranges:
    #
    #   User/source -> satellite : 4   - 8  ms
    #   Same-plane ISL           : 2   - 4  ms
    #   Inter-plane ISL           : 3   - 6  ms
    #   Satellite -> gateway     : 6   - 15 ms
    #
    # These ranges are only a toy abstraction. In the final
    # model, latency should preferably be derived from the
    # time-dependent geometric distance:
    #
    #       L_ij(k) = d_ij(k) / c
    #
    # together with any processing/queuing components.
    # --------------------------------------------------------

    rng = np.random.default_rng(1000 + k)

    # --------------------------------------------------------
    # Source -> satellite latency
    # --------------------------------------------------------
    #
    # Shorter range because this represents the access/service
    # link from the source/terminal side to a visible satellite.
    #
    L_US = rng.uniform(
        4.0,
        12.0,
        size=NUM_SATELLITES,
    )

    # --------------------------------------------------------
    # Satellite -> satellite ISL latency
    # --------------------------------------------------------
    #
    # Same-plane neighbouring satellites are relatively close.
    # Inter-plane satellites are assigned a somewhat larger
    # propagation delay.
    #
    L_ISL = np.zeros(
        (NUM_SATELLITES, NUM_SATELLITES),
        dtype=float,
    )

    for i in range(NUM_SATELLITES):

        for j in range(NUM_SATELLITES):

            if i == j:
                continue

            pi = satellite_plane_index(i)
            pj = satellite_plane_index(j)

            if pi == pj:

                # Immediate neighbour within the same
                # orbital plane.
                L_ISL[i, j] = rng.uniform(
                    1.0,
                    8.0,
                )

            else:

                # Cross-plane / inter-plane ISL.
                L_ISL[i, j] = rng.uniform(
                    1.0,
                    8.0,
                )

    # --------------------------------------------------------
    # Satellite -> gateway feeder latency
    # --------------------------------------------------------
    #
    # Gateway links have a wider range because the satellite
    # to gateway geometry can vary significantly.
    #
    L_F = rng.uniform(
        6.0,
        15.0,
        size=(NUM_SATELLITES, NUM_GATEWAYS),
    )
    # --------------------------------------------------------
    # ISL propagation / routing latency
    #
    # Toy deterministic values.
    #
    # In the real model:
    #
    #       L_ISL[i,j,k] = d_ij(k) / c
    #
    # or a more complete latency model.
    # --------------------------------------------------------

    for i in range(NUM_SATELLITES):

        for j in range(NUM_SATELLITES):

            if i != j:

                pi = satellite_plane_index(i)
                pj = satellite_plane_index(j)

                if pi == pj:
                    L_ISL[i, j] = 2.0
                else:
                    L_ISL[i, j] = 3.0

    return (
        C_US,
        C_ISL,
        C_F,
        L_US,
        L_ISL,
        L_F,
    )


# ============================================================
# PATH ENUMERATION
# ============================================================

def enumerate_source_gateway_paths(
    U,
    I,
    F,
    L_US,
    L_ISL,
    L_F,
    max_paths=None,
):
    """
    Enumerate simple source-to-gateway paths from a selected
    Stage-1 feasibility topology.

    Parameters
    ----------
    U :
        Selected service links / service-satellite indicators.

    I :
        Selected ISL matrix.

    F :
        Selected feeder matrix.

    L_US :
        Source-to-satellite latency.

    L_ISL :
        ISL latency matrix.

    L_F :
        Feeder latency matrix.

    max_paths :
        Optional maximum number of paths to report.

    Returns
    -------
    paths :
        List of dictionaries describing complete
        source-to-gateway paths.

    IMPORTANT:
    This function is ONLY a reporting / diagnostic function.

    It does not alter the optimization result.
    """

    selected_service = [
        SATELLITES[i]
        for i in range(NUM_SATELLITES)
        if U[i] > 0.5
    ]

    selected_isls = [
        (
            SATELLITES[i],
            SATELLITES[j],
        )
        for i in range(NUM_SATELLITES)
        for j in range(NUM_SATELLITES)
        if I[i, j] > 0.5
    ]

    selected_feeders = [
        (
            SATELLITES[i],
            GATEWAYS[g],
        )
        for i in range(NUM_SATELLITES)
        for g in range(NUM_GATEWAYS)
        if F[i, g] > 0.5
    ]

    # --------------------------------------------------------
    # Build adjacency list
    # --------------------------------------------------------

    adjacency = {
        node: []
        for node in SATELLITES
    }

    for si, sj in selected_isls:
        adjacency[si].append(sj)

    # --------------------------------------------------------
    # DFS
    # --------------------------------------------------------

    paths = []

    def dfs(
        current,
        path,
        visited,
        latency,
        isl_hops,
    ):

        if max_paths is not None and len(paths) >= max_paths:
            return

        # ----------------------------------------------------
        # Gateway termination
        # ----------------------------------------------------

        for si, gw in selected_feeders:

            if si != current:
                continue

            feeder_index = GATEWAYS.index(gw)

            total_latency = (
                latency
                + L_F[
                    SATELLITES.index(si),
                    feeder_index,
                ]
            )

            paths.append(
                {
                    "nodes": path + [gw],
                    "latency": total_latency,
                    "isl_hops": isl_hops,
                }
            )

        # ----------------------------------------------------
        # Continue through ISLs
        # ----------------------------------------------------

        for next_sat in adjacency.get(current, []):

            if next_sat in visited:
                continue

            i = SATELLITES.index(current)
            j = SATELLITES.index(next_sat)

            dfs(
                next_sat,
                path + [next_sat],
                visited | {next_sat},
                latency + L_ISL[i, j],
                isl_hops + 1,
            )

    # --------------------------------------------------------
    # Start DFS from every selected service satellite
    # --------------------------------------------------------

    for service_satellite in selected_service:

        i = SATELLITES.index(service_satellite)

        dfs(
            service_satellite,
            [SOURCE, service_satellite],
            {service_satellite},
            L_US[i],
            0,
        )

    return paths


# ============================================================
# STAGE-1 RESULT REPORTING
# ============================================================

def report_stage1_result(
    k,
    feasibility_result,
    L_US,
    L_ISL,
    L_F,
):
    """
    Print a human-readable Stage-1 feasibility result.

    The reporting distinguishes:

        feasible topology
        selected service links
        selected ISLs
        selected feeders
        complete source-to-gateway paths

    """

    if not feasibility_result.get("feasible", False):

        print(
            f"[network_data]   k={k}: "
            "Stage 1 INFEASIBLE"
        )

        return

    U = feasibility_result["U"]
    I = feasibility_result["I"]
    F = feasibility_result["F"]

    # --------------------------------------------------------
    # Selected service links
    # --------------------------------------------------------

    selected_service = [
        SATELLITES[i]
        for i in range(NUM_SATELLITES)
        if U[i] > 0.5
    ]

    # --------------------------------------------------------
    # Selected ISLs
    # --------------------------------------------------------

    selected_isls = [
        (
            SATELLITES[i],
            SATELLITES[j],
        )
        for i in range(NUM_SATELLITES)
        for j in range(NUM_SATELLITES)
        if I[i, j] > 0.5
    ]

    # --------------------------------------------------------
    # Selected feeders
    # --------------------------------------------------------

    selected_feeders = [
        (
            SATELLITES[i],
            GATEWAYS[g],
        )
        for i in range(NUM_SATELLITES)
        for g in range(NUM_GATEWAYS)
        if F[i, g] > 0.5
    ]

    print(
        f"[network_data]     feasible service = "
        f"{selected_service}"
    )

    print(
        f"[network_data]     feasible ISLs = "
        f"{selected_isls}"
    )

    print(
        f"[network_data]     feasible feeders = "
        f"{selected_feeders}"
    )

    # --------------------------------------------------------
    # Stage-1 objective
    # --------------------------------------------------------

    objective_value = feasibility_result.get(
        "objective_value",
        None,
    )

    if objective_value is not None:

        print(
            f"[network_data]     Stage-1 objective = "
            f"{objective_value:.3f}"
        )

    # --------------------------------------------------------
    # Enumerate complete paths
    # --------------------------------------------------------

    paths = enumerate_source_gateway_paths(
        U,
        I,
        F,
        L_US,
        L_ISL,
        L_F,
    )

    print(
        f"[network_data]     "
        f"End-to-end source-to-gateway paths = "
        f"{len(paths)}"
    )

    # --------------------------------------------------------
    # Print every complete path
    # --------------------------------------------------------

    for p_index, path_info in enumerate(paths, start=1):

        nodes = path_info["nodes"]
        latency = path_info["latency"]
        isl_hops = path_info["isl_hops"]

        path_string = " -> ".join(nodes)

        print(
            f"[network_data]       "
            f"path {p_index}: "
            f"{path_string}"
        )

        print(
            f"[network_data]              "
            f"ISL hops = {isl_hops}, "
            f"latency = {latency:.3f}"
        )


# ============================================================
# NETWORK BUILD
# ============================================================

def build_network(reuse_stored_feasibility=None):
    """
    Build the time-dependent MEO network and solve Stage 1.

    Parameters
    ----------
    reuse_stored_feasibility :
        Controls Stage-1 feasibility reuse.

        None:
            If a stored Stage-1 solution exists, ask the user
            whether it should be retrieved.

        True:
            Always retrieve stored Stage-1 feasibility.

        False:
            Always regenerate Stage-1 feasibility.

    Returns
    -------
    network_data :
        Dictionary containing all time-dependent network data.

    feasibility_results :
        Dictionary containing the Stage-1 feasibility solution
        at every time snapshot.

    """

    # ========================================================
    # STAGE-1 FEASIBILITY STORAGE
    # ========================================================

    '''feasibility_store = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "stage1_feasibility.pkl"
    )'''

    # --------------------------------------------------------
    # Determine whether to retrieve stored feasibility
    # --------------------------------------------------------

    use_stored_feasibility = False

    if reuse_stored_feasibility is True:

        if feasibility_store.exists():

            use_stored_feasibility = True

        else:

            print(
                "[network_data] Requested stored Stage-1 "
                "feasibility, but no stored file exists."
            )

            print(
                "[network_data] Stage 1 will be regenerated."
            )

    elif reuse_stored_feasibility is False:

        use_stored_feasibility = False

    else:

        # ----------------------------------------------------
        # Automatic interactive behaviour
        # ----------------------------------------------------

        if feasibility_store.exists():

            answer = input(
                "[network_data] Stored Stage-1 feasibility "
                "found. Retrieve stored values? [Y/n]: "
            ).strip().lower()

            if answer in ("", "y", "yes"):

                use_stored_feasibility = True

            else:

                use_stored_feasibility = False

        else:

            print(
                "[network_data] No stored Stage-1 feasibility "
                "found. Stage 1 will be generated."
            )

    # ========================================================
    # LOAD STORED FEASIBILITY
    # ========================================================

    stored_feasibility = None

    if use_stored_feasibility:

        print(
            "[network_data] Loading stored Stage-1 "
            "feasibility..."
        )

        try:

            with feasibility_store.open("rb") as f:

                stored_feasibility = pickle.load(f)

        except Exception as exc:

            print(
                "[network_data] ERROR while loading stored "
                f"Stage-1 feasibility: {exc}"
            )

            print(
                "[network_data] Falling back to Stage-1 "
                "regeneration."
            )

            use_stored_feasibility = False
            stored_feasibility = None

        # ----------------------------------------------------
        # Validate stored snapshots
        # ----------------------------------------------------

        if (
            use_stored_feasibility
            and stored_feasibility is not None
        ):

            missing_snapshots = [
                k
                for k in TIME
                if k not in stored_feasibility
            ]

            if missing_snapshots:

                print(
                    "[network_data] Stored Stage-1 file is "
                    "incomplete."
                )

                print(
                    "[network_data] Missing snapshots: "
                    f"{missing_snapshots}"
                )

                print(
                    "[network_data] Falling back to "
                    "regeneration."
                )

                use_stored_feasibility = False
                stored_feasibility = None

    # ========================================================
    # DATA CONTAINERS
    # ========================================================

    E = {}
    C = {}
    L = {}

    availability = {}

    feasibility_results = {}

    # ========================================================
    # TIME SNAPSHOT LOOP
    # ========================================================

    for k in TIME:

        print()
        print("=" * 70)
        print(
            f"[network_data] Snapshot k={k}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Geometry -> physical availability
        # ----------------------------------------------------

        U_bar, I_bar, F_bar = (
            compute_geometry_availability(k)
        )

        availability[k] = {
            "U_bar": U_bar,
            "I_bar": I_bar,
            "F_bar": F_bar,
        }

        # ----------------------------------------------------
        # Edge capacities and latencies
        # ----------------------------------------------------

        (
            C_US,
            C_ISL,
            C_F,
            L_US,
            L_ISL,
            L_F,
        ) = build_edge_attributes(k)

        # ----------------------------------------------------
        # Store attributes
        # ----------------------------------------------------

        C[k] = {
            "US": C_US,
            "ISL": C_ISL,
            "F": C_F,
        }

        L[k] = {
            "US": L_US,
            "ISL": L_ISL,
            "F": L_F,
        }

        # ====================================================
        # STAGE 1
        # ====================================================

        if (
            use_stored_feasibility
            and stored_feasibility is not None
        ):

            # ------------------------------------------------
            # Retrieve previously solved Stage-1 result.
            #
            # IMPORTANT:
            #
            # We retrieve the ACTUAL optimization result:
            #
            #       U_fes
            #       I_fes
            #       F_fes
            #
            # rather than simply copying:
            #
            #       U_bar
            #       I_bar
            #       F_bar
            #
            # Therefore Stage 2 sees exactly the same feasible
            # topology that Stage 1 produced originally.
            # ------------------------------------------------

            stored = stored_feasibility[k]

            feasibility_result = stored.copy()

            feasibility_results[k] = (
                feasibility_result
            )

            print(
                f"[network_data]   k={k}: "
                "Stage 1 -> RETRIEVED FROM STORAGE"
            )

        else:

            # ------------------------------------------------
            # Solve Stage 1 from geometry-derived availability
            # ------------------------------------------------
            if k == 0:
                serving_satellite = FIXED_SERVING_SATELLITE
            else:
                serving_satellite = None

            feasibility_result = (
                solve_connectivity_feasibility(
                    U_bar=U_bar,
                    I_bar=I_bar,
                    F_bar=F_bar,
                    C_US=C_US,
                    C_ISL=C_ISL,
                    C_F=C_F,
                    demand=DEMAND,
                    num_paths=NUM_PATHS,
                    L_US=L_US,
                    L_ISL=L_ISL,
                    L_F=L_F,
                    fixed_serving_satellite = serving_satellite
                )
            )

            feasibility_results[k] = (
                feasibility_result
            )

            if feasibility_result.get(
                "feasible",
                False,
            ):

                print(
                    f"[network_data]   k={k}: "
                    "Stage 1 FEASIBLE"
                )

            else:

                print(
                    f"[network_data]   k={k}: "
                    "Stage 1 INFEASIBLE"
                )

        # ----------------------------------------------------
        # Report Stage-1 result
        # ----------------------------------------------------

        report_stage1_result(
            k,
            feasibility_result,
            L_US,
            L_ISL,
            L_F,
        )

    # ========================================================
    # SAVE NEWLY GENERATED STAGE-1 FEASIBILITY
    # ========================================================

    if not use_stored_feasibility:

        # ----------------------------------------------------
        # Ensure results directory exists
        # ----------------------------------------------------

        feasibility_store.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Store the complete Stage-1 results.
        #
        # Each snapshot contains the actual optimized:
        #
        #       U
        #       I
        #       F
        #
        # together with feasibility status and objective.
        # ----------------------------------------------------

        with feasibility_store.open("wb") as f:

            pickle.dump(
                feasibility_results,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print()
        print(
            "[network_data] Stage-1 feasibility results "
            "saved to:"
        )

        print(
            f"[network_data]   {feasibility_store}"
        )

    else:

        print()
        print(
            "[network_data] Stage-1 feasibility results "
            "were retrieved from storage."
        )

    # ========================================================
    # BUILD EDGE DICTIONARY
    # ========================================================

    for k in TIME:

        U_bar = availability[k]["U_bar"]
        I_bar = availability[k]["I_bar"]
        F_bar = availability[k]["F_bar"]

        edges = []

        # ----------------------------------------------------
        # Source -> satellite edges
        # ----------------------------------------------------

        for i, satellite in enumerate(SATELLITES):

            if U_bar[i] == 1:

                edges.append(
                    (
                        SOURCE,
                        satellite,
                    )
                )

        # ----------------------------------------------------
        # Satellite -> satellite edges
        # ----------------------------------------------------

        for i, si in enumerate(SATELLITES):

            for j, sj in enumerate(SATELLITES):

                if i == j:
                    continue

                if I_bar[i, j] == 1:

                    edges.append(
                        (
                            si,
                            sj,
                        )
                    )

        # ----------------------------------------------------
        # Satellite -> gateway edges
        # ----------------------------------------------------

        for i, satellite in enumerate(SATELLITES):

            for g, gateway in enumerate(GATEWAYS):

                if F_bar[i, g] == 1:

                    edges.append(
                        (
                            satellite,
                            gateway,
                        )
                    )

        E[k] = edges

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("[network_data] NETWORK BUILD COMPLETE")
    print("=" * 70)

    if use_stored_feasibility:

        print(
            "[network_data] Stage 1 source: STORED RESULTS"
        )

    else:

        print(
            "[network_data] Stage 1 source: NEW OPTIMIZATION"
        )

    print(
        f"[network_data] Number of snapshots: "
        f"{len(TIME)}"
    )

    feasible_count = sum(
        1
        for k in TIME
        if feasibility_results[k].get(
            "feasible",
            False,
        )
    )

    print(
        f"[network_data] Feasible snapshots: "
        f"{feasible_count}/{len(TIME)}"
    )

    print("=" * 70)

    # ========================================================
    # RETURN ALL NETWORK DATA
    # ========================================================

    network_data = {
        "E": E,
        "C": C,
        "L": L,
        "availability": availability,
    }

    return (
        network_data,
        feasibility_results,
    )


# ============================================================
# OPTIONAL DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    network_data, feasibility_results = (
        build_network()
    )

    print()
    print(
        "[network_data] Direct execution completed."
    )