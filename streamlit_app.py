# streamlit_app.py
import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# =========================
# Globals / constants
# =========================
BIG_M = 10000
EPSILON = 0.001
MIP_TIME_LIMIT_SEC = 45          # MIP solver wall time
NO_PROGRESS_MAX = 3              # stop if we can't reduce leftovers
CLONES_PER_TYPE = 5              # default clones per vehicle type
PER_TYPE_CLONES = {}             # e.g., {"40ft CS": 20} to override per type
debug_log = []                   # UI-visible debug lines

def dbg(msg: str):
    print(msg)
    debug_log.append(msg)

# =========================
# Optional dependency: rectpack (MaxRects)
# =========================
try:
    from rectpack import newPacker
    from rectpack.maxrects import MaxRectsBssf, MaxRectsBaf
    from rectpack.packer import SORT_AREA
    HAS_RECTPACK = True
except Exception:
    HAS_RECTPACK = False
    MaxRectsBssf = MaxRectsBaf = None
    SORT_AREA = None

# =========================
# Streamlit config & sidebar
# =========================
st.set_page_config(page_title="Truck Optimiser", layout="wide")
st.markdown(
    """
    <div style="background-color: #00008B; padding: 20px 10px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">
        <h1 style="color: #FFFFFF; margin-bottom: 5px;">🚚 Truck Optimiser</h1>
        <p style="color: #FFFFFF; font-size: 18px;">Optimise inventory placement in trucks for cost-effective and efficient booking</p>
    </div>
    """,
    unsafe_allow_html=True
)

show_debug = st.sidebar.checkbox("Show debug panel", value=False)

pack_options = ["MaxRects (BSSF)", "MaxRects (BAF)", "Shelf (Greedy)"]
default_idx = (0 if HAS_RECTPACK else 2)
PACK_MODE = st.sidebar.selectbox("Packing algorithm", pack_options, index=default_idx)

if ("MaxRects" in PACK_MODE) and (not HAS_RECTPACK):
    st.sidebar.warning("rectpack not installed — falling back to Shelf (Greedy). Add `rectpack` to requirements.txt.")

# =========================
# Vehicle TYPES (base specs; clones are created below with a FIXED count)
# =========================
vehicle_types = [
    ("Small van",        1.5,  1.2,   1.1,   360,   1.8,    100),
    ("Medium wheel base",3.0,  1.2,   1.9,  1400,   3.6,    130),
    ("Sprinter van",     4.2,  1.2,   1.75,  950,   5.04,   135),
    ("luton van",        4.0,  2.0,   2.0,  1000,   8.0,    160),
    ("7.5T CS",          6.0,  2.88,  2.2,  2600,  17.28,   150),
    ("18T CS",           7.3,  2.88,  2.3,  9800,  21.024,  175),
    ("40ft CS",         13.5,  3.0,   3.0, 28000,  40.5,    185),
    ("20ft FB",          7.3,  2.4,   300, 10500,  17.52,   180),
    ("40ft FB",         13.5,  2.4,   300, 30000,  32.4,    190),
    ("40T Low Loader",  13.5,  2.4,   300, 30000,  32.4,    195),
]

# =========================
# Inputs
# =========================
st.header("Inventory Inputs")
weights, lengths, widths, heights = [], [], [], []

num_individual = st.number_input("Number of Individual Inventory", min_value=0, max_value=200, value=0)
cols = st.columns(5)
for i in range(num_individual):
    with cols[0]:
        weights.append(st.number_input(f"Weight {i+1} (kg)", key=f"wt_{i}", value=100.0))
    with cols[1]:
        lengths.append(st.number_input(f"Length {i+1} (m)", key=f"len_{i}", value=1.0))
    with cols[2]:
        widths.append(st.number_input(f"Width {i+1} (m)", key=f"wid_{i}", value=1.0))
    with cols[3]:
        heights.append(st.number_input(f"Height (m)", key=f"hei_{i}", value=1.0))
    with cols[4]:
        st.markdown("&nbsp;")

st.markdown("---")
st.subheader("Bulk Inventory Entries")
bulk_entries = st.number_input("Number of Bulk Inventory Types", min_value=0, max_value=20, value=0)

for i in range(bulk_entries):
    st.markdown(f"**Bulk Parcel Type {i+1}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        quantity = st.number_input(f"Quantity", min_value=1, value=1, key=f"qty_{i}")
    with c2:
        weight = st.number_input("Weight (kg)", value=100.0, key=f"b_wt_{i}")
    with c3:
        length = st.number_input("Length (m)", value=1.0, key=f"b_len_{i}")
    with c4:
        width = st.number_input("Width (m)", value=1.0, key=f"b_wid_{i}")
    with c5:
        height = st.number_input("Height (m)", value=1.0, key=f"b_hei_{i}")

    for _ in range(quantity):
        weights.append(weight)
        lengths.append(length)
        widths.append(width)
        heights.append(height)

areas = [lengths[i] * widths[i] for i in range(len(weights))]
total_weight = sum(weights)
total_area = sum(areas)

# =========================
# Build vehicles with a FIXED number of clones per type (with per-type override)
# =========================
def build_fixed_vehicles(clones_per_type=CLONES_PER_TYPE, per_type_override=None):
    per_type_override = per_type_override or {}
    vehicles = {}
    vehicle_groups = {}  # base name -> [clone names in order]
    for name, l, w, h, wt_cap, ar_cap, cost in vehicle_types:
        n_clones = int(per_type_override.get(name, clones_per_type))
        names = []
        for i in range(1, n_clones + 1):
            nm = f"{name}{i if i > 1 else ''}"
            vehicles[nm] = {
                "base": name,
                "max_length": l, "max_width": w, "max_height": h,
                "max_weight": wt_cap, "max_area": ar_cap, "cost": cost
            }
            names.append(nm)
        vehicle_groups[name] = names
    dbg("Clone caps per type (fixed): " + ", ".join(
        f"{base}×{len(clones)}" for base, clones in vehicle_groups.items()
    ))
    return vehicles, vehicle_groups

vehicles, vehicle_groups = build_fixed_vehicles(per_type_override=PER_TYPE_CLONES)

# =========================
# Stage A: Feasible vehicle mapping (per-parcel; enforces height)
# =========================
parcel_feasible_vehicles = {}
invalid_parcels = []

dbg("\n=== FEASIBILITY CHECK DEBUG ===")
for i in range(len(weights)):
    fits_in = []
    dbg(f"\nParcel {i} -> weight={weights[i]}, dims=({lengths[i]} x {widths[i]} x {heights[i]}), area={areas[i]:.2f}")
    for truck_name, v in vehicles.items():
        reason = []
        if weights[i] > v["max_weight"]:
            reason.append(f"FAIL weight {weights[i]} > {v['max_weight']}")
        if areas[i] > v["max_area"]:
            reason.append(f"FAIL area {areas[i]:.2f} > {v['max_area']}")
        dim_fit = False
        if not reason:
            for l_, w_ in [(lengths[i], widths[i]), (widths[i], lengths[i])]:
                if l_ <= v["max_length"] and w_ <= v["max_width"] and heights[i] <= v["max_height"]:
                    dim_fit = True
                    break
            if not dim_fit:
                reason.append("FAIL dims (L/W/H) in both orientations")
        if not reason:
            fits_in.append(truck_name)
            dbg(f"  ✔ {truck_name} PASSES all checks")
        else:
            dbg(f"  ✘ {truck_name} excluded: {', '.join(reason)}")
    if fits_in:
        parcel_feasible_vehicles[i] = fits_in
    else:
        invalid_parcels.append(i)

if invalid_parcels:
    st.warning(f"{len(invalid_parcels)} parcel(s) were too large to fit in any truck and were excluded from optimization.")

valid_parcels = list(parcel_feasible_vehicles.keys())

# =========================
# Solver picker (tries HiGHS, then CBC, then GLPK; else None)
# =========================
def pick_solver():
    # Try HiGHS (needs highs binary for HiGHS_CMD; if not present, available() returns False)
    try:
        from pulp import HiGHS_CMD
        s = HiGHS_CMD(msg=False, timeLimit=MIP_TIME_LIMIT_SEC)
        if getattr(s, "available", lambda: False)():
            return s, "HiGHS"
    except Exception:
        pass
    # Try CBC
    try:
        s = pulp.PULP_CBC_CMD(msg=False, timeLimit=MIP_TIME_LIMIT_SEC)
        if getattr(s, "available", lambda: False)():
            return s, "CBC"
    except Exception:
        pass
    # Try GLPK
    try:
        s = pulp.GLPK_CMD(msg=False, options=["--tmlim", str(MIP_TIME_LIMIT_SEC)])
        if getattr(s, "available", lambda: False)():
            return s, "GLPK"
    except Exception:
        pass
    return None, None

# =========================
# Optimizer (tight + symmetry-breaking + time limit) with solver guard
# =========================
def run_optimizer(parcel_indices):
    dbg("\n=== RUN OPTIMIZER ===")
    dbg(f"Optimizing parcels: {parcel_indices}")
    model = pulp.LpProblem("Truck Optimization", pulp.LpMinimize)

    IJ = [(i, j) for i in parcel_indices for j in parcel_feasible_vehicles[i]]
    truck_universe = list(vehicles.keys())

    x = pulp.LpVariable.dicts("Assign", IJ, cat="Binary")
    y = pulp.LpVariable.dicts("UseVehicle", truck_universe, cat="Binary")

    # Objective: minimise fixed truck costs
    model += pulp.lpSum(vehicles[j]["cost"] * y[j] for j in truck_universe)

    # Each parcel exactly once
    for i in parcel_indices:
        feas = [j for j in parcel_feasible_vehicles[i]]
        model += pulp.lpSum(x[i, j] for j in feas) == 1

    # Link x to y
    for (i, j) in IJ:
        model += x[i, j] <= y[j]

    # Capacities (weight + area)
    for j in truck_universe:
        feas_i = [i for i in parcel_indices if (i, j) in x]
        if not feas_i:
            model += y[j] == 0
            continue
        model += pulp.lpSum(weights[i] * x[i, j] for i in feas_i) <= vehicles[j]["max_weight"] * y[j]
        model += pulp.lpSum(areas[i]   * x[i, j] for i in feas_i) <= vehicles[j]["max_area"]   * y[j]

    # Symmetry breaking across clones
    for base, clones in vehicle_groups.items():
        for k in range(len(clones) - 1):
            model += y[clones[k]] >= y[clones[k + 1]]

    # Pick solver
    solver, solver_name = pick_solver()
    if solver is None:
        st.error(
            "No MILP solver is available in this environment. "
            "Install a solver (e.g., CBC on PATH) to enable optimisation. "
            "The app will continue without the MIP step."
        )
        dbg("No solver available → skipping optimisation.")
        return {}, set(), float("inf")

    status = model.solve(solver)
    dbg(f"Solver used: {solver_name}; status: {pulp.LpStatus[model.status]}")

    assignment, used_vehicles = {}, set()
    for (i, j) in IJ:
        if pulp.value(x[i, j]) == 1:
            assignment[i] = j
            used_vehicles.add(j)

    obj_val = pulp.value(model.objective)
    dbg(f"Objective={obj_val}, used_vehicles={sorted(list(used_vehicles))}, assigned={len(assignment)}/{len(parcel_indices)}")
    return assignment, used_vehicles, obj_val

# =========================
# Shelf packer (fallback / optional)
# =========================
def _shelf_pack_once(order, truck):
    layout, failed = [], []
    x_cursor, y_cursor, row_height = 0, 0, 0
    Lmax, Wmax = truck["max_length"], truck["max_width"]

    for i in order:
        placed = False
        orientations = sorted(
            [(lengths[i], widths[i]), (widths[i], lengths[i])],
            key=lambda lw: lw[1]  # try smaller height first
        )
        for l, w in orientations:
            if l <= Lmax - x_cursor and w <= Wmax - y_cursor:
                layout.append((i, x_cursor, y_cursor, l, w))
                x_cursor += l
                row_height = max(row_height, w)
                placed = True
                break
        if placed:
            continue
        x_cursor = 0
        y_cursor += row_height
        row_height = 0
        for l, w in orientations:
            if l <= Lmax and w <= Wmax - y_cursor:
                layout.append((i, x_cursor, y_cursor, l, w))
                x_cursor += l
                row_height = max(row_height, w)
                placed = True
                break
        if not placed:
            failed.append(i)
    return layout, failed

def _shelf_fit_layout(parcel_indices, truck_name):
    truck = vehicles[truck_name]
    order = sorted(
        parcel_indices,
        key=lambda i: (max(lengths[i], widths[i]), min(lengths[i], widths[i])),
        reverse=True,
    )
    layout1, fail1 = _shelf_pack_once(order, truck)
    order2 = sorted(
        parcel_indices,
        key=lambda i: (max(lengths[i], widths[i]), max(lengths[i], widths[i])),
        reverse=True,
    )
    layout2, fail2 = _shelf_pack_once(order2, truck)
    if len(fail2) < len(fail1) or (len(fail2) == len(fail1) and len(layout2) > len(layout1)):
        return layout2, fail2
    return layout1, fail1

# =========================
# Fit layout for a truck (MaxRects via rectpack, with Shelf fallback)
# =========================
def fit_layout_for_truck(parcel_indices, truck_name):
    truck = vehicles[truck_name]
    dbg(f"\n=== FIT LAYOUT in '{truck_name}' ===  (algo: {PACK_MODE}{'' if HAS_RECTPACK else ' [fallback Shelf]'})")
    dbg(f"Truck caps: L={truck['max_length']} W={truck['max_width']} H={truck['max_height']} weight={truck['max_weight']} area={truck['max_area']}")

    if "MaxRects" in PACK_MODE and HAS_RECTPACK:
        SCALE = 1000
        algo_cls = MaxRectsBssf if "BSSF" in PACK_MODE else MaxRectsBaf

        packer = newPacker(rotation=True, pack_algo=algo_cls, sort_algo=SORT_AREA)
        packer.add_bin(int(round(truck["max_length"] * SCALE)), int(round(truck["max_width"] * SCALE)), count=1)

        for i in parcel_indices:
            w = max(1, int(round(lengths[i] * SCALE)))
            h = max(1, int(round(widths[i]  * SCALE)))
            packer.add_rect(w, h, rid=i)

        packer.pack()

        layout, placed = [], set()
        for b, x, y, w, h, i in packer.rect_list():
            layout.append((i, x / SCALE, y / SCALE, w / SCALE, h / SCALE))
            placed.add(i)

        failed = [i for i in parcel_indices if i not in placed]
        layout.sort(key=lambda t: (t[2], t[1]))
        dbg(f"Layout result: placed={len(layout)}, failed={failed}")
        return layout, failed

    layout, failed = _shelf_fit_layout(parcel_indices, truck_name)
    dbg(f"Layout result: placed={len(layout)}, failed={failed}")
    return layout, failed

def fit_layout_for_truck_list(assignment):
    failed = []
    layout_dict = {v: [] for v in set(assignment.values())}
    for v in layout_dict:
        layout, fail = fit_layout_for_truck([i for i, a in assignment.items() if a == v], v)
        layout_dict[v].extend(layout)
        failed.extend(fail)
    return layout_dict, failed

# =========================
# Layout validator (overlaps/bounds)
# =========================
def validate_layout(layout_data, vehicles, tol=1e-9):
    report = {}
    for truck, rects in layout_data.items():
        L = vehicles[truck]["max_length"]
        W = vehicles[truck]["max_width"]

        bounds_ok = True
        overlaps = []

        for i, x, y, l, w in rects:
            if x < -tol or y < -tol or x + l > L + tol or y + w > W + tol:
                bounds_ok = False

        for a in range(len(rects)):
            i1, x1, y1, l1, w1 = rects[a]
            x1b, y1b = x1 + l1, y1 + w1
            for b in range(a + 1, len(rects)):
                i2, x2, y2, l2, w2 = rects[b]
                x2b, y2b = x2 + l2, y2 + w2
                sep = (x1b <= x2 + tol) or (x2b <= x1 + tol) or (y1b <= y2 + tol) or (y2b <= y1 + tol)
                if not sep:
                    overlaps.append((i1, i2))

        report[truck] = {"bounds_ok": bounds_ok, "overlaps": overlaps}
    return report

# =========================
# Visualisation (gutters + adaptive labels)
# =========================
def visualize_layout(layout_data):
    fig, axes = plt.subplots(len(layout_data), 1, figsize=(10, 5 * len(layout_data)))
    if len(layout_data) == 1:
        axes = [axes]

    for ax, (vehicle, parcels) in zip(axes, layout_data.items()):
        truck = vehicles[vehicle]
        ax.set_title(f"{vehicle} (L: {truck['max_length']}m, W: {truck['max_width']}m, H: {truck['max_height']}m)", fontsize=14)
        ax.set_xlim(0, truck["max_length"])
        ax.set_ylim(0, truck["max_width"])
        ax.set_aspect('equal')
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        ax.set_xticks(range(int(truck["max_length"]) + 1))
        ax.set_yticks(range(int(truck["max_width"]) + 1))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_facecolor('white')
        ax.grid(False)

        for p in parcels:
            i, x, y, l, w = p
            # tiny visual gap so borders don't sit flush
            g = 0.02
            gx = min(g, max(l, 0) * 0.2)
            gy = min(g, max(w, 0) * 0.2)
            rect = patches.Rectangle(
                (x + gx/2, y + gy/2),
                max(l - gx, 0),
                max(w - gy, 0),
                linewidth=1.2,
                edgecolor='black',
                facecolor='skyblue',
            )
            ax.add_patch(rect)

            # move/shrink labels for small boxes
            min_side = min(l, w)
            if min_side < 0.7:
                ax.text(x + gx/2 + 0.02, y + gy/2 + 0.02, f"{i + 1}", ha='left', va='bottom', fontsize=7)
            elif min_side < 1.1:
                ax.text(x + l/2, y + w/2, f"{i + 1}", ha='center', va='center', fontsize=9)
            else:
                ax.text(x + l/2, y + w/2, f"{i + 1}", ha='center', va='center', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

# =========================
# Repack-from-scratch helper (prevents overlaps across iterations)
# =========================
def recompute_layout_for_all_assignments(all_assignment):
    layout = {}
    failed = []
    used = sorted(set(all_assignment.values()), key=lambda t: vehicles[t]["cost"])
    for v in used:
        indices = [i for i, a in all_assignment.items() if a == v]
        lay, fail = fit_layout_for_truck(indices, v)
        layout[v] = lay          # replace, don't extend
        failed.extend(fail)
    return layout, failed

# =========================
# Post-pass: remove smallest vehicles by moving into the most-empty open truck
# =========================
def eliminate_small_trucks_postpass(all_assignment):
    truck_to_items = {}
    for i, t in all_assignment.items():
        truck_to_items.setdefault(t, []).append(i)

    def free_metrics(truck):
        v = vehicles[truck]
        used_w = sum(weights[i] for i in truck_to_items.get(truck, []))
        used_a = sum(areas[i]   for i in truck_to_items.get(truck, []))
        return (v["max_area"] - used_a, v["max_weight"] - used_w, v["max_area"])

    donors = [t for t, lst in truck_to_items.items() if lst]
    donors.sort(key=lambda t: (vehicles[t]["max_area"], vehicles[t]["cost"]))

    changed = False

    for donor in donors:
        items = list(truck_to_items.get(donor, []))
        if not items:
            continue

        recipients = [r for r in truck_to_items.keys() if r != donor]
        recipients.sort(key=lambda r: free_metrics(r), reverse=True)

        v_donor = vehicles[donor]
        dbg(f"\n🔁 Post-pass: try removing '{donor}' ({len(items)} items, cost £{v_donor['cost']})")

        moved = False
        for r in recipients:
            v = vehicles[r]
            cand = truck_to_items[r] + items
            tw = sum(weights[i] for i in cand)
            ta = sum(areas[i]   for i in cand)
            if tw > v["max_weight"] + 1e-9 or ta > v["max_area"] + 1e-9:
                dbg(f"  • Skip '{r}' — caps overflow (Δw={tw-v['max_weight']:.2f}, Δa={ta-v['max_area']:.2f}).")
                continue

            lay, fail = fit_layout_for_truck(cand, r)
            if not fail:
                truck_to_items[r] = [i for (i,*_) in lay]
                truck_to_items[donor] = []
                dbg(f"  ✅ Removed '{donor}' by moving {len(items)} item(s) → '{r}'. Saved £{v_donor['cost']}.")
                changed = True
                moved = True
                break
            else:
                dbg(f"  • '{r}' failed geometry when adding all {len(items)} item(s).")

        if not moved:
            dbg(f"  ↪ Could not remove '{donor}' — no single recipient could absorb ALL items.")

    if not changed:
        dbg("ℹ️ Post-pass: no small vehicle could be removed.")
        prev_layout, _ = recompute_layout_for_all_assignments(all_assignment)
        return all_assignment, prev_layout, set(prev_layout.keys())

    new_assignment = {i: t for t, lst in truck_to_items.items() for i in lst}
    new_layout, failed = recompute_layout_for_all_assignments(new_assignment)
    if failed:
        dbg(f"⚠️ Post-pass produced failures {failed}; reverting.")
        prev_layout, _ = recompute_layout_for_all_assignments(all_assignment)
        return all_assignment, prev_layout, set(prev_layout.keys())

    return new_assignment, new_layout, set(new_layout.keys())

# =========================
# Pre-MIP geometry helpers
# =========================
def try_single_truck_for_all(parcel_indices):
    sorted_trucks = sorted(vehicles.keys(), key=lambda t: vehicles[t]["cost"])
    all_idx = list(parcel_indices)
    for truck_name in sorted_trucks:
        if not all(truck_name in parcel_feasible_vehicles.get(i, []) for i in all_idx):
            continue
        caps = vehicles[truck_name]
        tot_w = sum(weights[i] for i in all_idx)
        tot_a = sum(areas[i]   for i in all_idx)
        if tot_w > caps["max_weight"] or tot_a > caps["max_area"]:
            continue
        layout, failed = fit_layout_for_truck(all_idx, truck_name)
        if not failed:
            assignment = {i: truck_name for i in all_idx}
            layout_dict = {truck_name: layout}
            used = {truck_name}
            dbg(f"✅ Global single-truck pick: '{truck_name}' fits ALL parcels.")
            return assignment, layout_dict, used
    return None

def seed_with_largest_truck(parcel_indices):
    seed_truck = max(vehicles.keys(), key=lambda t: (vehicles[t]["max_area"], vehicles[t]["cost"]))
    feasible = [i for i in parcel_indices if seed_truck in parcel_feasible_vehicles.get(i, [])]
    if not feasible:
        dbg(f"⚠️ Largest truck '{seed_truck}' is not feasible for any parcel; skipping seed.")
        return {}, {}, set(), list(parcel_indices)

    caps = vehicles[seed_truck]
    dbg(f"🔎 Seeding with largest truck '{seed_truck}' (area={caps['max_area']}, cost={caps['cost']})")

    layout, failed = fit_layout_for_truck(feasible, seed_truck)
    placed = [i for (i, *_rest) in layout]

    def total_weight_of(indices): return sum(weights[i] for i in indices)
    placed_set = set(placed)
    while total_weight_of(list(placed_set)) > caps["max_weight"] and placed_set:
        rem = max(placed_set, key=lambda i: weights[i])
        placed_set.remove(rem)
        layout, _fail = fit_layout_for_truck(list(placed_set), seed_truck)
        placed_set = {i for (i, *_r) in layout}

    placed = list(placed_set)
    assignment = {i: seed_truck for i in placed}
    layout_dict = {seed_truck: layout} if placed else {}
    used = {seed_truck} if placed else set()
    leftovers = [i for i in parcel_indices if i not in placed]

    if placed:
        dbg(f"✅ Seeded '{seed_truck}' with {len(placed)} parcel(s); {len(leftovers)} leftover.")
    else:
        dbg(f"ℹ️ Seeding placed none on '{seed_truck}'.")
    return assignment, layout_dict, used, leftovers

# =========================
# Stage B: Main loop
# =========================
if st.button("Run Optimization"):
    debug_log.clear()
    if not valid_parcels:
        st.error("No valid parcels to optimize. All parcels exceed truck dimensions.")
    else:
        unassigned = valid_parcels.copy()
        all_assignment = {}
        used_trucks = set()
        all_layout = {}

        # 1) Global single-truck attempt
        single = try_single_truck_for_all(unassigned)
        if single is not None:
            all_assignment, all_layout, used_trucks = single
            unassigned = []  # everything placed

        # 2) Seed with the largest/most expensive truck
        if unassigned:
            seed_assign, seed_layout, seed_used, leftovers = seed_with_largest_truck(unassigned)
            all_assignment.update(seed_assign)
            if seed_layout:
                all_layout.update(seed_layout)
            used_trucks.update(seed_used)
            unassigned = leftovers

        progress_text = st.empty()
        progress_bar = st.progress(0)

        attempt = 1
        max_attempts = 100
        no_progress_count = 0
        last_unassigned_count = len(unassigned)

        while unassigned and attempt <= max_attempts:
            progress_text.text(f"Optimization attempt: {attempt}")
            progress_bar.progress(min(1.0, attempt / max_attempts))

            sorted_trucks = sorted(vehicles.keys(), key=lambda t: vehicles[t]["cost"])
            placed_this_round = False

            # 3) Single-truck shortcut (subset) tries ALL trucks (pre-MIP geometry)
            any_tried = False
            for truck_name in sorted_trucks:
                if not all(truck_name in parcel_feasible_vehicles.get(i, []) for i in unassigned):
                    continue
                caps = vehicles[truck_name]
                total_weight_batch = sum(weights[i] for i in unassigned)
                total_area_batch   = sum(areas[i]   for i in unassigned)
                if total_weight_batch > caps["max_weight"] or total_area_batch > caps["max_area"]:
                    continue

                any_tried = True
                dbg(f"\n--- Trying single-truck shortcut with '{truck_name}' for parcels {unassigned} ---")

                for i in unassigned:
                    all_assignment[i] = truck_name

                all_layout, failed = recompute_layout_for_all_assignments(all_assignment)

                for i in failed:
                    if i in all_assignment:
                        del all_assignment[i]

                used_trucks = set(all_layout.keys())
                unassigned = failed

                if len(unassigned) == 0:
                    dbg(f"✅ Accept '{truck_name}' — geometry fits and caps satisfied for all.")
                    placed_this_round = True
                    break
                else:
                    dbg(f"ℹ️ '{truck_name}' took a subset; {len(unassigned)} unplaced remain → will try next truck.")

            if not placed_this_round:
                if not any_tried:
                    dbg("\nNo single truck could take the whole batch. Falling back to optimiser.")
                else:
                    dbg("\nTried all trucks with shortcut; still have leftovers. Falling back to optimiser.")

                prev_unassigned = set(unassigned)

                # 4) MIP on leftovers (may be skipped if no solver available)
                assignment, used, _ = run_optimizer(unassigned)

                # merge new assignments
                for i, v in assignment.items():
                    all_assignment[i] = v

                counts = Counter(all_assignment.values())
                dbg("Repacking trucks (current counts): " + ", ".join(f"{t}:{counts[t]}" for t in sorted(counts)))

                # geometry validate
                all_layout, failed = recompute_layout_for_all_assignments(all_assignment)

                # drop failed items back to leftovers
                failed_set = set(failed)
                for i in failed_set:
                    if i in all_assignment:
                        del all_assignment[i]

                used_trucks = set(all_layout.keys())

                assigned_now = set(assignment.keys())
                packed_ok = assigned_now - failed_set
                new_unassigned = list(prev_unassigned - packed_ok)

                if len(new_unassigned) >= last_unassigned_count:
                    no_progress_count += 1
                else:
                    no_progress_count = 0
                    last_unassigned_count = len(new_unassigned)

                unassigned = new_unassigned

                if no_progress_count >= NO_PROGRESS_MAX:
                    st.error(f"Stopped after {NO_PROGRESS_MAX} no-progress rounds. Could not place {len(unassigned)} parcel(s): {sorted(unassigned)}")
                    break

            attempt += 1

        progress_bar.progress(1.0)
        progress_text.text("Optimization complete.")

        # ==========================
        # Post-pass — remove smallest vehicles if a single recipient can absorb them
        # ==========================
        if all_assignment:
            all_assignment, all_layout, used_trucks = eliminate_small_trucks_postpass(all_assignment)

        # ==========================
        # Results
        # ==========================
        placed_count = len(all_assignment)
        total_valid = len(valid_parcels)

        if placed_count == total_valid:
            st.success(f"All parcels placed successfully. ({placed_count}/{total_valid})")
        else:
            st.warning(f"{total_valid - placed_count} parcel(s) could not be placed. ({placed_count}/{total_valid})")

        if all_assignment:
            truck_summary = pd.Series(list(all_assignment.values())).value_counts().reset_index()
            truck_summary.columns = ["Truck", "Number of Parcels"]
            st.dataframe(truck_summary)

            grouped = {}
            for i, t in all_assignment.items():
                grouped.setdefault(t, []).append(i + 1)
            st.markdown("#### Parcels per truck")
            for t in sorted(grouped.keys(), key=lambda k: vehicles[k]["cost"]):
                ids = sorted(grouped[t])
                st.markdown(f"**{t}**  — {len(ids)} parcels")
                st.code(", ".join(map(str, ids)))

            report = validate_layout(all_layout, vehicles)
            bad = {tt: rr for tt, rr in report.items() if (not rr["bounds_ok"] or rr["overlaps"])}
            if bad:
                st.error("Geometry check found issues (true overlaps or out-of-bounds):")
                for tt, rr in bad.items():
                    st.write(f"**{tt}** → bounds_ok={rr['bounds_ok']}, overlaps={[(a+1,b+1) for a,b in rr['overlaps']]}")
            else:
                st.success("Geometry check: ✅ no overlaps and all parcels inside deck bounds.")

            visualize_layout(all_layout)

# =========================
# Sidebar diagnostics
# =========================
with st.sidebar.expander("Environment diagnostics", expanded=False):
    st.write("PuLP:", pulp.__version__)
    try:
        from pulp import HiGHS_CMD
        st.write("HiGHS available():", HiGHS_CMD().available())
    except Exception as e:
        st.write("HiGHS check:", str(e))
    try:
        st.write("CBC available():", pulp.PULP_CBC_CMD().available())
    except Exception as e:
        st.write("CBC check:", str(e))
    try:
        st.write("GLPK available():", pulp.GLPK_CMD().available())
    except Exception as e:
        st.write("GLPK check:", str(e))
    st.write("rectpack present:", HAS_RECTPACK)

# =========================
# Debug panel
# =========================
if show_debug:
    st.markdown("### 🐞 Debug Log")
    debug_text = "\n".join(debug_log)
    if len(debug_text) > 12000:
        debug_text = "...(truncated)…\n" + debug_text[-12000:]
    st.text_area("Internal debug output (also printed to server console)", value=debug_text, height=320)
