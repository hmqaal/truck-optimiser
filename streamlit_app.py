import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BIG_M = 10000
EPSILON = 0.001

# ──────── Streamlit Page Setup ─────────
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

# ──────── 1. Vehicle Data & Replicates ─────────
vehicle_data = [
    ("Small van",        1.5, 1.2, 1.1,   360,   1.8,    50),
    ("Medium wheel base",3,   1.2, 1.9,  1400,   3.6,    80),
    ("Sprinter van",     4.2, 1.2, 1.75,  950,   5.04,   85),
    ("luton van",        4,   2,   2,    1000,   8,     110),
    ("7.5T CS",          6,   2.88,2.2,  2600,  17.28,  100),
    ("18T CS",           7.3, 2.88,2.3,  9800,  21.024, 125),
    ("40ft CS",          13.5,3,   3,   28000,  40.5,   135),
    ("20ft FB",          7.3, 2.4,300, 10500,  17.52,  130),
    ("40ft FB",          13.5,2.4,300, 30000,  32.4,   140),
    ("40T Low Loader",   13.5,2.4,300, 30000,  32.4,   145),
]

# replicate each type 10 times
vehicles = {}
for i in range(1, 11):
    for name, l, w, h, wt, ar, cost in vehicle_data:
        suffix = str(i) if i > 1 else ""
        vehicles[f"{name}{suffix}"] = {
            "max_length": l,
            "max_width": w,
            "max_height": h,
            "max_weight": wt,
            "max_area": ar,
            "cost": cost + 50  # add driver cost
        }

# ──────── 2. User Inputs ─────────
st.header("Inventory Inputs")
num_individual = st.number_input("Number of Individual Inventory", min_value=0, max_value=200, value=0)
weights, lengths, widths, heights = [], [], [], []
cols = st.columns(5)
for i in range(num_individual):
    with cols[0]:
        weights.append(st.number_input(f"Weight {i+1} (kg)", key=f"wt_{i}", value=100.0))
    with cols[1]:
        lengths.append(st.number_input(f"Length {i+1} (m)", key=f"len_{i}", value=1.0))
    with cols[2]:
        widths.append(st.number_input(f"Width {i+1} (m)", key=f"wid_{i}", value=1.0))
    with cols[3]:
        heights.append(st.number_input(f"Height {i+1} (m)", key=f"hei_{i}", value=1.0))
    with cols[4]:
        st.markdown("&nbsp;")

st.markdown("---")
st.subheader("Bulk Inventory Entries")
bulk_entries = st.number_input("Number of Bulk Inventory Types", min_value=0, max_value=20, value=0)

for i in range(bulk_entries):
    st.markdown(f"**Bulk Parcel Type {i+1}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        quantity = st.number_input("Quantity", min_value=1, value=1, key=f"qty_{i}")
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

# ──────── 3. Feasible Replicates per Parcel ─────────
parcel_feasible_vehicles = {}
invalid_parcels = []
for i in range(len(weights)):
    fits = []
    for v_name, spec in vehicles.items():
        for (l, w) in ((lengths[i], widths[i]), (widths[i], lengths[i])):
            if (l <= spec["max_length"] and
                w <= spec["max_width"]  and
                heights[i] <= spec["max_height"]):
                fits.append(v_name)
                break
    if fits:
        parcel_feasible_vehicles[i] = fits
    else:
        invalid_parcels.append(i)

if invalid_parcels:
    st.warning(f"{len(invalid_parcels)} parcel(s) too large for any truck were excluded.")

valid_parcels = list(parcel_feasible_vehicles.keys())

# ──────── 4. Group Replicates into Types ─────────
truck_types = {}
for v_name, spec in vehicles.items():
    # strip digits from end to get base type
    base = ''.join(c for c in v_name if not c.isdigit()).strip()
    if base not in truck_types:
        truck_types[base] = spec

# map each parcel to feasible types (not replicate names)
parcel_feasible_types = {
    i: list({ ''.join(c for c in v if not c.isdigit()).strip()
              for v in parcel_feasible_vehicles[i] })
    for i in valid_parcels
}

# ──────── 5. Layout Fitting Function (unchanged) ─────────
def fit_layout(assign_map):
    failed = []
    layout = {v: [] for v in set(assign_map.values())}
    for v in layout:
        spec = vehicles[v]
        parcels = [i for i, a in assign_map.items() if a == v]
        x_cur, y_cur, row_h = 0, 0, 0
        for i in parcels:
            L, W = lengths[i], widths[i]
            placed = False
            for dL, dW in ((L, W), (W, L)):
                if (dL <= spec["max_length"] and dW <= spec["max_width"] and
                    x_cur + dL <= spec["max_length"] and
                    y_cur + dW <= spec["max_width"]):
                    layout[v].append((i, x_cur, y_cur, dL, dW))
                    x_cur += dL
                    row_h = max(row_h, dW)
                    placed = True
                    break
            if not placed:
                x_cur = 0
                y_cur += row_h
                row_h = 0
                if (L <= spec["max_length"] and W <= spec["max_width"] and
                    y_cur + W <= spec["max_width"]):
                    layout[v].append((i, x_cur, y_cur, L, W))
                    x_cur += L
                    row_h = W
                else:
                    failed.append(i)
    return layout, failed

# ──────── 6. Visualization (updated to show H) ─────────
def visualize_layout(layout_data):
    fig, axes = plt.subplots(len(layout_data), 1,
                             figsize=(10, 5 * len(layout_data)))
    if len(layout_data) == 1:
        axes = [axes]
    for ax, (vehicle, parcels) in zip(axes, layout_data.items()):
        spec = vehicles[vehicle]
        ax.set_title(
            f"{vehicle} "
            f"(L: {spec['max_length']}m, "
            f"W: {spec['max_width']}m, "
            f"H: {spec['max_height']}m)",
            fontsize=14
        )
        ax.set_xlim(0, spec["max_length"])
        ax.set_ylim(0, spec["max_width"])
        ax.set_aspect('equal')
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        ax.set_xticks(range(int(spec["max_length"]) + 1))
        ax.set_yticks(range(int(spec["max_width"]) + 1))
        ax.set_facecolor('white')
        ax.grid(False)

        for i, x, y, l, w in parcels:
            rect = patches.Rectangle((x, y), l, w,
                                     linewidth=1.5,
                                     edgecolor='black',
                                     facecolor='skyblue')
            ax.add_patch(rect)
            ax.text(x + l/2, y + w/2,
                    f"{i+1}",
                    ha='center', va='center',
                    fontsize=12, color='black')
    plt.tight_layout()
    st.pyplot(fig)

# ──────── 7. Run Grouped MILP + Assignment to Replicates ─────────
if st.button("Run Optimization"):
    if not valid_parcels:
        st.error("No valid parcels to optimise.")
    else:
        # Build MILP
        model = pulp.LpProblem("Grouped_Truck_Optimization", pulp.LpMinimize)

        # Integer: how many trucks of each type
        n_trucks = pulp.LpVariable.dicts(
            "NumTruck", truck_types.keys(),
            lowBound=0, upBound=len(valid_parcels),
            cat="Integer"
        )

        # Binary: parcel i → type T
        x = pulp.LpVariable.dicts(
            "Assign", 
            ((i, T) for i in valid_parcels for T in parcel_feasible_types[i]),
            cat="Binary"
        )

        # Objective
        model += pulp.lpSum(truck_types[T]["cost"] * n_trucks[T]
                            for T in truck_types)

        # Each parcel assigned exactly once
        for i in valid_parcels:
            model += pulp.lpSum(x[i, T] for T in parcel_feasible_types[i]) == 1

        # Capacity constraints per type
        for T, spec in truck_types.items():
            model += (
                pulp.lpSum(weights[i] * x[i, T]
                           for i in valid_parcels if (i, T) in x)
                <= spec["max_weight"] * n_trucks[T]
            )
            model += (
                pulp.lpSum(areas[i] * x[i, T]
                           for i in valid_parcels if (i, T) in x)
                <= spec["max_area"] * n_trucks[T]
            )

        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=120))

        # Extract type‐level assignments & counts
        total_cost = pulp.value(model.objective)
        type_counts = {T: int(pulp.value(n_trucks[T])) for T in truck_types}
        type_assign = {
            i: next(T for T in parcel_feasible_types[i]
                    if pulp.value(x[i, T]) > 0.5)
            for i in valid_parcels
        }

        # Post‐processing: assign to specific replicates
        replica_of = {}
        by_type = {}
        for i, T in type_assign.items():
            by_type.setdefault(T, []).append(i)

        for T, plist in by_type.items():
            spec = truck_types[T]
            Wcap, Acap = spec["max_weight"], spec["max_area"]
            R = type_counts[T]
            # first‐fit decreasing by area
            plist.sort(key=lambda i: areas[i], reverse=True)
            loads = [{"rem_w": Wcap, "rem_a": Acap} for _ in range(R)]
            for i in plist:
                for r, load in enumerate(loads, start=1):
                    if (weights[i] <= load["rem_w"] and
                        areas[i] <= load["rem_a"]):
                        # replicate name matches vehicles keys
                        suffix = str(r) if r > 1 else ""
                        replicate_name = f"{T}{suffix}"
                        replica_of[i] = replicate_name
                        load["rem_w"] -= weights[i]
                        load["rem_a"] -= areas[i]
                        break

        # Build layout & visualize
        layout, failed = fit_layout(replica_of)
        if failed:
            st.error(f"{len(failed)} parcel(s) failed layout.")
        else:
            st.success(f"All parcels placed! Total Cost: £{total_cost:.2f}")

        # Summary table
        if replica_of:
            df = (pd.Series(list(replica_of.values()))
                  .value_counts()
                  .rename_axis("Truck")
                  .reset_index(name="Parcels"))
            st.dataframe(df)

        # Visualise
        if layout:
            visualize_layout(layout)
