import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BIG_M = 10000
EPSILON = 0.001

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

# Vehicle data
vehicle_data = [
    ("Small van", 1.5, 1.2, 1.1, 360, 1.8, 50),
    ("Medium wheel base", 3, 1.2, 1.9, 1400, 3.6, 80),
    ("Sprinter van", 4.2, 1.2, 1.75, 950, 5.04, 85),
    ("luton van", 4, 2, 2, 1000, 8, 110),
    ("7.5T CS", 6, 2.88, 2.2, 2600, 17.28, 100),
    ("18T CS", 7.3, 2.88, 2.3, 9800, 21.024, 125),
    ("40ft CS", 13.5, 3, 3, 28000, 40.5, 135),
    ("20ft FB", 7.3, 2.4, 300, 10500, 17.52, 130),
    ("40ft FB", 13.5, 2.4, 300, 30000, 32.4, 140),
    ("40T Low Loader", 13.5, 2.4, 300, 30000, 32.4, 145),
]

vehicles = {}
for i in range(1, 11):
    for name, l, w, h, wt, ar, cost in vehicle_data:
        vehicles[f"{name}{i if i > 1 else ''}"] = {
            "max_length": l,
            "max_width": w,
            "max_height": h,
            "max_weight": wt,
            "max_area": ar,
            "cost": cost + 50  # Add driver cost
        }

# Input form
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

# ✅ New: Feasible vehicle mapping (with rotation-aware logic)
parcel_feasible_vehicles = {}
invalid_parcels = []

for i in range(len(weights)):
    fits_in = []
    for truck_name, v in vehicles.items():
        for l, w in [(lengths[i], widths[i]), (widths[i], lengths[i])]:
            if (
                l <= v["max_length"] and
                w <= v["max_width"] and
                heights[i] <= v["max_height"]
            ):
                fits_in.append(truck_name)
                break
    if fits_in:
        parcel_feasible_vehicles[i] = fits_in
    else:
        invalid_parcels.append(i)

if invalid_parcels:
    st.warning(f"{len(invalid_parcels)} parcel(s) were too large to fit in any truck and were excluded from optimization.")

valid_parcels = list(parcel_feasible_vehicles.keys())

# Optimizer
def run_optimizer(parcel_indices):
    model = pulp.LpProblem("Truck Optimization", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("Assign", ((i, j) for i in parcel_indices for j in parcel_feasible_vehicles[i]), cat="Binary")
    y = pulp.LpVariable.dicts("UseVehicle", (j for j in vehicles), cat="Binary")

    model += pulp.lpSum(vehicles[j]["cost"] * y[j] for j in vehicles)

    for i in parcel_indices:
        model += pulp.lpSum(x[i, j] for j in parcel_feasible_vehicles[i]) == 1

    for j in vehicles:
        model += pulp.lpSum(weights[i] * x[i, j] for i in parcel_indices if j in parcel_feasible_vehicles[i]) <= vehicles[j]["max_weight"] * y[j]
        model += pulp.lpSum(areas[i] * x[i, j] for i in parcel_indices if j in parcel_feasible_vehicles[i]) <= vehicles[j]["max_area"] + BIG_M * (1 - y[j])

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    assignment, used_vehicles = {}, set()
    for i in parcel_indices:
        for j in parcel_feasible_vehicles[i]:
            if pulp.value(x[i, j]) == 1:
                assignment[i] = j
                used_vehicles.add(j)
                break

    total_cost = pulp.value(model.objective)
    return assignment, used_vehicles, total_cost

# Layout fitting
def fit_layout(assignment):
    failed = []
    layout = {v: [] for v in set(assignment.values())}

    for v in layout:
        truck = vehicles[v]
        parcels = [i for i, a in assignment.items() if a == v]
        x_cursor, y_cursor, row_height = 0, 0, 0

        for i in parcels:
            L, W = lengths[i], widths[i]
            placed = False
            for rotate in [(L, W), (W, L)]:
                l, w = rotate
                if l > truck["max_length"] or w > truck["max_width"]:
                    continue
                if x_cursor + l <= truck["max_length"] and y_cursor + w <= truck["max_width"]:
                    layout[v].append((i, x_cursor, y_cursor, l, w))
                    x_cursor += l
                    row_height = max(row_height, w)
                    placed = True
                    break
            if not placed:
                x_cursor = 0
                y_cursor += row_height
                row_height = 0
                if L <= truck["max_length"] and W <= truck["max_width"] and y_cursor + W <= truck["max_width"]:
                    layout[v].append((i, x_cursor, y_cursor, L, W))
                    x_cursor += L
                    row_height = W
                else:
                    failed.append(i)
    return layout, failed

# Visualization
def visualize_layout(layout_data):
    fig, axes = plt.subplots(len(layout_data), 1, figsize=(10, 5 * len(layout_data)))
    if len(layout_data) == 1:
        axes = [axes]

    for ax, (vehicle, parcels) in zip(axes, layout_data.items()):
        truck = vehicles[vehicle]
        ax.set_title(f"{vehicle} (L: {truck['max_length']}m, W: {truck['max_width']}m, H: {truck['max_height']}m)",  fontsize=14)
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
            rect = patches.Rectangle((x, y), l, w, linewidth=1.5, edgecolor='black', facecolor='skyblue')
            ax.add_patch(rect)
            ax.text(x + l / 2, y + w / 2, f"{i + 1}", ha='center', va='center', fontsize=12, color='black')

    plt.tight_layout()
    st.pyplot(fig)

# Run optimization
if st.button("Run Optimization"):
    if not valid_parcels:
        st.error("No valid parcels to optimize. All parcels exceed truck dimensions.")
    else:
        unassigned = valid_parcels.copy()
        all_assignment = {}
        used_trucks = set()
        total_cost = 0
        all_layout = {}
        attempt = 1
        max_attempts = 100

        progress_text = st.empty()
        progress_bar = st.progress(0)

        while unassigned and attempt <= max_attempts:
            progress_text.text(f"Optimization attempt: {attempt}")
            progress_bar.progress(attempt / max_attempts)

            assignment, used, cost = run_optimizer(unassigned)
            layout, failed = fit_layout(assignment)

            for i in assignment:
                if i not in failed:
                    all_assignment[i] = assignment[i]

            for v, layout_list in layout.items():
                if v not in all_layout:
                    all_layout[v] = []
                all_layout[v].extend(layout_list)

            unassigned = failed
            used_trucks.update(used)
            total_cost += cost

            if not failed:
                break
            attempt += 1

        progress_bar.progress(1.0)
        progress_text.text("Optimization complete.")

        if unassigned:
            st.error("Some parcels could not be placed after retries.")
        else:
            st.success("All parcels placed successfully.")

        if all_assignment:
            truck_summary = pd.Series(list(all_assignment.values())).value_counts().reset_index()
            truck_summary.columns = ["Truck", "Number of Parcels"]
            st.dataframe(truck_summary)

            visualize_layout(all_layout)
