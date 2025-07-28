import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BIG_M = 10000
EPSILON = 0.001

st.set_page_config(page_title="Truck Optimiser", layout="wide")
st.title("🚚 Truck Optimiser")

# Vehicle data
vehicle_data = [
    ("Small van", 1.5, 1.2, 1.1, 360, 1.44, 80),
    ("Medium wheel base", 3, 1.2, 1.9, 1400, 3.6, 50),
    ("Sprinter van", 4.2, 1.2, 1.75, 950, 5.04, 85),
    ("luton van", 4, 2, 2, 1000, 8, 110),
    ("7.5T CS", 6, 2.4, 2.2, 2600, 14.4, 100),
    ("18T CS", 8, 2.4, 2.3, 9800, 19.2, 130),
    ("40ft CS", 13.5, 2.5, 3, 28000, 33.75, 140),
    ("20ft FB", 8, 2.4, 300, 10500, 19.2, 110),
    ("40ft FB", 13.5, 2.4, 300, 30000, 32.4, 135),
    ("40T Low Loader", 13.5, 2.4, 300, 30000, 32.4, 135),
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
            "cost": cost
        }

# Inputs
st.header("Parcel Inputs")
num_individual = st.number_input("Number of Individual Parcels", min_value=0, max_value=200, value=0)
weights, lengths, widths, heights = [], [], [], []
cols = st.columns(5)
for i in range(num_individual):
    with cols[0]:
        weights.append(st.number_input(f"Weight {i+1}", key=f"wt_{i}", value=100.0))
    with cols[1]:
        lengths.append(st.number_input(f"Length {i+1}", key=f"len_{i}", value=1.0))
    with cols[2]:
        widths.append(st.number_input(f"Width {i+1}", key=f"wid_{i}", value=1.0))
    with cols[3]:
        heights.append(st.number_input(f"Height {i+1}", key=f"hei_{i}", value=1.0))
    with cols[4]:
        st.markdown("&nbsp;")

# Derived
areas = [lengths[i] * widths[i] for i in range(len(weights))]

# Filter parcels too large for any vehicle
valid_parcels, invalid_parcels = [], []
for i in range(len(weights)):
    fits_any = any(
        (lengths[i] <= v["max_length"] and widths[i] <= v["max_width"]) or
        (widths[i] <= v["max_length"] and lengths[i] <= v["max_width"])
        for v in vehicles.values()
    )
    if fits_any:
        valid_parcels.append(i)
    else:
        invalid_parcels.append(i)

if invalid_parcels:
    st.warning(f"{len(invalid_parcels)} parcel(s) excluded due to size.")

# Optimizer
def run_optimizer(parcel_indices):
    model = pulp.LpProblem("Truck Optimization", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("Assign", ((i, j) for i in parcel_indices for j in vehicles), cat="Binary")
    y = pulp.LpVariable.dicts("UseVehicle", (j for j in vehicles), cat="Binary")

    model += pulp.lpSum(vehicles[j]["cost"] * y[j] for j in vehicles)

    for i in parcel_indices:
        model += pulp.lpSum(x[i, j] for j in vehicles) == 1

    for j in vehicles:
        model += pulp.lpSum(weights[i] * x[i, j] for i in parcel_indices) <= vehicles[j]["max_weight"] * y[j]
        model += pulp.lpSum(areas[i] * x[i, j] for i in parcel_indices) <= vehicles[j]["max_area"] + BIG_M * (1 - y[j])

    # 🚫 Prevent assigning parcels to trucks that can't fit their dimensions
    for i in parcel_indices:
        for j in vehicles:
            l_ok = lengths[i] <= vehicles[j]["max_length"]
            w_ok = widths[i] <= vehicles[j]["max_width"]
            rotated_l_ok = widths[i] <= vehicles[j]["max_length"]
            rotated_w_ok = lengths[i] <= vehicles[j]["max_width"]

            if not ((l_ok and w_ok) or (rotated_l_ok and rotated_w_ok)):
                model += x[i, j] == 0

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    assignment = {}
    used_vehicles = set()
    for i in parcel_indices:
        for j in vehicles:
            if pulp.value(x[i, j]) == 1:
                assignment[i] = j
                used_vehicles.add(j)
                print(f"Parcel {i} assigned to {j} — Truck L:{vehicles[j]['max_length']} W:{vehicles[j]['max_width']} — Parcel L:{lengths[i]} W:{widths[i]}")
                break
    return assignment, used_vehicles, pulp.value(model.objective)

# Layout fitter
def fit_layout(assignment):
    layout = {v: [] for v in set(assignment.values())}
    failed = []

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
                    print(f"❌ Parcel {i} does NOT fit in {v} during layout phase")
                    failed.append(i)

    return layout, failed

# Visualizer
def visualize_layout(layout_data):
    fig, axes = plt.subplots(len(layout_data), 1, figsize=(10, 5 * len(layout_data)))
    if len(layout_data) == 1:
        axes = [axes]

    for ax, (vehicle, parcels) in zip(axes, layout_data.items()):
        truck = vehicles[vehicle]
        ax.set_title(f"{vehicle} (L: {truck['max_length']}m, W: {truck['max_width']}m)")
        ax.set_xlim(0, truck["max_length"])
        ax.set_ylim(0, truck["max_width"])
        ax.set_aspect('equal')
        ax.set_facecolor('white')

        for p in parcels:
            i, x, y, l, w = p
            rect = patches.Rectangle((x, y), l, w, linewidth=1.5, edgecolor='black', facecolor='skyblue')
            ax.add_patch(rect)
            ax.text(x + l / 2, y + w / 2, f"P{i}", ha='center', va='center')

    st.pyplot(fig)

# Run
if st.button("Run Optimization"):
    if not valid_parcels:
        st.error("No valid parcels to optimize.")
    else:
        unassigned = valid_parcels.copy()
        all_assignment = {}
        used_trucks = set()
        all_layout = {}
        max_attempts = 100

        for attempt in range(1, max_attempts + 1):
            st.write(f"Attempt {attempt}")
            assignment, used, _ = run_optimizer(unassigned)
            layout, failed = fit_layout(assignment)

            for i in assignment:
                if i not in failed:
                    all_assignment[i] = assignment[i]

            for v, layout_list in layout.items():
                all_layout.setdefault(v, []).extend(layout_list)

            unassigned = failed
            used_trucks.update(used)

            if not failed:
                break

        if unassigned:
            st.error("❌ Some parcels could not be placed after retries.")
        else:
            st.success("✅ All parcels placed successfully.")

        if all_assignment:
            st.subheader("Truck Summary")
            summary = pd.Series(list(all_assignment.values())).value_counts().reset_index()
            summary.columns = ["Truck", "Parcel Count"]
            st.dataframe(summary)

            st.subheader("Truck Layouts")
            visualize_layout(all_layout)
