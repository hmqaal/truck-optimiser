
import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Page config
st.set_page_config(page_title="Truck Optimizer", layout="wide")

# Vehicle specs (45 total)
base_vehicles = {
    "Small van": [1.5, 1.2, 1.0, 350, 80],
    "Short wheel base": [2.0, 1.2, 1.3, 800, 95],
    "Medium wheel base": [3.0, 1.2, 1.9, 1400, 50],
    "4 meter sprinter": [4.2, 1.25, 1.9, 1250, 85],
    "luton van": [4.0, 2.0, 2.0, 1000, 110],
    "7.5 tonne": [6.0, 2.3, 2.3, 2800, 100],
    "18 tonne": [7.3, 2.4, 2.4, 9000, 130],
    "26 tonne": [8.0, 2.4, 2.5, 15000, 140],
    "arctic": [13.5, 2.5, 2.7, 24000, 200],
}

vehicles = {}
for i in range(1, 6):
    for name, (length, width, height, max_weight, cost) in base_vehicles.items():
        key = f"{name}{i}" if i > 1 else name
        vehicles[key] = {
            "length": length,
            "width": width,
            "height": height,
            "max_weight": max_weight,
            "cost": cost
        }

# UI
st.title("🚛 Truck Optimizer")
num_parcels = st.number_input("Number of Parcels", 1, 20, 2)

weights, lengths, widths, heights = [], [], [], []

cols = st.columns(5)
for i in range(num_parcels):
    with cols[0]:
        weights.append(st.number_input(f"Parcel {i+1} Weight (kg)", 0.1, 10000.0, 100.0, key=f"weight_{i}"))
    with cols[1]:
        lengths.append(st.number_input(f"Parcel {i+1} Length (m)", 0.1, 20.0, 1.0, key=f"length_{i}"))
    with cols[2]:
        widths.append(st.number_input(f"Parcel {i+1} Width (m)", 0.1, 10.0, 1.0, key=f"width_{i}"))
    with cols[3]:
        heights.append(st.number_input(f"Parcel {i+1} Height (m)", 0.1, 10.0, 1.0, key=f"height_{i}"))

areas = [lengths[i] * widths[i] for i in range(num_parcels)]

if st.button("Run Optimization"):
    model = pulp.LpProblem("TruckOptimization", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_parcels) for j in vehicles), cat="Binary")
    y = pulp.LpVariable.dicts("y", (j for j in vehicles), cat="Binary")

    # Objective
    model += pulp.lpSum(vehicles[j]["cost"] * y[j] for j in vehicles)

    # Constraints
    for i in range(num_parcels):
        model += pulp.lpSum(x[i, j] for j in vehicles) == 1

    for j in vehicles:
        model += pulp.lpSum(weights[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["max_weight"] * y[j]
        model += pulp.lpSum(areas[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["length"] * vehicles[j]["width"] * y[j]

        # Enforce sum of parcel dimensions fit inside vehicle dimensions
        model += pulp.lpSum(lengths[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["length"] * y[j]
        model += pulp.lpSum(widths[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["width"] * y[j]
        model += pulp.lpSum(heights[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["height"] * y[j]

    model.solve()

    st.subheader("Optimization Results")
    st.write(f"**Status:** {pulp.LpStatus[model.status]}")
    st.write(f"**Total Cost:** £{pulp.value(model.objective):.2f}")

    results = []
    vehicle_parcels = {}

    for j in vehicles:
        if pulp.value(y[j]) == 1:
            vehicle_parcels[j] = []
            for i in range(num_parcels):
                if pulp.value(x[i, j]) == 1:
                    results.append({
                        "Vehicle": j,
                        "Parcel #": i + 1,
                        "Weight (kg)": weights[i],
                        "Length (m)": lengths[i],
                        "Width (m)": widths[i],
                        "Height (m)": heights[i],
                    })
                    vehicle_parcels[j].append(i)

    if results:
        df = pd.DataFrame(results)
        st.dataframe(df[["Parcel #", "Vehicle", "Weight (kg)", "Length (m)", "Width (m)", "Height (m)"]])

        # Draw diagrams
        st.subheader("📐 Parcel Layouts Inside Vehicles")

        for vehicle_name, parcel_indices in vehicle_parcels.items():
            fig, ax = plt.subplots(figsize=(6, 3))
            v = vehicles[vehicle_name]
            ax.set_title(vehicle_name)
            ax.set_xlim(0, v["length"])
            ax.set_ylim(0, v["width"])
            ax.set_xlabel("Length (m)")
            ax.set_ylabel("Width (m)")

            # Simple placement: place parcels side by side along length
            x_cursor = 0
            for idx in parcel_indices:
                l, w = lengths[idx], widths[idx]
                if x_cursor + l <= v["length"]:
                    rect = patches.Rectangle((x_cursor, 0), l, w, edgecolor='black', facecolor='skyblue')
                    ax.add_patch(rect)
                    ax.text(x_cursor + l/2, w/2, f"P{idx+1}", ha='center', va='center')
                    x_cursor += l
                else:
                    st.warning(f"Parcel {idx+1} exceeds truck length in {vehicle_name}. Skipping.")

            ax.set_aspect('equal')
            st.pyplot(fig)
    else:
        st.warning("❌ No feasible assignment found.")
