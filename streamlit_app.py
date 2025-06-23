import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set page config and background
st.set_page_config(page_title="Truck Optimizer", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #48cae4;
    }
    </style>
""", unsafe_allow_html=True)

# --- Vehicle Data with dimensions ---
vehicle_data = [
    ("Small van", 1.5, 1.2, 1, 350, 1.8, 80),
    ("Short wheel base", 2, 1.2, 1.3, 800, 2.4, 95),
    ("Medium wheel base", 3, 1.2, 1.9, 1400, 3.6, 50),
    ("4 meter sprinter", 4.2, 1.25, 1.9, 1250, 5.25, 85),
    ("luton van", 4, 2, 2, 1000, 8, 110),
    ("7.5 tonne", 6, 2.3, 2.3, 2800, 13.8, 100),
    ("18 tonne", 7.3, 2.4, 2.4, 9000, 17.52, 130),
    ("26 tonne", 8, 2.4, 2.5, 15000, 19.2, 140),
    ("arctic", 13.5, 2.5, 2.7, 24000, 33.75, 200),
]

# Create multiple copies of each vehicle
vehicles = {}
for i in range(1, 6):
    for name, l, w, h, wt, ar, cost in vehicle_data:
        vehicles[f"{name}{i if i > 1 else ''}"] = {
            "max_length": l,
            "max_width": w,
            "max_height": h,
            "max_weight": wt,
            "max_area": ar,
            "cost": cost
        }

st.title("🚚 Truck Optimizer")
st.write("Enter parcel dimensions and weight to find the cheapest truck configuration.")

# User inputs
num_parcels = st.number_input("Number of Parcels", min_value=1, max_value=20, value=2)

weights, areas, lengths, widths, heights = [], [], [], [], []

cols = st.columns(6)
for i in range(num_parcels):
    with cols[0]:
        weights.append(st.number_input(f"Parcel {i+1} Weight (kg)", value=100.0, key=f"w_{i}"))
    with cols[1]:
        areas.append(st.number_input(f"Parcel {i+1} Area (m²)", value=1.0, key=f"a_{i}"))
    with cols[2]:
        lengths.append(st.number_input(f"Parcel {i+1} Length (m)", value=1.0, key=f"l_{i}"))
    with cols[3]:
        widths.append(st.number_input(f"Parcel {i+1} Width (m)", value=1.0, key=f"wd_{i}"))
    with cols[4]:
        heights.append(st.number_input(f"Parcel {i+1} Height (m)", value=1.0, key=f"h_{i}"))

if st.button("Run Optimization"):
    model = pulp.LpProblem("Truck Optimization", pulp.LpMinimize)

    # Decision Variables
    x = pulp.LpVariable.dicts("Assign", ((i, j) for i in range(num_parcels) for j in vehicles), cat="Binary")
    y = pulp.LpVariable.dicts("UseVehicle", (j for j in vehicles), cat="Binary")

    # Objective: minimize cost
    model += pulp.lpSum(vehicles[j]["cost"] * y[j] for j in vehicles)

    # Each parcel must be assigned to one vehicle
    for i in range(num_parcels):
        model += pulp.lpSum(x[i, j] for j in vehicles) == 1

    # Vehicle capacity and size constraints
    for j in vehicles:
        model += pulp.lpSum(weights[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["max_weight"] * y[j]
        model += pulp.lpSum(areas[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["max_area"] * y[j]

        for i in range(num_parcels):
            model += lengths[i] * x[i, j] <= vehicles[j]["max_length"] * y[j]
            model += widths[i] * x[i, j] <= vehicles[j]["max_width"] * y[j]
            model += heights[i] * x[i, j] <= vehicles[j]["max_height"] * y[j]

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    st.subheader("Optimization Results")
    st.write(f"**Status:** {pulp.LpStatus[model.status]}")
    st.write(f"**Total Cost:** £{pulp.value(model.objective):.2f}")

    results = []
    for j in vehicles:
        if pulp.value(y[j]) == 1:
            for i in range(num_parcels):
                if pulp.value(x[i, j]) == 1:
                    results.append({
                        "Vehicle": j,
                        "Parcel #": i + 1,
                        "Weight (kg)": weights[i],
                        "Area (m²)": areas[i],
                        "Length (m)": lengths[i],
                        "Width (m)": widths[i],
                        "Height (m)": heights[i],
                        "Cost (£)": vehicles[j]["cost"]
                    })

    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("Download Results", csv, "truck_assignment.csv", "text/csv")

        # --- 2D Visualization Section ---
        st.subheader("📦 Parcel Placement in Selected Trucks (Top-Down View)")

        used_vehicles = set(df["Vehicle"])
        for v in used_vehicles:
            vehicle_info = vehicles[v]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_title(f"Vehicle: {v} ({vehicle_info['max_length']}m × {vehicle_info['max_width']}m)")
            ax.set_xlim(0, vehicle_info['max_length'])
            ax.set_ylim(0, vehicle_info['max_width'])
            ax.set_aspect('equal')
            ax.add_patch(patches.Rectangle((0, 0), vehicle_info['max_length'], vehicle_info['max_width'], edgecolor='black', facecolor='lightgrey'))

            x_offset = 0
            y_offset = 0
            max_row_height = 0.0

            parcels = df[df["Vehicle"] == v]
            for _, row in parcels.iterrows():
                pl = row["Length (m)"]
                pw = row["Width (m)"]
                pid = row["Parcel #"]

                if x_offset + pl > vehicle_info['max_length']:
                    x_offset = 0
                    y_offset += max_row_height
                    max_row_height = 0

                if y_offset + pw > vehicle_info['max_width']:
                    st.warning(f"Parcel {pid} visually overflows truck area — adjust parcel size or layout.")
                    continue

                rect = patches.Rectangle((x_offset, y_offset), pl, pw, edgecolor='blue', facecolor='skyblue', lw=2)
                ax.add_patch(rect)
                ax.text(x_offset + pl/2, y_offset + pw/2, f"P{int(pid)}", ha='center', va='center', fontsize=8, color='black')

                x_offset += pl
                max_row_height = max(max_row_height, pw)

            st.pyplot(fig)
    else:
        st.warning("No solution found. Please adjust parcel dimensions or weights.")
