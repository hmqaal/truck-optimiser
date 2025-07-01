import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Page Config ---
st.set_page_config(page_title="Truck Optimizer", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #48cae4;
    }
    </style>
""", unsafe_allow_html=True)

# --- Vehicle Data ---
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

# Create vehicle variations
vehicles = {}
for i in range(1, 50):
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

# --- Bulk Parcel Entry ---
st.subheader("Bulk Parcel Entry")
bulk_weights, bulk_lengths, bulk_widths, bulk_heights = [], [], [], []
num_groups = st.number_input("Number of Bulk Parcel Groups", min_value=0, max_value=10, value=0)

for g in range(num_groups):
    st.markdown(f"**Group {g+1}**")
    count = st.number_input(f"Number of Parcels in Group {g+1}", min_value=1, max_value=200, value=10, key=f"count_{g}")
    weight = st.number_input(f"Weight per Parcel (kg) - Group {g+1}", value=100.0, key=f"bw_{g}")
    length = st.number_input(f"Length per Parcel (m) - Group {g+1}", value=1.0, key=f"bl_{g}")
    width = st.number_input(f"Width per Parcel (m) - Group {g+1}", value=1.0, key=f"bwth_{g}")
    height = st.number_input(f"Height per Parcel (m) - Group {g+1}", value=1.0, key=f"bh_{g}")

    bulk_weights.extend([weight] * count)
    bulk_lengths.extend([length] * count)
    bulk_widths.extend([width] * count)
    bulk_heights.extend([height] * count)

# Calculate bulk areas from dimensions
bulk_areas = [l * w for l, w in zip(bulk_lengths, bulk_widths)]

# --- Manual Parcel Entry ---
st.subheader("Manual Parcel Entry")
num_parcels = st.number_input("Number of Individual Parcels", min_value=0, max_value=50, value=2)
manual_weights, manual_lengths, manual_widths, manual_heights = [], [], [], []

cols = st.columns(5)
for i in range(num_parcels):
    with cols[0]:
        manual_weights.append(st.number_input(f"Parcel {i+1} Weight (kg)", value=100.0, key=f"w_{i}"))
    with cols[1]:
        manual_lengths.append(st.number_input(f"Parcel {i+1} Length (m)", value=1.0, key=f"l_{i}"))
    with cols[2]:
        manual_widths.append(st.number_input(f"Parcel {i+1} Width (m)", value=1.0, key=f"wd_{i}"))
    with cols[3]:
        manual_heights.append(st.number_input(f"Parcel {i+1} Height (m)", value=1.0, key=f"h_{i}"))

# Calculate manual areas from dimensions
manual_areas = [l * w for l, w in zip(manual_lengths, manual_widths)]

# Combine all parcels
weights = bulk_weights + manual_weights
lengths = bulk_lengths + manual_lengths
widths = bulk_widths + manual_widths
heights = bulk_heights + manual_heights
areas = bulk_areas + manual_areas
num_total_parcels = len(weights)

# --- Run Optimization ---
if st.button("Run Optimization"):
    model = pulp.LpProblem("Truck Optimization", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("Assign", ((i, j) for i in range(num_total_parcels) for j in vehicles), cat="Binary")
    y = pulp.LpVariable.dicts("UseVehicle", (j for j in vehicles), cat="Binary")

    model += pulp.lpSum(vehicles[j]["cost"] * y[j] for j in vehicles)

    for i in range(num_total_parcels):
        model += pulp.lpSum(x[i, j] for j in vehicles) == 1

    for j in vehicles:
        model += pulp.lpSum(weights[i] * x[i, j] for i in range(num_total_parcels)) <= vehicles[j]["max_weight"] * y[j]
        model += pulp.lpSum(areas[i] * x[i, j] for i in range(num_total_parcels)) <= vehicles[j]["max_area"] * y[j]

        for i in range(num_total_parcels):
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
            for i in range(num_total_parcels):
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

        # --- Visualization ---
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

