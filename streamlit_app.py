
import streamlit as st
import pulp

# --- Vehicle Data ---
vehicles = {
    0: {"cost": 500, "max_weight": 1000, "max_area": 20},
    1: {"cost": 600, "max_weight": 1200, "max_area": 22},
    2: {"cost": 700, "max_weight": 1500, "max_area": 25},
    3: {"cost": 800, "max_weight": 1800, "max_area": 30},
    4: {"cost": 950, "max_weight": 2000, "max_area": 35},
    5: {"cost": 1100, "max_weight": 2200, "max_area": 40},
    6: {"cost": 1250, "max_weight": 2500, "max_area": 45},
    7: {"cost": 1400, "max_weight": 2700, "max_area": 50},
    8: {"cost": 1600, "max_weight": 3000, "max_area": 60},
    9: {"cost": 500, "max_weight": 1000, "max_area": 20},
}

st.title("🚛 Truck Optimization Tool")
st.markdown("Select the cheapest combination of trucks to carry parcels based on weight and size using linear programming.")

num_parcels = st.number_input("Number of Parcels", min_value=1, max_value=20, step=1)

weights = []
areas = []

st.subheader("Enter Parcel Details")
for i in range(num_parcels):
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input(f"Parcel {i+1} Weight (kg)", key=f"w_{i}", min_value=0.0, value=500.0)
    with col2:
        area = st.number_input(f"Parcel {i+1} Area (m²)", key=f"a_{i}", min_value=0.0, value=20.0)
    weights.append(weight)
    areas.append(area)

if st.button("Run Optimization"):
    try:
        model = pulp.LpProblem("Minimize_Truck_Usage_Cost", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("Assign", ((i, j) for i in range(num_parcels) for j in vehicles), cat="Binary")
        y = pulp.LpVariable.dicts("UseVehicle", (j for j in vehicles), cat="Binary")

        model += pulp.lpSum(vehicles[j]["cost"] * y[j] for j in vehicles), "TotalCost"

        for i in range(num_parcels):
            model += pulp.lpSum(x[i, j] for j in vehicles) == 1

        for j in vehicles:
            model += pulp.lpSum(weights[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["max_weight"] * y[j]
            model += pulp.lpSum(areas[i] * x[i, j] for i in range(num_parcels)) <= vehicles[j]["max_area"] * y[j]

        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        st.success(f"Optimization Status: {pulp.LpStatus[model.status]}")
        st.write(f"**Total Cost:** {pulp.value(model.objective)}")

        for j in vehicles:
            if pulp.value(y[j]) == 1:
                st.markdown(f"### ✅ Vehicle {j} Used")
                st.write(f"- Cost: {vehicles[j]['cost']}")
                st.write(f"- Max Weight: {vehicles[j]['max_weight']}")
                st.write(f"- Max Area: {vehicles[j]['max_area']}")
                for i in range(num_parcels):
                    if pulp.value(x[i, j]) == 1:
                        st.write(f"  - Parcel {i+1} ➜ Weight: {weights[i]}, Area: {areas[i]}")

    except Exception as e:
        st.error(f"Error: {e}")
