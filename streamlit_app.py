
import streamlit as st
import pulp

# --- Vehicle Data ---
vehicles = {
    "Small van": {"cost": 80, "max_weight": 350, "max_area": 1.8},
    "Short wheel base": {"cost": 95, "max_weight": 800, "max_area": 2.4},
    "Medium wheel base": {"cost": 50, "max_weight": 1400, "max_area": 3.6},
    "4 meter sprinter": {"cost": 85, "max_weight": 1250, "max_area": 5.25},
    "luton van": {"cost": 110, "max_weight": 1000, "max_area": 8},
    "7.5 tonne": {"cost": 100, "max_weight": 2800, "max_area": 13.8},
    "18 tonne": {"cost": 130, "max_weight": 9000, "max_area": 17.52},
    "26 tonne": {"cost": 140, "max_weight": 15000, "max_area": 19.2},
    "arctic": {"cost": 200, "max_weight": 24000, "max_area": 33.75},
    "Small van2": {"cost": 80, "max_weight": 350, "max_area": 1.8},
    "Short wheel base2": {"cost": 95, "max_weight": 800, "max_area": 2.4},
    "Medium wheel base2": {"cost": 50, "max_weight": 1400, "max_area": 3.6},
    "4 meter sprinter2": {"cost": 85, "max_weight": 1250, "max_area": 5.25},
    "luton van2": {"cost": 110, "max_weight": 1000, "max_area": 8},
    "7.5 tonne2": {"cost": 100, "max_weight": 2800, "max_area": 13.8},
    "18 tonne2": {"cost": 130, "max_weight": 9000, "max_area": 17.52},
    "26 tonne2": {"cost": 140, "max_weight": 15000, "max_area": 19.2},
    "arctic2": {"cost": 200, "max_weight": 24000, "max_area": 33.75},
    "Small van3": {"cost": 80, "max_weight": 350, "max_area": 1.8},
    "Short wheel base3": {"cost": 95, "max_weight": 800, "max_area": 2.4},
    "Medium wheel base3": {"cost": 50, "max_weight": 1400, "max_area": 3.6},
    "4 meter sprinter3": {"cost": 85, "max_weight": 1250, "max_area": 5.25},
    "luton van3": {"cost": 110, "max_weight": 1000, "max_area": 8},
    "7.5 tonne3": {"cost": 100, "max_weight": 2800, "max_area": 13.8},
    "18 tonne3": {"cost": 130, "max_weight": 9000, "max_area": 17.52},
    "26 tonne3": {"cost": 140, "max_weight": 15000, "max_area": 19.2},
    "arctic3": {"cost": 200, "max_weight": 24000, "max_area": 33.75},
    "Small van4": {"cost": 80, "max_weight": 350, "max_area": 1.8},
    "Short wheel base4": {"cost": 95, "max_weight": 800, "max_area": 2.4},
    "Medium wheel base4": {"cost": 50, "max_weight": 1400, "max_area": 3.6},
    "4 meter sprinter4": {"cost": 85, "max_weight": 1250, "max_area": 5.25},
    "luton van4": {"cost": 110, "max_weight": 1000, "max_area": 8},
    "7.5 tonne4": {"cost": 100, "max_weight": 2800, "max_area": 13.8},
    "18 tonne4": {"cost": 130, "max_weight": 9000, "max_area": 17.52},
    "26 tonne4": {"cost": 140, "max_weight": 15000, "max_area": 19.2},
    "arctic4": {"cost": 200, "max_weight": 24000, "max_area": 33.75},
    "Small van5": {"cost": 80, "max_weight": 350, "max_area": 1.8},
    "Short wheel base5": {"cost": 95, "max_weight": 800, "max_area": 2.4},
    "Medium wheel base5": {"cost": 50, "max_weight": 1400, "max_area": 3.6},
    "4 meter sprinter5": {"cost": 85, "max_weight": 1250, "max_area": 5.25},
    "luton van5": {"cost": 110, "max_weight": 1000, "max_area": 8},
    "7.5 tonne5": {"cost": 100, "max_weight": 2800, "max_area": 13.8},
    "18 tonne5": {"cost": 130, "max_weight": 9000, "max_area": 17.52},
    "26 tonne5": {"cost": 140, "max_weight": 15000, "max_area": 19.2},
    "arctic5": {"cost": 200, "max_weight": 24000, "max_area": 33.75},
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
