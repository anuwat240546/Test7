import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
@st.cache_resource
def load_assets():
    sc_in = joblib.load('scaler_All_input.pkl')
    sc_out = joblib.load('scaler_All_output.pkl')
    model = load_model('model_All_1_(256, 256, 128).h5', custom_objects={'mse': MeanSquaredError()})
    return sc_in, sc_out, model
scaler, all_output, model_All = load_assets()
st.title("Machine Learning-Based Prediction for Mechanically Stabilized Earth Walls")
st.subheader("Fast and accurate estimation of Displacement and Factor of Safety (FOS) using Deep Learning trained on Finite Element Method.")
st.subheader("Please complete all fields to proceed.")
user1 = st.number_input("Height of wall (m) : ", min_value=0.1,value=6.0)
user2 = st.number_input("Distance behind the wall (m) : ",min_value=0.1,value=15.0)
user3 = st.number_input("Spacing of Geogrid (m) : ",min_value=0.1,value=0.5)
user4 = st.number_input("Length of Geogrid (m) : ",min_value=0.1,value=6.0)
user5 = st.number_input("Axial Stiffness of Geogrid (KN/m) : ",min_value=0.1,value=3200.0)
if st.button("ENTER", type="primary"):
    user_input = {
        'Spacing of Geogrid (m)': user3/user4,
        'Length of Geogrid (m)': user4/user1,
        'EA of Geogrid (KN/m)': user5,
        'Distance behind the wall' : user2/15}
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    All_pred = model_All.predict(user_scaled)
    All_pred_scaled = all_output.inverse_transform(All_pred).flatten()
    hor_disp = All_pred_scaled[0:16]
    ver_settle = All_pred_scaled[16:32]
    fos_value = All_pred_scaled[32]
    fig = go.Figure()
    # 1. วาดเส้นระดับดินเดิม (Original Ground Surface)
    dist_levels = np.linspace(0, user2, 16)
    original_ground = np.full(16, user1) 
    fig.add_trace(go.Scatter(x=dist_levels, y=original_ground, name="Original Ground", line=dict(color='grey', dash='dash')))
    # 2. วาดการทรุดตัวของผิวดิน (Settled Ground Surface)
    height_levels = np.linspace(-1.5, user1, 16)
    settled_ground = original_ground + ver_settle
    fig.add_trace(go.Scatter(x=dist_levels, y=settled_ground, name="Settled Surface", fill='tozeroy'))
    # 3. วาดการเคลื่อนตัวของกำแพง (Wall Displacement) 
    fig.add_trace(go.Scatter(x=hor_disp, y=height_levels, mode='lines+markers',name='Wall Displacement',line=dict(color='royalblue', width=4)))
    # 4. วาดตำแหน่งกำแพงเดิม (จุดอ้างอิง x=0)
    fig.add_trace(go.Scatter(x=[0, 0], y=[-1.5, user1],mode='lines',name='Original Wall Position',line=dict(color='black', width=1)))
    fig.add_trace(go.Scatter(x=[-1, 1], y=[-1.5, -1.5],mode='lines',name='Footing',line=dict(color='black', width=4)))
    # ตั้งค่า Layout ให้เหมือนรูปตัดขวาง
    fig.update_layout(title="MSE Wall Deformation Cross-section",xaxis_title="Horizontal Distance / Displacement (m)",yaxis_title="Elevation (m)",width=900, height=600,
        yaxis=dict(scaleanchor="x",scaleratio=1,range=[-2, user1 + 2]),xaxis=dict(range=[min(hor_disp) - 1, user2 + 1]),template="plotly_white",margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=False)

    st.divider()
    st.metric(label="Maximum Horizontal Displacement (m)", value=f"{min(hor_disp):.3f} m")
    st.divider()
    st.metric(label="Maximum Vertical Displacement (m)", value=f"{min(ver_settle):.3f} m")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Factor of Safety (FOS)", value=f"{fos_value:.3f}")
    with col2:
        if fos_value < 1.0:
            st.error("Status: Unsafety (Stable Failure)")
        else:
            st.success("Status: Safety")
else :
    st.info("Please complete all fields to proceed.")

