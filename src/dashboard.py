import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fleet Equipment Health Dashboard",
    page_icon="🏭",
    layout="wide"
)

# ── Load model and data ───────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/ai4i2020.csv")
    return df

model, scaler = load_model()
df = load_data()

# ── Feature engineering (match training) ─────────────────────
def prepare_features(df):
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    df = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    df['Torque_x_Toolwear'] = df['Torque [Nm]'] * df['Tool wear [min]']
    df['Temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    return df

df_model = prepare_features(df)
X = df_model.drop(columns=['Machine failure'])
X_scaled = scaler.transform(X)
df['Failure_Probability'] = model.predict_proba(X_scaled)[:, 1]
df['Risk'] = pd.cut(
    df['Failure_Probability'],
    bins=[-0.01, 0.3, 0.6, 1.01],
    labels=['Healthy', 'At Risk', 'Critical']
)

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.title("Filters")
type_filter = st.sidebar.multiselect(
    "Machine Type", options=['H', 'L', 'M'], default=['H', 'L', 'M']
)
risk_filter = st.sidebar.multiselect(
    "Risk Level", options=['Healthy', 'At Risk', 'Critical'],
    default=['Healthy', 'At Risk', 'Critical']
)
prob_threshold = st.sidebar.slider(
    "Failure Probability Threshold", 0.0, 1.0, 0.5, 0.05
)

filtered = df[df['Type'].isin(type_filter) & df['Risk'].isin(risk_filter)]

# ── Title ─────────────────────────────────────────────────────
st.title("Fleet-Wide Equipment Health Dashboard")
st.markdown("Real-time machine failure risk monitoring powered by Random Forest (ROC-AUC: 0.97)")

st.divider()

# ── KPI Cards ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
total = len(filtered)
healthy = (filtered['Risk'] == 'Healthy').sum()
at_risk = (filtered['Risk'] == 'At Risk').sum()
critical = (filtered['Risk'] == 'Critical').sum()

col1.metric("Total Machines", f"{total:,}")
col2.metric("Healthy", f"{healthy:,}", delta=None)
col3.metric("At Risk", f"{at_risk:,}")
col4.metric("Critical", f"{critical:,}")

st.divider()

# ── Row 1: Risk distribution + Failure probability histogram ──
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Fleet Risk Distribution")
    risk_counts = filtered['Risk'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    colors = {'Healthy': '#2ecc71', 'At Risk': '#f39c12', 'Critical': '#e74c3c'}
    fig_pie = px.pie(
        risk_counts, values='Count', names='Risk Level',
        color='Risk Level', color_discrete_map=colors
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.subheader("Failure Probability Distribution")
    fig_hist = px.histogram(
        filtered, x='Failure_Probability', nbins=50,
        color_discrete_sequence=['#2E75B6'],
        labels={'Failure_Probability': 'Failure Probability'}
    )
    fig_hist.add_vline(
        x=prob_threshold, line_dash="dash",
        line_color="red", annotation_text=f"Threshold: {prob_threshold}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Row 2: Sensor scatter plots ───────────────────────────────
st.subheader("Sensor Analysis — Torque vs Tool Wear by Risk Level")
fig_scatter = px.scatter(
    filtered, x='Tool wear [min]', y='Torque [Nm]',
    color='Risk', color_discrete_map=colors,
    hover_data=['Type', 'Failure_Probability'],
    opacity=0.6
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── Row 3: Machine type breakdown ─────────────────────────────
st.subheader("Failure Rate by Machine Type")
type_stats = filtered.groupby('Type').agg(
    Total=('Machine failure', 'count'),
    Failures=('Machine failure', 'sum'),
    Avg_Probability=('Failure_Probability', 'mean')
).reset_index()
type_stats['Failure Rate (%)'] = (type_stats['Failures'] / type_stats['Total'] * 100).round(2)

col_c, col_d = st.columns(2)
with col_c:
    fig_bar = px.bar(
        type_stats, x='Type', y='Failure Rate (%)',
        color='Type', text='Failure Rate (%)',
        color_discrete_sequence=['#2E75B6', '#E74C3C', '#2ECC71']
    )
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_d:
    st.dataframe(
        type_stats[['Type', 'Total', 'Failures', 'Failure Rate (%)', 'Avg_Probability']].round(3),
        use_container_width=True
    )

st.divider()

# ── Row 4: Alert table ────────────────────────────────────────
st.subheader("High Risk Machine Alerts")
high_risk = filtered[filtered['Failure_Probability'] >= prob_threshold].sort_values(
    'Failure_Probability', ascending=False
).head(20)[['UDI', 'Type', 'Air temperature [K]',
             'Rotational speed [rpm]', 'Torque [Nm]',
             'Tool wear [min]', 'Failure_Probability', 'Risk']]

st.dataframe(
    high_risk.style.background_gradient(
        subset=['Failure_Probability'], cmap='RdYlGn_r'
    ).format({'Failure_Probability': '{:.3f}'}),
    use_container_width=True
)

st.divider()

# ── Row 5: Live prediction form ───────────────────────────────
st.subheader("Live Failure Prediction")
st.markdown("Enter sensor readings to get an instant failure prediction from the model.")

with st.form("prediction_form"):
    p1, p2, p3 = st.columns(3)
    with p1:
        type_input = st.selectbox("Machine Type", ['L', 'M', 'H'])
        air_temp = st.number_input("Air Temperature [K]", value=300.0, step=0.1)
        process_temp = st.number_input("Process Temperature [K]", value=310.0, step=0.1)
    with p2:
        rot_speed = st.number_input("Rotational Speed [rpm]", value=1500, step=10)
        torque = st.number_input("Torque [Nm]", value=40.0, step=0.5)
    with p3:
        tool_wear = st.number_input("Tool Wear [min]", value=100, step=1)

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(['H', 'L', 'M'])
    type_encoded = le.transform([type_input])[0]
    torque_x_toolwear = torque * tool_wear
    temp_diff = process_temp - air_temp

    features = np.array([[type_encoded, air_temp, process_temp,
                          rot_speed, torque, tool_wear,
                          torque_x_toolwear, temp_diff]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if pred == 1:
        st.error(f"FAILURE DETECTED — Failure Probability: {prob:.1%}")
    else:
        st.success(f"Normal Operation — Failure Probability: {prob:.1%}")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Failure Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if prob > 0.5 else "green"},
            'steps': [
                {'range': [0, 30], 'color': '#2ecc71'},
                {'range': [30, 60], 'color': '#f39c12'},
                {'range': [60, 100], 'color': '#e74c3c'}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
