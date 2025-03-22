import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- Función para calcular griegas ---
def calcular_griegas(portafolio, S_range, r, sigma, T):
    delta_total = np.zeros_like(S_range)
    theta_total = np.zeros_like(S_range)

    for _, fila in portafolio.iterrows():
        tipo = fila['tipo']
        cantidad = fila['cantidad']
        K = fila['strike']
        tipo_opcion = fila['tipo_opcion']

        if tipo == 'accion':
            delta_total += cantidad  # Cada acción tiene delta = 1
        elif tipo == 'opcion' and pd.notna(K):
            d1 = (np.log(S_range / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if tipo_opcion == 'call':
                delta = norm.cdf(d1)
                theta = (- (S_range * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                         - r * K * np.exp(-r * T) * norm.cdf(d2))
            elif tipo_opcion == 'put':
                delta = norm.cdf(d1) - 1
                theta = (- (S_range * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                         + r * K * np.exp(-r * T) * norm.cdf(-d2))
            else:
                continue

            delta_total += cantidad * delta
            theta_total += (cantidad * theta)/365

    return delta_total, theta_total

# --- UI Streamlit ---
st.title("Análisis Interactivo de Griegas con Plotly")

st.sidebar.header("Parámetros del Mercado")
S_actual = st.sidebar.number_input("Precio actual del subyacente (S)", value=7500.0)
r = st.sidebar.number_input("Tasa libre de riesgo (%)", value=29.5) / 100
sigma = st.sidebar.number_input("Volatilidad (%)", value=57.0) / 100
dias = st.sidebar.number_input("Días al vencimiento", value=26)
T = dias / 365

st.sidebar.markdown("---")
S_min = st.sidebar.number_input("Precio mínimo", value=6000.0)
S_max = st.sidebar.number_input("Precio máximo", value=9000.0)

# Portafolio editable
ejemplo = pd.DataFrame({
    'tipo': ['accion', 'opcion', 'opcion'],
    'cantidad': [5000, 4000, -19000],
    'strike': [np.nan, 7578.3, 8578.3],
    'tipo_opcion': [None, 'call', 'call']
})

st.header("Cargar Portafolio")
portafolio = st.data_editor(ejemplo, num_rows="dynamic", use_container_width=True)

# Calcular griegas individuales actualizadas
updated_deltas = []
updated_gammas = []
updated_thetas = []
updated_vegas = []
updated_delta_unit = []
updated_gamma_unit = []
updated_theta_unit = []
updated_vega_unit = []

for _, fila in portafolio.iterrows():
    tipo = fila['tipo']
    cantidad = fila['cantidad']
    K = fila['strike']
    tipo_opcion = fila['tipo_opcion']

    if tipo == 'accion':
        du, gu, tu, vu = 1, 0, 0, 0
    elif tipo == 'opcion' and pd.notna(K):
        d1 = (np.log(S_actual / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        gu = norm.pdf(d1) / (S_actual * sigma * np.sqrt(T))
        vu = (S_actual * norm.pdf(d1) * np.sqrt(T)) / 100

        if tipo_opcion == 'call':
            du = norm.cdf(d1)
            tu = ((- (S_actual * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
        elif tipo_opcion == 'put':
            du = norm.cdf(d1) - 1
            tu = ((- (S_actual * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
        else:
            du = tu = gu = vu = 0
    else:
        du = tu = gu = vu = 0

    updated_delta_unit.append(du)
    updated_gamma_unit.append(gu)
    updated_theta_unit.append(tu)
    updated_vega_unit.append(vu)
    updated_deltas.append(du * cantidad)
    updated_gammas.append(gu * cantidad)
    updated_thetas.append(tu * cantidad)
    updated_vegas.append(vu * cantidad)

portafolio['Delta Unitario'] = updated_delta_unit
portafolio['Gamma Unitario'] = updated_gamma_unit
portafolio['Theta Unitario'] = updated_theta_unit
portafolio['Vega Unitario'] = updated_vega_unit
portafolio['Delta'] = updated_deltas
portafolio['Gamma'] = updated_gammas
portafolio['Theta'] = updated_thetas
portafolio['Vega'] = updated_vegas

# Cálculo para gráfico actualizado
S_range = np.linspace(S_min, S_max, 100)
delta_total, theta_total = calcular_griegas(portafolio, S_range, r, sigma, T)

# Gráfico 3D
norm = mcolors.Normalize(vmin=min(theta_total), vmax=max(theta_total))
colors = [mcolors.to_hex(cm.viridis(norm(val))) for val in theta_total]

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=S_range,
    y=delta_total,
    z=theta_total,
    mode='lines+markers',
    marker=dict(size=4, color=colors),
    line=dict(color='grey'),
    name='Delta vs Theta en función de S'
))

# Punto actual con etiqueta
delta_actual_punto = np.interp(S_actual, S_range, delta_total)
theta_actual_punto = np.interp(S_actual, S_range, theta_total)
fig.add_trace(go.Scatter3d(
    x=[S_actual],
    y=[delta_actual_punto],
    z=[theta_actual_punto],
    mode='markers+text',
    marker=dict(size=6, color='red', symbol='diamond'),
    text=[f"S={S_actual}<br>Δ={delta_actual_punto:.2f}<br>Θ={theta_actual_punto:.2f}"],
    textposition='top center',
    name='Punto Actual'
))

fig.update_layout(
    title="Relación Delta y Theta del Portafolio según Precio Subyacente",
    scene=dict(
        xaxis_title='Precio Subyacente',
        yaxis_title='Delta',
        zaxis_title='Theta',
    ),
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# Mostrar griegas totales del portafolio
st.subheader("Griegas Totales del Portafolio")
total_delta = portafolio['Delta'].sum()
total_gamma = portafolio['Gamma'].sum()
total_theta = portafolio['Theta'].sum()
total_vega = portafolio['Vega'].sum()

st.markdown(f"**Delta total:** {total_delta:.2f}")
st.markdown(f"**Gamma total:** {total_gamma:.2f}")
st.markdown(f"**Theta total:** {total_theta:.2f}")
st.markdown(f"**Vega total:** {total_vega:.2f}")
