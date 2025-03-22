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

# --- NUEVA FUNCIONALIDAD: Cálculo de opciones adicionales para Delta Neutral ---
def calcular_opciones_adicionales_delta_neutral(delta_actual, delta_opcion_existente):
    if delta_opcion_existente != 0:
        return -delta_actual / delta_opcion_existente
    else:
        return np.nan

# --- UI Streamlit ---
st.title("Análisis Interactivo de Griegas")

st.sidebar.header("Parámetros del Mercado")
S_actual = st.sidebar.number_input("Precio actual del subyacente (S)", value=7500.0)
r = st.sidebar.number_input("Tasa libre de riesgo (%)", value=29.5) / 100
sigma = st.sidebar.number_input("Volatilidad (%)", value=57.0) / 100
dias = st.sidebar.number_input("Días al vencimiento", value=26)
T = dias / 365

st.sidebar.markdown("---")
st.sidebar.write("Rango de precios para análisis")
S_min = st.sidebar.number_input("Precio mínimo", value=6000.0)
S_max = st.sidebar.number_input("Precio máximo", value=9000.0)

# Portafolio ejemplo
portafolio = pd.DataFrame({
    'tipo': ['accion', 'opcion', 'opcion'],
    'cantidad': [5000, 4000, -19000],
    'strike': [np.nan, 7578.3, 8578.3],
    'tipo_opcion': [None, 'call', 'call'],
    
})

# Calcular griegas individuales y agregarlas al portafolio antes de mostrar
deltas = []
gammas = []
thetas = []
vegas = []
delta_unit = []
gamma_unit = []
theta_unit = []
vega_unit = []
precios_actuales_calculados = []

for _, fila in portafolio.iterrows():
    tipo = fila['tipo']
    cantidad = fila['cantidad']
    K = fila['strike']
    tipo_opcion = fila['tipo_opcion']

    if tipo == 'accion':
        du = 1
        gu = 0
        tu = 0
        vu = 0
        pc = 0
    elif tipo == 'opcion' and pd.notna(K):
        d1 = (np.log(S_actual / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        gu = norm.pdf(d1) / (S_actual * sigma * np.sqrt(T))
        vu = (S_actual * norm.pdf(d1) * np.sqrt(T)) / 100

        if tipo_opcion == 'call':
            du = norm.cdf(d1)
            tu = ((- (S_actual * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
            pc = S_actual * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif tipo_opcion == 'put':
            du = norm.cdf(d1) - 1
            tu = ((- (S_actual * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
            pc = K * np.exp(-r * T) * norm.cdf(-d2) - S_actual * norm.cdf(-d1)
        else:
            du = tu = gu = vu = pc = 0
    else:
        du = tu = gu = vu = pc = 0

    delta_unit.append(du)
    gamma_unit.append(gu)
    theta_unit.append(tu)
    vega_unit.append(vu)
    deltas.append(du * cantidad)
    gammas.append(gu * cantidad)
    thetas.append(tu * cantidad)
    vegas.append(vu * cantidad)
    precios_actuales_calculados.append(pc)

portafolio['Delta Unitario'] = delta_unit
portafolio['Gamma Unitario'] = gamma_unit
portafolio['Theta Unitario'] = theta_unit
portafolio['Vega Unitario'] = vega_unit
portafolio['Delta'] = deltas
portafolio['Gamma'] = gammas
portafolio['Theta'] = thetas
portafolio['Vega'] = vegas

st.header("Portafolio")
st.data_editor(portafolio, num_rows="dynamic", use_container_width=True)

# Cálculo para gráfico
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
st.subheader("Totales del Portafolio")
total_delta = portafolio['Delta'].sum()
total_gamma = portafolio['Gamma'].sum()
total_theta = portafolio['Theta'].sum()
total_vega = portafolio['Vega'].sum()

st.markdown(f"**Delta total:** {total_delta:.2f}")
st.markdown(f"**Gamma total:** {total_gamma:.2f}")
st.markdown(f"**Theta total:** {total_theta:.2f}")
st.markdown(f"**Vega total:** {total_vega:.2f}")
