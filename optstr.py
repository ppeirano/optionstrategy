import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Clase para representar una opción
class Opcion:
    def __init__(self, tipo, operacion, strike, prima, cantidad):
        self.tipo = tipo  # 'Call' o 'Put'
        self.operacion = operacion  # 'Comprada' o 'Vendida'
        self.strike = strike  # Precio de ejercicio
        self.prima = prima  # Prima pagada o recibida
        self.cantidad = cantidad*100  # Cantidad de opciones

    def calcular_pnl(self, S):
        """Calcula el P&L basado en el tipo de opción y el precio del subyacente."""
        if self.tipo == 'Call':
            if self.operacion == 'Comprada':
                pnl = np.maximum(0, S - self.strike) - self.prima  # P&L de una opción call comprada
            else:  # 'Vendida'
                pnl = self.prima - np.maximum(0, S - self.strike)  # P&L de una opción call vendida
        else:  # 'Put'
            if self.operacion == 'Comprada':
                pnl = np.maximum(0, self.strike - S) - self.prima  # P&L de una opción put comprada
            else:  # 'Vendida'
                pnl = self.prima - np.maximum(0, self.strike - S)  # P&L de una opción put vendida
        return pnl * self.cantidad  # Multiplicar por la cantidad

    def calcular_pnl_bs(self, S, T, r, sigma):
        """Calcula el P&L usando Black-Scholes."""
        d1 = (np.log(S / self.strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.tipo == 'Call':
            precio_opcion = S * norm.cdf(d1) - self.strike * np.exp(-r * T) * norm.cdf(d2)
            if self.operacion == 'Comprada':
                pnl_bs = precio_opcion - self.prima  # Call comprada
            else:  # 'Vendida'
                pnl_bs = self.prima - precio_opcion  # Call vendida
        else:  # 'Put'
            precio_opcion = self.strike * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            if self.operacion == 'Comprada':
                pnl_bs = precio_opcion - self.prima  # Put comprada
            else:  # 'Vendida'
                pnl_bs = self.prima - precio_opcion  # Put vendida
            
        return pnl_bs * self.cantidad  # Multiplicar por la cantidad

# Clase para representar el portafolio
class Portafolio:
    def __init__(self):
        self.opciones = []

    def agregar_opcion(self, opcion):
        self.opciones.append(opcion)

    def eliminar_opcion(self, index):
        if 0 <= index < len(self.opciones):
            self.opciones.pop(index)

    def calcular_pnl_total(self, precios_subyacente):
        pnl_total = np.zeros_like(precios_subyacente)

        for opcion in self.opciones:
            pnl_total += opcion.calcular_pnl(precios_subyacente)

        return pnl_total

    def calcular_pnl_total_bs(self, precios_subyacente, T, r, sigma):
        pnl_total_bs = np.zeros_like(precios_subyacente)

        for opcion in self.opciones:
            pnl_total_bs += opcion.calcular_pnl_bs(precios_subyacente, T, r, sigma)

        return pnl_total_bs

    def guardar_portafolio(self, filename):
        opciones_data = {
            'Tipo': [opcion.tipo for opcion in self.opciones],
            'Operacion': [opcion.operacion for opcion in self.opciones],
            'Strike': [opcion.strike for opcion in self.opciones],
            'Prima': [opcion.prima for opcion in self.opciones],
            'Cantidad': [opcion.cantidad for opcion in self.opciones]
        }
        df = pd.DataFrame(opciones_data)
        df.to_csv(filename, index=False)

    def cargar_portafolio(self, filename):
        df = pd.read_csv(filename)
        self.opciones = []
        for _, row in df.iterrows():
            self.agregar_opcion(Opcion(row['Tipo'], row['Operacion'], row['Strike'], row['Prima'], row['Cantidad']))

# Interfaz de usuario de Streamlit
st.title('Monitoreo de Opciones y Portafolio')

# Inicializar el portafolio en el estado de sesión si no existe
if 'portafolio' not in st.session_state:
    st.session_state.portafolio = Portafolio()

# Barra lateral para obtener parámetros de entrada del usuario
st.sidebar.header('Parámetros del Portafolio')
subyacente = st.sidebar.text_input('Símbolo del Subyacente', value='GGAL.BA')
tasa_libre_riesgo = st.sidebar.number_input('Tasa libre de riesgo (%)', min_value=0.0, max_value=100.0, value=0.38) / 100
volatilidad = st.sidebar.number_input('Volatilidad histórica anualizada (σ)', min_value=0.0, max_value=1.0, value=0.55)
fecha_ejercicio = st.sidebar.date_input('Fecha de ejercicio', pd.Timestamp.now().date())

# Sección para cargar opciones manualmente
st.sidebar.header('Agregar Opción')
tipo_opcion = st.sidebar.selectbox('Tipo de Opción', ['Call', 'Put'])
operacion_opcion = st.sidebar.selectbox('Operación', ['Comprada', 'Vendida'])
strike = st.sidebar.number_input('Strike Price', min_value=0.0, value=100.0)
prima = st.sidebar.number_input('Prima', min_value=0.0, value=5.0)
cantidad = st.sidebar.number_input('Cantidad Lotes', min_value=1, value=1)

# Botón para agregar opción al portafolio
if st.sidebar.button('Agregar Opción'):
    st.session_state.portafolio.agregar_opcion(Opcion(tipo_opcion, operacion_opcion, strike, prima, cantidad))
    st.sidebar.success(f'Opción {operacion_opcion} {tipo_opcion} agregada al portafolio.')

# Mostrar las opciones cargadas en una tabla
if st.session_state.portafolio.opciones:
    st.subheader('Opciones en el Portafolio:')
    opciones_data = {
        'Tipo': [opcion.tipo for opcion in st.session_state.portafolio.opciones],
        'Operación': [opcion.operacion for opcion in st.session_state.portafolio.opciones],
        'Strike': [opcion.strike for opcion in st.session_state.portafolio.opciones],
        'Prima': [opcion.prima for opcion in st.session_state.portafolio.opciones],
        'Cantidad': [opcion.cantidad for opcion in st.session_state.portafolio.opciones]
    }
    df_opciones = pd.DataFrame(opciones_data)
    st.dataframe(df_opciones)

    # Opción para eliminar una opción
    eliminar_opcion = st.selectbox('Selecciona la opción a eliminar', [f"{i + 1}: {opcion.operacion} {opcion.tipo} - Strike: {opcion.strike}" for i, opcion in enumerate(st.session_state.portafolio.opciones)])
    
    # Botón para eliminar la opción seleccionada
    if st.button('Eliminar Opción'):
        index_to_delete = int(eliminar_opcion.split(':')[0]) - 1  # Extraer el índice
        st.session_state.portafolio.eliminar_opcion(index_to_delete)
        st.success(f'Opción {eliminar_opcion} eliminada del portafolio.')

# Barra lateral para guardar y cargar el portafolio
st.sidebar.header('Guardar/Cargar Portafolio')
if st.sidebar.button('Guardar Portafolio'):
    st.session_state.portafolio.guardar_portafolio('portafolio_opciones.csv')
    st.sidebar.success('Portafolio guardado como "portafolio_opciones.csv".')

# Cargar el portafolio
uploaded_file = st.sidebar.file_uploader("Cargar portafolio", type=["csv"])
if uploaded_file is not None:
    st.session_state.portafolio.cargar_portafolio(uploaded_file)
    st.success('Portafolio cargado con éxito.')

    # Mostrar las opciones cargadas después de cargar el archivo
    opciones_data = {
        'Tipo': [opcion.tipo for opcion in st.session_state.portafolio.opciones],
        'Operación': [opcion.operacion for opcion in st.session_state.portafolio.opciones],
        'Strike': [opcion.strike for opcion in st.session_state.portafolio.opciones],
        'Prima': [opcion.prima for opcion in st.session_state.portafolio.opciones],
        'Cantidad': [opcion.cantidad for opcion in st.session_state.portafolio.opciones]
    }
    df_opciones = pd.DataFrame(opciones_data)
    st.dataframe(df_opciones)

# Obtener el precio actual del subyacente
activo = yf.Ticker(subyacente)
precio_actual_subyacente = activo.history(period="1d")['Close'].iloc[-1]
st.write(f'Precio actual de {subyacente}: ${precio_actual_subyacente:.2f}')

# Rango de precios del subyacente para el gráfico
precios_subyacente = np.linspace(0.8 * precio_actual_subyacente, 1.2 * precio_actual_subyacente, 100)

# Calcular el P&L total del portafolio
pnl_teorico_total = st.session_state.portafolio.calcular_pnl_total(precios_subyacente)

# Calcular el P&L total usando Black-Scholes
T = (pd.to_datetime(fecha_ejercicio) - pd.Timestamp.now()).days / 365  # Convertir a años
pnl_bs_total = st.session_state.portafolio.calcular_pnl_total_bs(precios_subyacente, T, tasa_libre_riesgo, volatilidad)

# Graficar ambos P&L
plt.figure(figsize=(10, 6))
plt.plot(precios_subyacente, pnl_teorico_total, label='P&L Teórico del Portafolio', color='blue')
plt.plot(precios_subyacente, pnl_bs_total, label='P&L del Portafolio (Black-Scholes)', color='green', linestyle='--')
plt.axhline(0, color='red', linestyle='--', label='Break-even')
plt.axvline(precio_actual_subyacente, color='purple', linestyle='--', label='Precio Actual del Subyacente')
plt.title('P&L de un Portafolio: Call y Put')
plt.xlabel('Precio del Subyacente')
plt.ylabel('P&L')
plt.legend()
plt.grid(True)
st.pyplot(plt)
