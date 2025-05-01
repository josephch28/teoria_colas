import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulación de Colas", layout="centered")
st.title("📊 Simulación de colas en un banco")

# Parámetros de entrada interactivos
tiempo_total = st.slider("⏱ Tiempo total de la jornada (min)", 60, 720, 480, step=30)
tasa_llegada = st.slider("👥 Tasa de llegada por minuto", 0.05, 1.0, 0.5, step=0.05)
num_cajeros = st.slider("💼 Número de cajeros", 1, 10, 4)
tiempo_min_servicio = st.slider("🔧 Tiempo mínimo de servicio (min)", 1, 10, 2)
tiempo_max_servicio = st.slider("🔧 Tiempo máximo de servicio (min)", 5, 20, 10)

# Clases
class Cliente:
    def __init__(self, llegada):
        self.llegada = llegada
        self.tiempo_servicio = np.random.randint(tiempo_min_servicio, tiempo_max_servicio + 1)
        self.inicio_servicio = None
        self.fin_servicio = None

class Cajero:
    def __init__(self):
        self.ocupado = False
        self.cliente_actual = None
        self.tiempo_restante = 0

    def atender(self, cliente, tiempo_actual):
        self.cliente_actual = cliente
        self.tiempo_restante = cliente.tiempo_servicio
        self.ocupado = True
        cliente.inicio_servicio = tiempo_actual

    def avanzar_tiempo(self, tiempo_actual):
        if self.ocupado:
            self.tiempo_restante -= 1
            if self.tiempo_restante <= 0:
                self.cliente_actual.fin_servicio = tiempo_actual
                cliente_atendidos.append(self.cliente_actual)
                self.cliente_actual = None
                self.ocupado = False

# Simulación
fila = []
cliente_atendidos = []
cajeros = [Cajero() for _ in range(num_cajeros)]
longitud_cola_por_minuto = []

for minuto in range(tiempo_total):
    if np.random.rand() < tasa_llegada:
        fila.append(Cliente(minuto))

    for cajero in cajeros:
        if not cajero.ocupado and fila:
            cliente = fila.pop(0)
            cajero.atender(cliente, minuto)

    for cajero in cajeros:
        cajero.avanzar_tiempo(minuto)

    longitud_cola_por_minuto.append(len(fila))

# Estadísticas
total_atendidos = len(cliente_atendidos)
tiempos_espera = [c.inicio_servicio - c.llegada for c in cliente_atendidos]
tiempo_prom_espera = np.mean(tiempos_espera) if tiempos_espera else 0
tiempo_max_espera = np.max(tiempos_espera) if tiempos_espera else 0
tiempo_prom_servicio = np.mean([c.tiempo_servicio for c in cliente_atendidos]) if cliente_atendidos else 0

# Resultados
st.subheader("📋 Resultados")
st.write(f"🔹 Clientes atendidos: **{total_atendidos}**")
st.write(f"🔹 Tiempo promedio de espera: **{tiempo_prom_espera:.2f} min**")
st.write(f"🔹 Tiempo máximo de espera: **{tiempo_max_espera} min**")
st.write(f"🔹 Tiempo promedio de servicio: **{tiempo_prom_servicio:.2f} min**")
st.write(f"🔹 Cajeros: **{num_cajeros}**")

# Gráfico evolución cola
st.subheader("📈 Evolución de la cola durante la jornada")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(longitud_cola_por_minuto, label="Longitud de la cola")
ax.set_xlabel("Minuto del día")
ax.set_ylabel("Clientes en cola")
ax.set_title("Clientes en cola por minuto")
ax.grid(True)
st.pyplot(fig)

# Histograma tiempos de espera
st.subheader("⏳ Distribución del tiempo de espera")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.hist(tiempos_espera, bins=range(0, max(tiempos_espera)+2), color="skyblue", edgecolor="black")
ax2.set_xlabel("Tiempo de espera (min)")
ax2.set_ylabel("Clientes")
ax2.grid(True)
st.pyplot(fig2)
