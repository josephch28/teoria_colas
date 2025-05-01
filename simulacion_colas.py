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

# Función para realizar simulación de colas
def simular_cola(tiempo_total, tasa_llegada, num_cajeros, tiempo_min_servicio, tiempo_max_servicio):
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

    return tiempo_prom_espera, tiempo_max_espera, tiempo_prom_servicio, longitud_cola_por_minuto, tiempos_espera

# Comparación de diferentes escenarios
st.subheader("🔍 Comparación de Escenarios")
num_cajeros_opciones = st.multiselect("📊 Selecciona diferentes números de cajeros", range(1, 11), default=[num_cajeros])
resultados_comparacion = {}

for cajeros_opcion in num_cajeros_opciones:
    tiempo_prom_espera, tiempo_max_espera, tiempo_prom_servicio, _, _ = simular_cola(
        tiempo_total, tasa_llegada, cajeros_opcion, tiempo_min_servicio, tiempo_max_servicio)
    resultados_comparacion[cajeros_opcion] = tiempo_prom_espera

# Mostrar resultados de comparación
st.subheader("📋 Resultados de la comparación")
for cajeros_opcion, tiempo in resultados_comparacion.items():
    st.write(f"Con {cajeros_opcion} cajeros, el tiempo promedio de espera es: **{tiempo:.2f} minutos**")

# Gráfico de comparación de tiempos promedio de espera por número de cajeros
st.subheader("📈 Comparación de tiempos de espera promedio")
fig_comparacion, ax_comparacion = plt.subplots(figsize=(10, 5))
ax_comparacion.bar(resultados_comparacion.keys(), resultados_comparacion.values(), color='skyblue')
ax_comparacion.set_xlabel("Número de Cajeros")
ax_comparacion.set_ylabel("Tiempo Promedio de Espera (min)")
ax_comparacion.set_title("Comparación de tiempos de espera promedio por número de cajeros")
st.pyplot(fig_comparacion)

# Gráfico de evolución de la cola
st.subheader("📈 Evolución de la cola durante la jornada")
fig, ax = plt.subplots(figsize=(10, 4))
_, _, _, longitud_cola_por_minuto, _ = simular_cola(
    tiempo_total, tasa_llegada, num_cajeros, tiempo_min_servicio, tiempo_max_servicio)
ax.plot(longitud_cola_por_minuto, label="Longitud de la cola")
ax.set_xlabel("Minuto del día")
ax.set_ylabel("Clientes en cola")
ax.set_title("Evolución de la cola durante la jornada")
ax.grid(True)
st.pyplot(fig)

# Histograma de tiempos de espera
st.subheader("⏳ Distribución del tiempo de espera")
_, _, _, _, tiempos_espera = simular_cola(
    tiempo_total, tasa_llegada, num_cajeros, tiempo_min_servicio, tiempo_max_servicio)
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.hist(tiempos_espera, bins=range(0, max(tiempos_espera)+2), color="skyblue", edgecolor="black")
ax2.set_xlabel("Tiempo de espera (min)")
ax2.set_ylabel("Número de clientes")
ax2.set_title("Distribución de los tiempos de espera")
ax2.grid(True)
st.pyplot(fig2)


