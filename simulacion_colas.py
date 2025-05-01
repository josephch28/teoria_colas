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
            self.total_ocupado = 0  # Total de tiempo que estuvo ocupado

        def atender(self, cliente, tiempo_actual):
            self.cliente_actual = cliente
            self.tiempo_restante = cliente.tiempo_servicio
            self.ocupado = True
            cliente.inicio_servicio = tiempo_actual

        def avanzar_tiempo(self, tiempo_actual):
            if self.ocupado:
                self.tiempo_restante -= 1
                self.total_ocupado += 1
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
    total_ocupado_cajeros = sum([cajero.total_ocupado for cajero in cajeros])  # Total de tiempo ocupado por los cajeros
    clientes_en_cola_final = len(fila)  # Clientes restantes en la cola al final de la jornada
    tiempo_max_servicio = np.max([c.tiempo_servicio for c in cliente_atendidos]) if cliente_atendidos else 0

    # Resultados
    return {
        'total_atendidos': total_atendidos,
        'tiempo_prom_espera': tiempo_prom_espera,
        'tiempo_max_espera': tiempo_max_espera,
        'tiempo_prom_servicio': tiempo_prom_servicio,
        'total_ocupado_cajeros': total_ocupado_cajeros,
        'clientes_en_cola_final': clientes_en_cola_final,
        'tiempo_max_servicio': tiempo_max_servicio,
        'longitud_cola_por_minuto': longitud_cola_por_minuto,
        'tiempos_espera': tiempos_espera
    }

# Inicializar sesión de resultados
if 'resultados_comparacion' not in st.session_state:
    st.session_state.resultados_comparacion = {}

# Función para borrar los resultados antiguos y recalcular
def reiniciar_simulacion():
    st.session_state.resultados_comparacion = {}

# Reiniciar simulación si cambian los parámetros
if st.button("🔄 Reiniciar Simulación"):
    reiniciar_simulacion()

# Comparación de diferentes escenarios
st.subheader("🔍 Comparación de Escenarios")
num_cajeros_opciones = st.multiselect("📊 Selecciona diferentes números de cajeros", range(1, 11), default=[num_cajeros])

# Calcular resultados solo si se seleccionan nuevos números de cajeros
for cajeros_opcion in num_cajeros_opciones:
    if cajeros_opcion not in st.session_state.resultados_comparacion:
        st.session_state.resultados_comparacion[cajeros_opcion] = simular_cola(
            tiempo_total, tasa_llegada, cajeros_opcion, tiempo_min_servicio, tiempo_max_servicio)

# Mostrar resultados de comparación
st.subheader("📋 Resultados de la comparación")
for cajeros_opcion, resultados in st.session_state.resultados_comparacion.items():
    st.write(f"### Con {cajeros_opcion} cajeros:")
    st.write(f"🔹 Clientes atendidos: **{resultados['total_atendidos']}**")
    st.write(f"🔹 Tiempo promedio de espera: **{resultados['tiempo_prom_espera']:.2f} min**")
    st.write(f"🔹 Tiempo máximo de espera: **{resultados['tiempo_max_espera']} min**")
    st.write(f"🔹 Tiempo promedio de servicio: **{resultados['tiempo_prom_servicio']:.2f} min**")
    st.write(f"🔹 Tiempo total de ocupación de los cajeros: **{resultados['total_ocupado_cajeros']} min**")
    st.write(f"🔹 Clientes en cola al final de la jornada: **{resultados['clientes_en_cola_final']}**")
    st.write(f"🔹 Tiempo máximo de servicio: **{resultados['tiempo_max_servicio']} min**")
    st.write("-----")

# Gráfico de comparación de tiempos promedio de espera por número de cajeros
st.subheader("📈 Comparación de tiempos de espera promedio")
fig_comparacion, ax_comparacion = plt.subplots(figsize=(10, 5))
ax_comparacion.bar(st.session_state.resultados_comparacion.keys(), 
                   [resultados['tiempo_prom_espera'] for resultados in st.session_state.resultados_comparacion.values()], 
                   color='skyblue')
ax_comparacion.set_xlabel("Número de Cajeros")
ax_comparacion.set_ylabel("Tiempo Promedio de Espera (min)")
ax_comparacion.set_title("Comparación de tiempos de espera promedio por número de cajeros")
st.pyplot(fig_comparacion)

# Gráfico de comparación de la evolución de la cola
st.subheader("📈 Comparación de la evolución de la cola")
fig_evolucion, ax_evolucion = plt.subplots(figsize=(10, 5))
for cajeros_opcion, resultados in st.session_state.resultados_comparacion.items():
    ax_evolucion.plot(resultados['longitud_cola_por_minuto'], label=f"{cajeros_opcion} cajeros")

ax_evolucion.set_xlabel("Minuto del día")
ax_evolucion.set_ylabel("Clientes en cola")
ax_evolucion.set_title("Evolución de la cola con diferentes números de cajeros")
ax_evolucion.legend(title="Número de cajeros")
ax_evolucion.grid(True)
st.pyplot(fig_evolucion)

# Histograma de comparación de tiempos de espera
st.subheader("⏳ Comparación de la distribución del tiempo de espera")
fig_histograma, ax_histograma = plt.subplots(figsize=(8, 5))
for cajeros_opcion, resultados in st.session_state.resultados_comparacion.items():
    ax_histograma.hist(resultados['tiempos_espera'], bins=range(0, max(resultados['tiempos_espera'])+2), 
                       alpha=0.5, label=f"{cajeros_opcion} cajeros", edgecolor="black")

ax_histograma.set_xlabel("Tiempo de espera (min)")
ax_histograma.set_ylabel("Número de clientes")
ax_histograma.set_title("Distribución de tiempos de espera por número de cajeros")
ax_histograma.legend(title="Número de cajeros")
ax_histograma.grid(True)
st.pyplot(fig_histograma)



