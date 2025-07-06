import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

st.set_page_config(page_title="Simulación de Colas", layout="centered")
st.title("📊 Simulación de colas en un banco")

# Sidebar para seleccionar análisis
st.sidebar.title("🔧 Análisis de Regresión")
analisis_tipo = st.sidebar.selectbox(
    "Selecciona el tipo de análisis:",
    ["Simulación Básica", "Regresión Lineal - Tiempo de Espera", 
     "Regresión Múltiple - Factores del Sistema", "Análisis Predictivo"]
)

# Parámetros de entrada interactivos
tiempo_total = st.slider("⏱ Tiempo total de la jornada (min)", 60, 720, 480, step=30)
tasa_llegada = st.slider("👥 Tasa de llegada por minuto", 0.05, 1.0, 0.5, step=0.05)
num_cajeros = st.slider("💼 Número de cajeros", 1, 10, 4)
tiempo_min_servicio = st.slider("🔧 Tiempo mínimo de servicio (min)", 1, 10, 2)
tiempo_max_servicio = st.slider("🔧 Tiempo máximo de servicio (min)", 5, 20, 10)

# Inicialización de la sesión de resultados si no existe
if 'resultados_comparacion' not in st.session_state:
    st.session_state.resultados_comparacion = {}

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

# Función para generar datos de entrenamiento para regresión
def generar_datos_entrenamiento(n_simulaciones=100):
    """Genera datos de múltiples simulaciones para entrenar modelos de regresión"""
    datos = []
    
    for _ in range(n_simulaciones):
        # Parámetros aleatorios
        tiempo_total = np.random.randint(120, 721)
        tasa_llegada = np.random.uniform(0.1, 1.0)
        num_cajeros = np.random.randint(1, 11)
        tiempo_min_servicio = np.random.randint(1, 8)
        tiempo_max_servicio = np.random.randint(tiempo_min_servicio + 2, 21)
        
        # Simular
        resultado = simular_cola(tiempo_total, tasa_llegada, num_cajeros, 
                                tiempo_min_servicio, tiempo_max_servicio)
        
        datos.append({
            'tiempo_total': tiempo_total,
            'tasa_llegada': tasa_llegada,
            'num_cajeros': num_cajeros,
            'tiempo_min_servicio': tiempo_min_servicio,
            'tiempo_max_servicio': tiempo_max_servicio,
            'tiempo_prom_servicio': (tiempo_min_servicio + tiempo_max_servicio) / 2,
            'tiempo_prom_espera': resultado['tiempo_prom_espera'],
            'total_atendidos': resultado['total_atendidos'],
            'utilizacion_cajeros': resultado['total_ocupado_cajeros'] / (tiempo_total * num_cajeros),
            'clientes_en_cola_final': resultado['clientes_en_cola_final']
        })
    
    return pd.DataFrame(datos)

# Función para regresión lineal simple
def regresion_tiempo_espera():
    st.subheader("📈 Regresión Lineal: Predicción de Tiempo de Espera")
    
    # Generar datos
    with st.spinner("Generando datos de entrenamiento..."):
        df = generar_datos_entrenamiento(200)
    
    # Seleccionar variable independiente
    variable_x = st.selectbox(
        "Selecciona la variable independiente:",
        ['tasa_llegada', 'num_cajeros', 'tiempo_prom_servicio', 'utilizacion_cajeros']
    )
    
    # Preparar datos
    X = df[variable_x].values.reshape(-1, 1)
    y = df['tiempo_prom_espera'].values
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MSE", f"{mse:.4f}")
    
    # Gráfico de regresión
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test, y_test, alpha=0.6, label='Datos reales')
    ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicción')
    ax.set_xlabel(variable_x.replace('_', ' ').title())
    ax.set_ylabel('Tiempo Promedio de Espera (min)')
    ax.set_title(f'Regresión Lineal: {variable_x.replace("_", " ").title()} vs Tiempo de Espera')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Ecuación de la línea
    st.write(f"**Ecuación de la línea:** Tiempo de Espera = {modelo.intercept_:.2f} + {modelo.coef_[0]:.2f} × {variable_x.replace('_', ' ')}")
    
    # Interpretación
    st.write("**Interpretación:**")
    if modelo.coef_[0] > 0:
        st.write(f"📈 Un aumento en {variable_x.replace('_', ' ')} se asocia con un aumento en el tiempo de espera")
    else:
        st.write(f"📉 Un aumento en {variable_x.replace('_', ' ')} se asocia con una disminución en el tiempo de espera")

# Función para regresión múltiple
def regresion_multiple():
    st.subheader("📊 Regresión Múltiple: Análisis de Factores del Sistema")
    
    # Generar datos
    with st.spinner("Generando datos de entrenamiento..."):
        df = generar_datos_entrenamiento(300)
    
    # Seleccionar variables independientes
    variables_independientes = st.multiselect(
        "Selecciona las variables independientes:",
        ['tasa_llegada', 'num_cajeros', 'tiempo_prom_servicio', 'tiempo_total', 'utilizacion_cajeros'],
        default=['tasa_llegada', 'num_cajeros', 'tiempo_prom_servicio']
    )
    
    if len(variables_independientes) < 2:
        st.warning("Selecciona al menos 2 variables independientes")
        return
    
    # Variable dependiente
    variable_dependiente = st.selectbox(
        "Selecciona la variable dependiente:",
        ['tiempo_prom_espera', 'total_atendidos', 'clientes_en_cola_final']
    )
    
    # Preparar datos
    X = df[variables_independientes].values
    y = df[variable_dependiente].values
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Seleccionar tipo de regresión
    tipo_regresion = st.selectbox("Tipo de regresión:", ["Linear", "Ridge", "Lasso"])
    
    if tipo_regresion == "Linear":
        modelo = LinearRegression()
    elif tipo_regresion == "Ridge":
        alpha = st.slider("Alpha (Ridge):", 0.1, 10.0, 1.0)
        modelo = Ridge(alpha=alpha)
    else:  # Lasso
        alpha = st.slider("Alpha (Lasso):", 0.1, 10.0, 1.0)
        modelo = Lasso(alpha=alpha)
    
    # Entrenar modelo
    modelo.fit(X_train, y_train)
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MSE", f"{mse:.4f}")
    
    # Coeficientes
    st.write("**Coeficientes del modelo:**")
    coeficientes_df = pd.DataFrame({
        'Variable': variables_independientes,
        'Coeficiente': modelo.coef_
    })
    st.dataframe(coeficientes_df)
    
    # Gráfico de coeficientes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(variables_independientes, modelo.coef_)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Coeficiente')
    ax.set_title('Importancia de Variables en el Modelo')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Gráfico de predicciones vs reales
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Predicciones')
    ax.set_title('Predicciones vs Valores Reales')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Función para análisis predictivo
def analisis_predictivo():
    st.subheader("🔮 Análisis Predictivo: Optimización del Sistema")
    
    # Generar datos de entrenamiento
    with st.spinner("Generando datos de entrenamiento..."):
        df = generar_datos_entrenamiento(500)
    
    # Preparar modelo para predecir tiempo de espera
    X = df[['tasa_llegada', 'num_cajeros', 'tiempo_prom_servicio']].values
    y = df['tiempo_prom_espera'].values
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)
    
    st.write("**Predicción de tiempo de espera para diferentes configuraciones:**")
    
    # Interfaz para predicciones
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tasa_pred = st.number_input("Tasa de llegada:", 0.1, 2.0, 0.5, 0.1)
    with col2:
        cajeros_pred = st.number_input("Número de cajeros:", 1, 15, 4, 1)
    with col3:
        tiempo_servicio_pred = st.number_input("Tiempo promedio de servicio:", 1.0, 20.0, 6.0, 0.5)
    
    # Realizar predicción
    if st.button("🔮 Predecir Tiempo de Espera"):
        # Preparar datos para predicción
        X_pred = np.array([[tasa_pred, cajeros_pred, tiempo_servicio_pred]])
        X_pred_scaled = scaler.transform(X_pred)
        
        # Predicción
        tiempo_espera_pred = modelo.predict(X_pred_scaled)[0]
        
        st.success(f"**Tiempo de espera predicho: {tiempo_espera_pred:.2f} minutos**")
        
        # Análisis de sensibilidad
        st.write("**Análisis de sensibilidad:**")
        
        # Variar número de cajeros
        cajeros_range = range(1, 16)
        tiempos_cajeros = []
        
        for c in cajeros_range:
            X_temp = np.array([[tasa_pred, c, tiempo_servicio_pred]])
            X_temp_scaled = scaler.transform(X_temp)
            tiempo_temp = modelo.predict(X_temp_scaled)[0]
            tiempos_cajeros.append(tiempo_temp)
        
        # Gráfico de sensibilidad
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cajeros_range, tiempos_cajeros, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Número de Cajeros')
        ax.set_ylabel('Tiempo de Espera Predicho (min)')
        ax.set_title('Sensibilidad: Tiempo de Espera vs Número de Cajeros')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.7, label='Límite aceptable (5 min)')
        ax.legend()
        st.pyplot(fig)
        
        # Recomendación
        cajeros_optimos = [c for c, t in zip(cajeros_range, tiempos_cajeros) if t <= 5]
        if cajeros_optimos:
            st.info(f"💡 **Recomendación:** Para mantener el tiempo de espera bajo 5 minutos, se necesitan al menos {min(cajeros_optimos)} cajeros")
        else:
            st.warning("⚠️ **Advertencia:** Incluso con 15 cajeros, el tiempo de espera superaría los 5 minutos")

# Ejecutar análisis según selección
if analisis_tipo == "Simulación Básica":
    # Comparación de diferentes escenarios
    st.subheader("🔍 Comparación de Escenarios")
    num_cajeros_opciones = st.multiselect("📊 Selecciona diferentes números de cajeros", range(1, 11), default=[num_cajeros])

    # Solo borrar y recalcular resultados si se cambia la selección de cajeros
    if 'num_cajeros_opciones' not in st.session_state or st.session_state.num_cajeros_opciones != num_cajeros_opciones:
        st.session_state.resultados_comparacion = {}
        for cajeros_opcion in num_cajeros_opciones:
            st.session_state.resultados_comparacion[cajeros_opcion] = simular_cola(
                tiempo_total, tasa_llegada, cajeros_opcion, tiempo_min_servicio, tiempo_max_servicio)
        
        st.session_state.num_cajeros_opciones = num_cajeros_opciones

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

elif analisis_tipo == "Regresión Lineal - Tiempo de Espera":
    regresion_tiempo_espera()

elif analisis_tipo == "Regresión Múltiple - Factores del Sistema":
    regresion_multiple()

elif analisis_tipo == "Análisis Predictivo":
    analisis_predictivo()






