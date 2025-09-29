#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def conductividad_hidraulica(theta, theta_r, theta_s, n, K_s):
    """
    Calcula conductividad hidráulica usando van Genuchten-Mualem
    
    Parámetros:
        theta: contenido volumétrico de agua
        theta_r: contenido residual
        theta_s: contenido de saturación  
        n: parámetro de forma
        K_s: conductividad saturada (m/h)
    """
    m = 1.0 - 1.0/n
    Se = np.clip((theta - theta_r) / (theta_s - theta_r), 0, 1)
    K_rel = Se**0.5 * (1 - (1 - Se**(1/m))**m)**2
    return K_s * K_rel

def derivada_theta(theta, q_entrada, parametros, delta_z):
    """Calcula dtheta/dt para un elemento"""
    theta_r, theta_s, n, K_s = parametros
    q_salida = conductividad_hidraulica(theta, theta_r, theta_s, n, K_s)
    return (q_entrada - q_salida) / delta_z

def derivada_concentracion(C, theta, q_theta, qi, Ci, dtheta_dt, delta_z):
    """Calcula dC/dt para transporte de solutos"""
    if theta <= 0:
        return 0.0
    
    return -(1/theta) * (
        (q_theta * C) / delta_z - 
        (qi * Ci) / delta_z - 
        C * dtheta_dt
    )

def paso_runge_kutta_concentracion(C, theta, q_theta, qi, Ci, dtheta_dt, dt, delta_z):
    """Un paso de integración Runge-Kutta para concentración"""
    k1 = derivada_concentracion(C, theta, q_theta, qi, Ci, dtheta_dt, delta_z)
    k2 = derivada_concentracion(C + 0.5*dt*k1, theta, q_theta, qi, Ci, dtheta_dt, delta_z)
    k3 = derivada_concentracion(C + 0.5*dt*k2, theta, q_theta, qi, Ci, dtheta_dt, delta_z)
    k4 = derivada_concentracion(C + dt*k3, theta, q_theta, qi, Ci, dtheta_dt, delta_z)
    return C + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def paso_runge_kutta(theta, q_entrada, dt, parametros, delta_z):
    """Un paso de integración Runge-Kutta de 4to orden"""
    k1 = derivada_theta(theta, q_entrada, parametros, delta_z)
    k2 = derivada_theta(theta + 0.5*dt*k1, q_entrada, parametros, delta_z)
    k3 = derivada_theta(theta + 0.5*dt*k2, q_entrada, parametros, delta_z)
    k4 = derivada_theta(theta + dt*k3, q_entrada, parametros, delta_z)
    return theta + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# =============================================================================
# PARÁMETROS DE SIMULACIÓN
# =============================================================================

# Parámetros de concentración
concentracion_inicial = 3.2    # Concentración inicial en la columna
concentracion_entrada = 0.3   # Concentración del agua de riego

# Parámetros del suelo
theta_r = 0.05      # Contenido residual
theta_s = 0.30      # Contenido de saturación
n = 2.5             # Parámetro de forma van Genuchten
K_s = 5/1000 * 40   # Conductividad saturada (m/h)

# Parámetros de la columna
altura_total = 10.0     # m
delta_z = 0.1          # m (espaciado entre elementos)
n_elementos = int(altura_total / delta_z)

# Parámetros temporales
dt: float = 0.1  # Paso de tiempo en horas
dias_simulacion: int = 5
tiempo_total: float = 24 * dias_simulacion  # Total de horas a simular (10 días)
n_pasos: int = int(tiempo_total / dt)

# Condiciones iniciales
theta_inicial: float = 0.12
tasa_riego: float = 10/1000  # m/h (flujo de entrada en superficie)

# =============================================================================
# SIMULACIÓN
# =============================================================================

print("Iniciando simulación de infiltración...")
print(f"Elementos: {n_elementos}, Pasos temporales: {n_pasos}")

# Inicializar
parametros = (theta_r, theta_s, n, K_s)
theta_columna = np.full(n_elementos, theta_inicial)
concentracion_columna = np.full(n_elementos, concentracion_inicial)  # Nueva línea

# Crear concentración inicial con variación aleatoria alrededor del valor base
np.random.seed(42)  # Para reproducibilidad
variacion = 0.5  # 10% de variación
concentracion_columna = concentracion_inicial * (1 + variacion * (np.random.random(n_elementos) - 0.5))

profundidades = np.arange(delta_z, altura_total + delta_z, delta_z)


# Almacenar resultados
resultados_theta = []
resultados_concentracion = []  # Nueva línea
theta_columna_actual = theta_columna.copy()
concentracion_columna_actual = concentracion_columna.copy()  # Nueva línea
# Almacenar concentraciones de salida y sus tiempos correspondientes
concentraciones_salida = []
tiempos_salida = []


def obtener_tasa_riego(
    tiempo_actual: float,
    tasa_riego_base: float,
    duracion_riego: float = 10.0,
    duracion_reposo: float = 10.0,
    riego_continuo: bool = False
) -> float:
    """
    Controla el patrón de riego: continuo o alternando entre riego y reposo.
    
    Args:
        tiempo_actual (float): Tiempo actual en horas.
        tasa_riego_base (float): Tasa de riego normal (m/h).
        duracion_riego (float): Horas de riego en cada ciclo.
        duracion_reposo (float): Horas de reposo en cada ciclo.
        riego_continuo (bool): Si True, aplica riego continuo sin pausas.

    Returns:
        float: Tasa de riego para el tiempo actual.
    """
    if riego_continuo:
        return tasa_riego_base
    
    ciclo_total = duracion_riego + duracion_reposo
    ciclo = tiempo_actual % ciclo_total
    if ciclo < duracion_riego:
        return tasa_riego_base
    return 0.0

duracion_riego = 15.0
duracion_reposo = 5.0
riego_continuo = True  # Cambiar a True para riego continuo

# Simulación principal
for paso in range(n_pasos):
    tiempo_actual = paso * dt
    q0 = obtener_tasa_riego(
        tiempo_actual,
        tasa_riego,
        duracion_riego=duracion_riego,
        duracion_reposo=duracion_reposo,
        riego_continuo=riego_continuo
    )
    
    # Mostrar evolución conjunta de humedad y concentración en el primer elemento
    if paso % 50 == 0:
        print(f"Paso {paso}, Tiempo {tiempo_actual:.2f}h - Humedad inicial: {theta_columna_actual[0]:.6f}, "
              f"Concentración inicial: {concentracion_columna_actual[0]:.6f}")


    for i in range(n_elementos):
        # Calcular dtheta/dt antes de actualizar
        dtheta_dt = derivada_theta(theta_columna_actual[i], q0, parametros, delta_z)
        
        # Calcular q(theta) actual
        q_actual = conductividad_hidraulica(theta_columna_actual[i], theta_r, theta_s, n, K_s)
        
        # Actualizar concentración ANTES de actualizar theta

        concentracion_columna_actual[i] = paso_runge_kutta_concentracion(
            concentracion_columna_actual[i], 
            theta_columna_actual[i], 
            q_actual, 
            q0, 
            concentracion_entrada, 
            dtheta_dt, 
            dt, 
            delta_z
        )
        
        # Actualizar theta
        theta_columna_actual[i] = paso_runge_kutta(theta_columna_actual[i], q0, dt, parametros, delta_z)
        
        # El flujo de salida para el siguiente elemento
        q0 = conductividad_hidraulica(theta_columna_actual[i], theta_r, theta_s, n, K_s)


    # Guardar cada 10 pasos (~cada 1 hora)
    if paso % 10 == 0:
        resultados_theta.append(theta_columna_actual.copy())
        resultados_concentracion.append(concentracion_columna_actual.copy())
        # Guardar la concentración de salida (último elemento) y su tiempo correspondiente
        concentraciones_salida.append(concentracion_columna_actual[-1])
        tiempos_salida.append(tiempo_actual)

        # if paso % 100 == 0:
        #     print(f"Paso {paso}, Tiempo {tiempo_actual:.2f}h - Concentración salida antes: {concentracion_columna_actual[-1]:.6f}")
        #     print(f"Paso {paso}, Tiempo {tiempo_actual:.2f}h - Concentración salida después: {concentracion_columna_actual[-1]:.6f}")

print("Simulación completada!")

# Graficar la curva de concentración en el último elemento

plt.figure(figsize=(10, 6), dpi=140)
plt.plot(tiempos_salida, concentraciones_salida, 'b-', linewidth=2, marker='o', markersize=3)
plt.grid(True, alpha=0.3)
plt.xlabel('Tiempo (h)')
plt.ylabel('Concentración')
plt.title('Evolución de la concentración en el último elemento')
plt.tight_layout()
plt.show()


#%%
# Configurar la figura para la animación
fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.31)
ax.set_ylim(altura_total, 0)
ax.set_xlabel('Contenido de agua (θ)')
ax.set_ylabel('Profundidad (m)')
if riego_continuo:
    ax.set_title(
        f'Evolución del perfil de humedad\n'
        f'Riego continuo, Tasa = {tasa_riego} m/h'
    )
else:
    ax.set_title(
        f'Evolución del perfil de humedad\n'
        f'Riego: {duracion_riego}h ON / {duracion_reposo}h OFF, Tasa = {tasa_riego} m/h'
    )

# Crear la línea inicial (vacía)
line, = ax.plot([], [], marker='d', markersize=3)
tiempo_texto = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Añadir líneas de referencia para el contenido de saturación y residual
ax.axvline(x=theta_s, color='blue', linestyle='--', alpha=0.5, label=f'θ saturación = {theta_s}')
ax.axvline(x=theta_r, color='red', linestyle='--', alpha=0.5, label=f'θ residual = {theta_r}')
ax.legend(loc='upper right')

def init() -> tuple:
    """Función de inicialización para la animación."""
    line.set_data([], [])
    tiempo_texto.set_text('')
    return line, tiempo_texto

def animate(i: int) -> tuple:
    """
    Actualiza la animación para el frame i.

    Args:
        i (int): Índice del frame.

    Returns:
        tuple: Elementos actualizados para blitting.
    """
    theta_perfil = resultados_theta[i]
    line.set_data(theta_perfil, profundidades)
    # Cada frame representa 1 hora (10 pasos de 0.1h)
    tiempo_actual = i * 1.0  # 1 hora por frame
    tiempo_texto.set_text(f'Tiempo: {tiempo_actual:.1f} h')
    return line, tiempo_texto

# Crear la animación con todos los frames disponibles
ani = animation.FuncAnimation(
    fig, animate, frames=len(resultados_theta),
    init_func=init, blit=True, interval=200
)

# Guardar la animación como GIF con mayor calidad
output_path = Path("resultados_theta_infiltracion.gif")
ani.save(str(output_path), writer='pillow', fps=6, dpi=140)

# Mostrar estado inicial y final para comparación
fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax1.plot(resultados_theta[0], profundidades, 'b-o', markersize=3)
ax1.set_title('Estado inicial')
ax1.set_xlabel('Contenido de agua (θ)')
ax1.set_ylabel('Profundidad (m)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(altura_total, 0)

ax2.plot(resultados_theta[-1], profundidades, 'r-o', markersize=3)
ax2.set_title('Estado final')
ax2.set_xlabel('Contenido de agua (θ)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(altura_total, 0)

plt.tight_layout()
fig_comp.savefig('comparacion_inicial_final.png', dpi=140)

print(f"Animación guardada como: {output_path.absolute()}")
print("Comparación inicial-final guardada como: comparacion_inicial_final.png")

# Crear animación para la concentración de salida
fig_conc, ax_conc = plt.subplots(figsize=(10, 6), dpi=140)
ax_conc.grid(True, alpha=0.3)
ax_conc.set_xlim(0, tiempo_total)
ax_conc.set_ylim(0, max(concentraciones_salida) * 1.1)
ax_conc.set_xlabel('Tiempo (h)')
ax_conc.set_ylabel('Concentración de salida')
if riego_continuo:
    ax_conc.set_title(
        f'Evolución de la concentración de salida\n'
        f'Riego continuo, Tasa = {tasa_riego} m/h'
    )
else:
    ax_conc.set_title(
        f'Evolución de la concentración de salida\n'
        f'Riego: {duracion_riego}h ON / {duracion_reposo}h OFF, Tasa = {tasa_riego} m/h'
    )

# Crear la línea inicial (vacía)
line_conc, = ax_conc.plot([], [], 'g-', marker='o', markersize=3)
tiempo_texto_conc = ax_conc.text(0.02, 0.95, '', transform=ax_conc.transAxes)

# Añadir línea de referencia para la concentración inicial y de entrada
ax_conc.axhline(y=concentracion_inicial, color='blue', linestyle='--', alpha=0.5, 
                label=f'Concentración inicial = {concentracion_inicial}')
ax_conc.axhline(y=concentracion_entrada, color='red', linestyle='--', alpha=0.5, 
                label=f'Concentración entrada = {concentracion_entrada}')
ax_conc.legend(loc='upper right')

def init_conc() -> tuple:
    """Función de inicialización para la animación de concentración."""
    line_conc.set_data([], [])
    tiempo_texto_conc.set_text('')
    return line_conc, tiempo_texto_conc

def animate_conc(i: int) -> tuple:
    """
    Actualiza la animación de concentración para el frame i.

    Args:
        i (int): Índice del frame.

    Returns:
        tuple: Elementos actualizados para blitting.
    """
    # Mostrar datos hasta el frame actual
    tiempo_actual = tiempos_salida[i]
    line_conc.set_data(tiempos_salida[:i+1], concentraciones_salida[:i+1])
    tiempo_texto_conc.set_text(f'Tiempo: {tiempo_actual:.1f} h')
    return line_conc, tiempo_texto_conc

# Crear la animación con todos los frames disponibles
ani_conc = animation.FuncAnimation(
    fig_conc, animate_conc, frames=len(tiempos_salida),
    init_func=init_conc, blit=True, interval=200
)

# Guardar la animación como GIF con mayor calidad
output_path_conc = Path("resultados_concentracion_salida.gif")
ani_conc.save(str(output_path_conc), writer='pillow', fps=6, dpi=140)
print(f"Animación de concentración guardada como: {output_path_conc.absolute()}")

plt.show()

print("FIN")
# %%
