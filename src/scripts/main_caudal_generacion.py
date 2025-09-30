#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# FUNCIONES PRINCIPALES OPTIMIZADAS - SISTEMA ACOPLADO RK4
# =============================================================================

def conductividad_hidraulica(theta, theta_r, theta_s, n, K_s):
    """
    Calcula conductividad hidráulica usando van Genuchten-Mualem
    """
    m = 1.0 - 1.0/n
    Se = np.clip((theta - theta_r) / (theta_s - theta_r), 0, 1)
    K_rel = Se**0.5 * (1 - (1 - Se**(1/m))**m)**2
    return K_s * K_rel

def derivada_theta(theta, q_entrada, parametros_suelo, delta_z):
    """Función derivada para humedad: dtheta/dt"""
    theta_r, theta_s, n, K_s = parametros_suelo 
    q_theta = conductividad_hidraulica(theta, theta_r, theta_s, n, K_s)
    
    dtheta_dt=(q_entrada - q_theta) / delta_z
    return dtheta_dt

def derivada_concentracion(C, theta, q_theta, qi, Ci, dtheta_dt, dL_dt, delta_z):
    """Función derivada para concentración: dC/dt"""
    Ros=1650
    if theta <= 0:
        return 0.0
    dc_dt=-(1/theta) * (q_theta * C - qi * Ci) / delta_z - C /theta* dtheta_dt - Ros/theta*dL_dt
    return dc_dt

def derivada_ley_cobre(L, t, k, n, Loo):
    """Función derivada para ley de cobre: dL/dt"""
    dL_dt= -k*(L-Loo)*n*t**(n-1)
    return dL_dt

def RK4_sistema_ecuaciones(theta, C, L, t, dt, q_theta, qi, Ci, n, k, Loo, parametros_suelo, delta_z):

    # ========== ETAPA 1: k1 ==========
    k11 = derivada_theta(theta, qi, parametros_suelo, delta_z)
    k12 = derivada_ley_cobre(L, t, k, n, Loo)
    k13 = derivada_concentracion(C, theta, q_theta, qi, Ci, k11, k12, delta_z)
    
    # ========== ETAPA 2: k2 ==========
    k21 = derivada_theta(theta + 0.5*dt*k11, qi, parametros_suelo, delta_z)
    k22 = derivada_ley_cobre(L + 0.5*dt*k12, t + 0.5*dt, k, n, Loo)
    k23 = derivada_concentracion(C + 0.5*dt*k13, theta, q_theta, qi, Ci, k21, k22, delta_z)

    # ========== ETAPA 3: k3 ==========
    k31 = derivada_theta(theta + 0.5*dt*k21, qi, parametros_suelo, delta_z)
    k32 = derivada_ley_cobre(L + 0.5*dt*k22, t + 0.5*dt, k, n, Loo)
    k33 = derivada_concentracion(C + 0.5*dt*k23, theta, q_theta, qi, Ci, k31, k32, delta_z)

    # ========== ETAPA 4: k4 ==========
    k41 = derivada_theta(theta + dt*k31, qi, parametros_suelo, delta_z)
    k42 = derivada_ley_cobre(L + 0.5*dt*k32, t + 0.5*dt, k, n, Loo)
    k43 = derivada_concentracion(C + dt*k33, theta, q_theta, qi, Ci, k41, k42, delta_z)

    theta += (dt/6.0) * (k11 + 2*k21 + 2*k31 + k41)
    C += (dt/6.0) * (k13 + 2*k23 + 2*k33 + k43)
    L += (dt/6.0) * (k12 + 2*k22 + 2*k32 + k42)

    return theta, C, L

def obtener_tasa_riego(tiempo_actual, tasa_riego_base, duracion_riego=10.0, 
                      duracion_reposo=10.0, riego_continuo=False):
    """Controla el patrón de riego"""
    if riego_continuo:
        return tasa_riego_base
    
    ciclo_total = duracion_riego + duracion_reposo
    ciclo = tiempo_actual % ciclo_total
    if ciclo < duracion_riego:
        return tasa_riego_base
    return 0.0

# =============================================================================
# PARÁMETROS DE SIMULACIÓN
# =============================================================================

# Parámetros de Generación de Cobre
dens_real = 2650.0  # kg/m3
porosidad = 0.377   # m3/m3
dens_aparente = (1 - porosidad) * dens_real  # kg/m3
Ley_cu = 0.6/100      # kg/kg
k = 0.1/24          # m/h
Loo = 0.1/100
n_generacion = 1
Roo = (Ley_cu - Loo)/Ley_cu       # adimensional


Cobre_total_kg = 10 * dens_aparente * Ley_cu

print(f"Cobre en la pila: {Cobre_total_kg:.2f} kg")

# Parámetros de concentración
concentracion_inicial = 0.0
concentracion_entrada = 0.3

# Parámetros del suelo
theta_r = 0.05
theta_s = 0.30
n = 2.5
K_s = 5/1000 * 40  # m/h

# Parámetros de la columna
altura_total = 10.0     # m
delta_z = 0.1           # m
n_elementos = int(altura_total / delta_z)

# Parámetros temporales
dt = 0.4  # horas
dias_simulacion = 60
tiempo_total = 24 * dias_simulacion
n_pasos = int(tiempo_total / dt)

# Condiciones iniciales
theta_inicial = 0.05
tasa_riego = 10/1000  # m/h

# Parámetros para las funciones
parametros_suelo = (theta_r, theta_s, n, K_s)

# Parámetros de riego
duracion_riego = 15.0
duracion_reposo = 5.0
riego_continuo = True


# =============================================================================
# SIMULACIÓN OPTIMIZADA CON SISTEMA VERDADERAMENTE ACOPLADO
# =============================================================================

print("Iniciando simulación con sistema verdaderamente acoplado...")
print(f"Elementos: {n_elementos}, Pasos temporales: {n_pasos}")
print("Cada variable tiene sus propios k1, k2, k3, k4 con valores intermedios compartidos")

# Inicializar variables por separado
theta_columna = np.full(n_elementos, theta_inicial)
concentracion_columna = np.full(n_elementos, concentracion_inicial)
ley_columna = np.full(n_elementos, Ley_cu)

profundidades = np.arange(delta_z, altura_total + delta_z, delta_z)

# Vectores para almacenar resultados
cobre_que_sale_vector = []
cobre_recuperado_vector = []
Cu_off = 0

theta_nuevo = theta_columna.copy()
concentracion_nueva = concentracion_columna.copy()
ley_nueva = ley_columna.copy()

qi = tasa_riego
Ci = concentracion_inicial

concentraciones_salida = []
tiempos_salida = []

# Simulación principal optimizada
for paso in range(n_pasos):
    tiempo_actual = paso * dt
    qi = obtener_tasa_riego(
        tiempo_actual, tasa_riego, duracion_riego, duracion_reposo, riego_continuo
    )
    
    # Mostrar progreso cada 500 pasos
    if paso % 500 == 0:
        print(f"Paso {paso}/{n_pasos} ({100*paso/n_pasos:.1f}%) - "
              f"Tiempo: {tiempo_actual/24:.1f} días - "
              f"θ[0]={theta_nuevo[0]:.4f}, C[0]={concentracion_nueva[0]:.3f}, L[0]={ley_nueva[0]:.3f}")

    # Resolver sistema acoplado para cada elemento usando RK4 verdadero
    for i in range(n_elementos):
        
        q_theta = conductividad_hidraulica(theta_nuevo[i], theta_r, theta_s, n, K_s)

        theta_nuevo[i], concentracion_nueva[i], ley_nueva[i] = RK4_sistema_ecuaciones(
            theta_nuevo[i], concentracion_nueva[i], ley_nueva[i],
            tiempo_actual, dt, 
            q_theta,
            qi, Ci,
            n, k, Loo,
            parametros_suelo, 
            delta_z
        )
        
        # Actualizar flujo para el siguiente elemento usando el nuevo valor de thetass
        qi = conductividad_hidraulica(
            theta_nuevo[i], theta_r, theta_s, n, K_s
        )
        Ci = concentracion_nueva[i]
        
    if paso % 10 == 0:
        concentraciones_salida.append(concentracion_nueva[-1])
        tiempos_salida.append(tiempo_actual)
    
print("Simulación con sistema acoplado completada!")


# =============================================================================
# VISUALIZACIÓN DE RESULTADOS
# =============================================================================

sns.lineplot(x=tiempos_salida, y=concentraciones_salida, marker='o', markersize=3)
plt.grid(True, alpha=0.3)
plt.xlabel('Tiempo (h)')
plt.ylabel('Concentración')
plt.title('Evolución de la concentración en el último elemento')
plt.tight_layout()
plt.show()
    
# %%