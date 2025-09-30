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
def derivada_L(L,t,Loo,k,n):
    dL_dt=-k*(L-Loo)*n*t**(n-1)
    return dL_dt
def derivada_concentracion(C, theta, q_theta, qi, Ci, dtheta_dt,dL_dt, delta_z):
    Ros=1650
    """Calcula dC/dt para transporte de solutos"""
    if theta <= 0:
        return 0.0
    dc_dt=-(1/theta) * (q_theta * C - qi * Ci) / delta_z - C /theta* dtheta_dt - Ros/theta*dL_dt
    return dc_dt
def paso_runge_kutta_sistema_ecuaciones(theta, C,L,t ,q_theta,qi, Ci, dt, parametros,Loo,k_l,n_l, delta_z):
    """Un paso de integración Runge-Kutta de 4to orden"""
    k10 = derivada_theta(theta, qi, parametros, delta_z)
    k11 = derivada_L(L, t, Loo,k_l,n_l)
    k12 = derivada_concentracion(C, theta, q_theta, qi, Ci, k10,k11, delta_z)

    k20 = derivada_theta(theta + 0.5*dt*k10, qi, parametros, delta_z)
    k21 = derivada_L(L+0.5*dt*k11, t+0.5*dt,Loo, k_l,n_l)
    k22 = derivada_concentracion(C + 0.5*dt*k12, theta,q_theta, qi, Ci, k20,k21, delta_z)

    k30 = derivada_theta(theta + 0.5*dt*k20, qi, parametros, delta_z)
    k31 = derivada_L(L+0.5*dt*k21, t+0.5*dt,Loo, k_l,n_l)
    k32 = derivada_concentracion(C + 0.5*dt*k22, theta,q_theta,  qi, Ci, k30,k31, delta_z)

    k40 = derivada_theta(theta + dt*k30, qi, parametros, delta_z)
    k41 = derivada_L(L+0.5*dt*k31, t+0.5*dt,Loo, k_l,n_l)
    k42 = derivada_concentracion(C + dt*k32, theta,q_theta, qi, Ci, k40,k41, delta_z)

    theta += (dt/6.0) * (k10 + 2*k20 + 2*k30 + k40)
    L += (dt/6.0) * (k11 + 2*k21 + 2*k31 + k41)
    C += (dt/6.0) * (k12 + 2*k22 + 2*k32 + k42)
    return theta,L,C

# =============================================================================
# PARÁMETROS DE SIMULACIÓN
# =============================================================================

# Parámetros de concentración
concentracion_inicial = 0.0    # Concentración inicial en la columna
concentracion_entrada = 0.3   # Concentración del agua de riego

# Parámetros del suelo
theta_r = 0.05      # Contenido residual
theta_s = 0.30      # Contenido de saturación
n = 2.5             # Parámetro de forma van Genuchten
K_s = 5/1000 * 40   # Conductividad saturada (m/h)

par_Loo=0.1/100
par_k=0.1/24
par_n=1.0

# Parámetros de la columna
area_columna=1.0 #m2
altura_total = 10.0     # m
delta_z = 0.1          # m (espaciado entre elementos)
n_elementos = int(altura_total / delta_z)
PeA=1650
# Parámetros temporales
dt: float = 0.1  # Paso de tiempo en horas
dias_simulacion: int = 60
tiempo_total: float = 24 * dias_simulacion  # Total de horas a simular (10 días)
n_pasos: int = int(tiempo_total / dt)

# Condiciones iniciales
theta_inicial: float = 0.05
L_inicial= 0.6/100
tasa_riego: float = 10/1000  # m/h (flujo de entrada en superficie)
Roo=(L_inicial-par_Loo)/L_inicial

peso_columna=area_columna*altura_total*PeA
CuF=peso_columna*L_inicial


# =============================================================================
# SIMULACIÓN
# =============================================================================

print("Iniciando simulación de infiltración...")
print(f"Elementos: {n_elementos}, Pasos temporales: {n_pasos}")

# Inicializar
parametros = (theta_r, theta_s, n, K_s)
theta_columna = np.full(n_elementos, theta_inicial)
L_columna = np.full(n_elementos, L_inicial)
concentracion_columna = np.full(n_elementos, concentracion_inicial)  # Nueva línea

# Crear concentración inicial con variación aleatoria alrededor del valor base
np.random.seed(42)  # Para reproducibilidad
variacion = 0.5  # 10% de variación
concentracion_columna = concentracion_inicial * (1 + variacion * (np.random.random(n_elementos) - 0.5))
profundidades = np.arange(delta_z, altura_total + delta_z, delta_z)

# Almacenar resultados
resultados_theta = []
resultados_L = []
resultados_concentracion = []  # Nueva línea
rec_off=[]
tpo_off=[]
recuperacion_off=[]
theta_columna_actual = theta_columna.copy()
concentracion_columna_actual = concentracion_columna.copy()  # Nueva línea
L_columna_actual = L_columna.copy()  # Nueva línea
# Almacenar concentraciones de salida y sus tiempos correspondientes
concentraciones_salida = []
tiempos_salida = []


duracion_riego = 15.0
duracion_reposo = 5.0
riego_continuo = True  # Cambiar a True para riego continuo
delta_rec_off=0
# Simulación principal
for paso in range(n_pasos):
    tiempo_actual = paso * dt
    qi = 10.0/1000.0
    ci=concentracion_entrada
    for i in range(n_elementos):
         # Calcular q(theta) actual
        q_theta = conductividad_hidraulica(theta_columna_actual[i], theta_r, theta_s, n, K_s)
        # Actualizar theta
        theta_columna_actual[i],L_columna_actual[i],concentracion_columna_actual[i] =paso_runge_kutta_sistema_ecuaciones(
                                                                                    theta_columna_actual[i],
                                                                                    concentracion_columna_actual[i],
                                                                                    L_columna_actual[i],
                                                                                    tiempo_actual,
                                                                                    q_theta,
                                                                                    qi,ci,dt,
                                                                                    parametros,
                                                                                    par_Loo,par_k,par_n, 
                                                                                    delta_z)
                # El flujo de salida para el siguiente elemento
        qi = conductividad_hidraulica(theta_columna_actual[i], theta_r, theta_s, n, K_s)
        ci=concentracion_columna_actual[i]


    cuf_on=area_columna*qi*dt*concentracion_entrada  #kg Cu/h
    cuf_off=area_columna*qi*dt*ci  #kg Cu/h
    delta_cuf=cuf_off-cuf_on
    delta_rec_off +=delta_cuf/CuF*100
    rec_off.append(delta_rec_off)
    tpo_off.append(tiempo_actual)
    # --- impresión de resultados ---
    # print("Resultados balance de cobre:")
    # print(f"cuf_on   = {cuf_on:.4f}   kg Cu/h")
    # print(f"cuf_off  = {cuf_off:.4f}   kg Cu/h")
    # print(f"delta_cuf = {delta_cuf:.4f}   kg Cu/h")
    # print(f"delta_rec_off acumulado = {delta_rec_off:.4f}   %")
    # print()

    # Guardar cada 10 pasos (~cada 1 hora)
    if paso % 10 == 0:
        resultados_theta.append(theta_columna_actual.copy())
        resultados_L.append(L_columna_actual.copy())
        resultados_concentracion.append(concentracion_columna_actual.copy())
        # Guardar la concentración de salida (último elemento) y su tiempo correspondiente
        concentraciones_salida.append(concentracion_columna_actual[-1])
        recuperacion_off.append(rec_off[-1])
        tiempos_salida.append(tiempo_actual)

tiempo_dias=np.array(tiempos_salida, dtype=float)/24

print("Simulación completada!")
recuperacion_off = np.ravel(recuperacion_off)  # asegura que sea 1D
#%%
#Recuperacion
plt.figure(figsize=(10, 6), dpi=140)
plt.plot(tiempo_dias, Roo*100*(1-np.exp(-par_k* np.array(tiempos_salida, dtype=float))), 'b-', linewidth=2, marker='o', markersize=3)
plt.plot(tiempo_dias, recuperacion_off, 'b-', linewidth=2, marker='o', markersize=3)
plt.grid(True, alpha=0.3)
plt.xlabel('Tiempo (h)')
plt.ylabel('Concentración')
plt.title('Evolución de la concentración en el último elemento')
plt.tight_layout()
plt.show()


# Graficar la curva de concentración en el último elemento
plt.figure(figsize=(10, 6), dpi=140)
plt.plot(tiempo_dias, concentraciones_salida, 'b-', linewidth=2, marker='o', markersize=3)
plt.grid(True, alpha=0.3)
plt.xlabel('Tiempo (h)')
plt.ylabel('Concentración')
plt.title('Evolución de la concentración en el último elemento')
plt.tight_layout()
plt.show()

print("Parámetros usados en conductividad_hidraulica:")
print(f"theta_r  = {theta_r}")
print(f"theta_s  = {theta_s}")
print(f"n        = {n}")
print(f"K_s      = {K_s}")

theta = np.linspace(0.05, 0.3, 50)
qout = conductividad_hidraulica(theta, theta_r, theta_s, n, K_s) * 1000
# Tomar el último valor de resultados_theta
theta_sim = float(resultados_theta[-1].max())
# Calcular q_sim para ese único theta_sim
q_sim = conductividad_hidraulica(theta_sim, theta_r, theta_s, n, K_s) * 1000
# --- Gráfico principal ---
plt.plot(theta, qout, label="Curva q(θ)")
plt.yscale("log")
# Dibujar una línea vertical y horizontal en el punto (theta_sim, q_sim)
plt.axvline(theta_sim, color="r", linestyle="--", label=f"θ_sim = {theta_sim:.3f}")
plt.axhline(q_sim, color="g", linestyle="--", label=f"q_sim = {q_sim:.1f}")
# Marcar el punto exacto
plt.plot(theta_sim, q_sim, "ro")
# Etiquetas
plt.title(f"Resultado Theta en simulación = {theta_sim:.4f}")
plt.xlabel("θ")
plt.ylabel("qout")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.show()# %%

# %%
