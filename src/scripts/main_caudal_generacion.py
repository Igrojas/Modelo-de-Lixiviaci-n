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

def derivada_theta(theta, C, L, t, q_entrada, parametros_suelo, delta_z):
    """Función derivada para humedad: dtheta/dt"""
    theta_r, theta_s, n, K_s = parametros_suelo
    q_salida = conductividad_hidraulica(theta, theta_r, theta_s, n, K_s)
    
    if theta <= 0:
        return 0.0
    return (q_entrada - q_salida) / delta_z

def derivada_concentracion(theta, C, L, t, q_entrada, parametros_suelo, parametros_cobre, delta_z):
    """Función derivada para concentración: dC/dt"""
    theta_r, theta_s, n, K_s = parametros_suelo
    k_gen, n_gen, L_infty, rho_aparente, C_entrada = parametros_cobre
    
    if theta <= 0:
        return 0.0
    
    # Calcular q(theta) y dtheta/dt
    q_theta = conductividad_hidraulica(theta, theta_r, theta_s, n, K_s)
    dtheta_dt = derivada_theta(theta, C, L, t, q_entrada, parametros_suelo, delta_z)
    
    # Calcular dL/dt
    dL_dt = -k_gen * n_gen * t**(n_gen-1) * (L - L_infty)
    
    return -(1/theta) * (
        (q_theta * C) / delta_z - 
        (q_entrada * C_entrada) / delta_z + 
        C * dtheta_dt + 
        rho_aparente * dL_dt
    )

def derivada_ley_cobre(theta, C, L, t, parametros_cobre):
    """Función derivada para ley de cobre: dL/dt"""
    k_gen, n_gen, L_infty, rho_aparente, C_entrada = parametros_cobre
    return -k_gen * n_gen * t**(n_gen-1) * (L - L_infty)

def runge_kutta_4_sistema_acoplado(theta, C, L, t, dt, q_entrada, parametros_suelo, parametros_cobre, delta_z):
    """
    Runge-Kutta de 4° orden para sistema acoplado de 3 ecuaciones.
    Cada variable tiene sus propios k1, k2, k3, k4 pero se evalúan usando 
    valores intermedios de TODAS las variables.
    """
    
    # ========== ETAPA 1: k1 ==========
    k1_theta = dt * derivada_theta(theta, C, L, t, q_entrada, parametros_suelo, delta_z)
    k1_C = dt * derivada_concentracion(theta, C, L, t, q_entrada, parametros_suelo, parametros_cobre, delta_z)
    k1_L = dt * derivada_ley_cobre(theta, C, L, t, parametros_cobre)
    
    # ========== ETAPA 2: k2 ==========
    theta_temp = theta + k1_theta/2
    C_temp = C + k1_C/2
    L_temp = L + k1_L/2
    t_temp = t + dt/2
    
    k2_theta = dt * derivada_theta(theta_temp, C_temp, L_temp, t_temp, q_entrada, parametros_suelo, delta_z)
    k2_C = dt * derivada_concentracion(theta_temp, C_temp, L_temp, t_temp, q_entrada, parametros_suelo, parametros_cobre, delta_z)
    k2_L = dt * derivada_ley_cobre(theta_temp, C_temp, L_temp, t_temp, parametros_cobre)
    
    # ========== ETAPA 3: k3 ==========
    theta_temp = theta + k2_theta/2
    C_temp = C + k2_C/2
    L_temp = L + k2_L/2
    # t_temp sigue siendo t + dt/2
    
    k3_theta = dt * derivada_theta(theta_temp, C_temp, L_temp, t_temp, q_entrada, parametros_suelo, delta_z)
    k3_C = dt * derivada_concentracion(theta_temp, C_temp, L_temp, t_temp, q_entrada, parametros_suelo, parametros_cobre, delta_z)
    k3_L = dt * derivada_ley_cobre(theta_temp, C_temp, L_temp, t_temp, parametros_cobre)
    
    # ========== ETAPA 4: k4 ==========
    theta_temp = theta + k3_theta
    C_temp = C + k3_C
    L_temp = L + k3_L
    t_temp = t + dt
    
    k4_theta = dt * derivada_theta(theta_temp, C_temp, L_temp, t_temp, q_entrada, parametros_suelo, delta_z)
    k4_C = dt * derivada_concentracion(theta_temp, C_temp, L_temp, t_temp, q_entrada, parametros_suelo, parametros_cobre, delta_z)
    k4_L = dt * derivada_ley_cobre(theta_temp, C_temp, L_temp, t_temp, parametros_cobre)
    
    # ========== ACTUALIZACIÓN FINAL ==========
    theta_new = theta + (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)/6
    C_new = C + (k1_C + 2*k2_C + 2*k3_C + k4_C)/6
    L_new = L + (k1_L + 2*k2_L + 2*k3_L + k4_L)/6
    
    return theta_new, C_new, L_new

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
Ley_cu = 0.0090      # kg/kg
R_infty = 0.85      # adimensional
k = 0.1/24          # m/h
L_infty = Ley_cu - R_infty * Ley_cu  # kg/kg
n_generacion = 1
Cobre_total_kg = 10 * dens_aparente * Ley_cu

print(f"Cobre en la pila: {Cobre_total_kg:.2f} kg")

# Parámetros de concentración
concentracion_inicial = 3.2
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
dt = 0.5  # horas
dias_simulacion = 60
tiempo_total = 24 * dias_simulacion
n_pasos = int(tiempo_total / dt)

# Condiciones iniciales
theta_inicial = 0.12
tasa_riego = 10/1000  # m/h

# Parámetros para las funciones
parametros_suelo = (theta_r, theta_s, n, K_s)
parametros_cobre = (k, n_generacion, L_infty, dens_aparente, concentracion_entrada)

# Parámetros de riego
duracion_riego = 15.0
duracion_reposo = 5.0
riego_continuo = False


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

# Simulación principal optimizada
for paso in range(n_pasos):
    tiempo_actual = paso * dt
    q_entrada = obtener_tasa_riego(
        tiempo_actual, tasa_riego, duracion_riego, duracion_reposo, riego_continuo
    )
    
    # Mostrar progreso cada 500 pasos
    if paso % 500 == 0:
        print(f"Paso {paso}/{n_pasos} ({100*paso/n_pasos:.1f}%) - "
              f"Tiempo: {tiempo_actual/24:.1f} días - "
              f"θ[0]={theta_columna[0]:.4f}, C[0]={concentracion_columna[0]:.3f}")
    
    # Crear copias para almacenar nuevos valores
    theta_nuevo = np.zeros(n_elementos)
    concentracion_nueva = np.zeros(n_elementos)
    ley_nueva = np.zeros(n_elementos)
    
    q_actual = q_entrada
    
    # Resolver sistema acoplado para cada elemento usando RK4 verdadero
    for i in range(n_elementos):
        theta_nuevo[i], concentracion_nueva[i], ley_nueva[i] = runge_kutta_4_sistema_acoplado(
            theta_columna[i], 
            concentracion_columna[i], 
            ley_columna[i],
            tiempo_actual, 
            dt, 
            q_actual, 
            parametros_suelo, 
            parametros_cobre, 
            delta_z
        )
        
        # Actualizar flujo para el siguiente elemento usando el nuevo valor de theta
        q_actual = conductividad_hidraulica(
            theta_nuevo[i], theta_r, theta_s, n, K_s
        )
    
    # Actualizar todas las variables simultáneamente
    theta_columna = theta_nuevo.copy()
    concentracion_columna = concentracion_nueva.copy()
    ley_columna = ley_nueva.copy()
    
    # Calcular cobre que sale
    q_salida_final = conductividad_hidraulica(
        theta_columna[-1], theta_r, theta_s, n, K_s
    )
    Cu_off += (concentracion_columna[-1] * q_salida_final - concentracion_entrada * tasa_riego) * dt
    
    cobre_que_sale_vector.append(Cu_off)
    cobre_recuperado_vector.append(Cu_off / Cobre_total_kg * 100)

print("Simulación con sistema acoplado completada!")

#%%
# =============================================================================
# VISUALIZACIÓN DE RESULTADOS
# =============================================================================
type(cobre_recuperado_vector)

for i in range(len(cobre_recuperado_vector)):
    if cobre_recuperado_vector[i+1] > cobre_recuperado_vector[i]:
        print(i , cobre_recuperado_vector[i])
    
#%%
plt.plot(np.array(cobre_recuperado_vector), marker='o', markersize=1, linestyle='--')
plt.grid(True, alpha=0.3)
plt.xlabel('Tiempo (días)')
plt.ylabel('Cobre Recuperado (%)')
plt.show()

#%%
# Gráfico principal de evolución temporal
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

tiempo_dias = np.arange(len(cobre_que_sale_vector)) * dt / 24

# Gráfico 1: Cobre recuperado
ax1.plot(tiempo_dias, cobre_recuperado_vector, 'b-', linewidth=2, label='% Cobre Recuperado')
ax1.set_xlabel('Tiempo (días)')
ax1.set_ylabel('Cobre Recuperado (%)')
ax1.set_title('Evolución de la Recuperación de Cobre - Sistema Acoplado RK4')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Gráfico 2: Masa de cobre que sale
ax2.plot(tiempo_dias, cobre_que_sale_vector, 'r-', linewidth=2, label='Cobre que sale (kg)')
ax2.set_xlabel('Tiempo (días)')
ax2.set_ylabel('Cobre que sale (kg)')
ax2.set_title('Masa Acumulada de Cobre que Sale de la Columna')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# # Gráfico de perfiles finales
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# # Perfil de humedad
# ax1.plot(theta_columna, profundidades, 'b-', linewidth=2)
# ax1.set_xlabel('Contenido de Humedad θ')
# ax1.set_ylabel('Profundidad (m)')
# ax1.set_title('Perfil Final de Humedad')
# ax1.invert_yaxis()
# ax1.grid(True, alpha=0.3)

# # Perfil de concentración
# ax2.plot(concentracion_columna, profundidades, 'g-', linewidth=2)
# ax2.set_xlabel('Concentración (kg/m³)')
# ax2.set_ylabel('Profundidad (m)')
# ax2.set_title('Perfil Final de Concentración')
# ax2.invert_yaxis()
# ax2.grid(True, alpha=0.3)

# # Perfil de ley de cobre
# ax3.plot(ley_columna, profundidades, 'r-', linewidth=2)
# ax3.set_xlabel('Ley de Cobre (kg/kg)')
# ax3.set_ylabel('Profundidad (m)')
# ax3.set_title('Perfil Final de Ley de Cobre')
# ax3.invert_yaxis()
# ax3.grid(True, alpha=0.3)

# # Perfil de conductividad hidráulica
# K_perfil = [conductividad_hidraulica(theta, theta_r, theta_s, n, K_s) for theta in theta_columna]
# ax4.plot(K_perfil, profundidades, 'm-', linewidth=2)
# ax4.set_xlabel('Conductividad Hidráulica (m/h)')
# ax4.set_ylabel('Profundidad (m)')
# ax4.set_title('Perfil Final de Conductividad Hidráulica')
# ax4.invert_yaxis()
# ax4.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# Resumen de resultados
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS - SISTEMA ACOPLADO RK4")
print("="*60)
print(f"Método: Runge-Kutta 4° orden con sistema verdaderamente acoplado")
print(f"Cada variable (θ, C, L) tiene sus propios k1, k2, k3, k4")
print(f"Evaluados usando valores intermedios compartidos")
print("-"*60)
print(f"Tiempo total de simulación: {dias_simulacion} días")
print(f"Cobre total en la pila: {Cobre_total_kg:.2f} kg")
print(f"Cobre recuperado final: {Cu_off:.2f} kg")
print(f"Porcentaje de recuperación: {cobre_recuperado_vector[-1]:.2f}%")
print(f"Humedad promedio final: {np.mean(theta_columna):.4f}")
print(f"Concentración promedio final: {np.mean(concentracion_columna):.4f} kg/m³")
print(f"Ley promedio final: {np.mean(ley_columna):.6f} kg/kg")
print("="*60)
# %%
