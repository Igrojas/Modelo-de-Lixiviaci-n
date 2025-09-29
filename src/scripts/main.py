#%%
import numpy as np
import matplotlib.pyplot as plt

# Ecuación: L(t) = L₀₀ + (L₀ - L₀₀)e^(-kt^n)

# Parámetros
L_00 = 80    # Valor asintótico
L_0 = 0  # Valor inicial  
k = 0.1      # Constante de decaimiento
n_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]  # Valores de n entre 0.1 y 2
# Crear array de tiempo
t = np.linspace(0, 50, 1000)

# Graficar
plt.figure(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))  # Paleta de colores para mejor visualización

# Calcular y graficar L(t) para cada valor de n
for i, n in enumerate(n_values):
    L_t = L_00 + (L_0 - L_00) * np.exp(-k * t**n)
    plt.plot(t, L_t, color=colors[i], linewidth=1.5, label=f'n = {n}')

plt.xlabel('Tiempo (t)')
plt.ylabel('L(t)')
plt.title('L(t) = L₀₀ + (L₀ - L₀₀)e^(-kt^n)')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
