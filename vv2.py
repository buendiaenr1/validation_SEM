import pandas as pd
import semopy
from semopy import Model
from semopy.means import estimate_means

# 1. Carga de datos
data = pd.read_csv('datos.csv', sep=';')

# 2. Especificación del Modelo (SEM)
model_spec = """
    Bienestar =~ i1 + i2 + i3 + i4 + i5
"""

# 3. Inicialización y ajuste del modelo
mod = Model(model_spec)
res = mod.fit(data)

# 4. Obtención de resultados
inspeccion = mod.inspect()
print("--- Cargas Factoriales (Raw) ---")
print(inspeccion)

# 5. Cálculo de Índices de Ajuste
stats = semopy.calc_stats(mod)
print("\n--- Índices de Ajuste Global ---")
print(stats.T)

# 6. Cálculo de Consistencia Interna (Alfa de Cronbach)
def cronbach_alpha(df):
    item_vars = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    n_items = df.shape[1]
    return (n_items / (n_items - 1)) * (1 - (item_vars.sum() / total_var))

alpha = cronbach_alpha(data[['i1', 'i2', 'i3', 'i4', 'i5']])
print(f"\n\n Enrique R.P. Buendia Lozada [Marzo 2026]")
print(f"\nConsistencia Interna (Alfa de Cronbach): {alpha:.3f}")

# 7. Cálculo de Fiabilidad (Omega de McDonald) - VERSIÓN FINAL CORREGIDA
def mcdonald_omega(model_object):
    """
    Calcula el Omega de McDonald correctamente filtrando la tabla de inspección.
    """
    insp = model_object.inspect()
    
    # CORRECCIÓN CRÍTICA:
    # En semopy, 'Bienestar =~ i1' se guarda como 'i1 ~ Bienestar'.
    # Por tanto, el factor 'Bienestar' está en la columna derecha (rval), no en la izquierda (lval).
    params = insp[(insp['rval'] == 'Bienestar') & (insp['op'] == '~')]
    
    # Seleccionar la columna de cargas (priorizando estandarizadas, usando Estimate como fallback)
    if 'Std. Est' in params.columns:
        loadings = params['Std. Est']
    elif 'Std.Est' in params.columns:
        loadings = params['Std.Est']
    elif 'Value' in params.columns:
        loadings = params['Value']
    else:
        # Fallback a Estimate (que es estandarizado si el factor latente tiene varianza = 1)
        loadings = params['Estimate']
    
    # Cálculo de la varianza del error único (theta)
    error_variances = 1 - loadings**2
    
    # Fórmula de Omega de McDonald
    sum_loadings = loadings.sum()
    sum_errors = error_variances.sum()
    
    omega = (sum_loadings ** 2) / ((sum_loadings ** 2) + sum_errors)
    return omega

# Ejecutar el cálculo de Omega
omega = mcdonald_omega(mod)
print(f"Fiabilidad (Omega de McDonald): {omega:.3f}")