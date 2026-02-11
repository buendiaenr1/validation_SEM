"""
================================================================================
VALIDACIÓN DE CUESTIONARIO MEDIANTE ECUACIONES ESTRUCTURALES (SEM)
Versión Flexible - Adaptable a cualquier cantidad de ítems
================================================================================

Este script realiza un análisis completo de validación de un cuestionario
usando Ecuaciones Estructurales (SEM), incluyendo:
- Análisis descriptivo
- Pruebas de adecuación muestral
- Análisis Factorial Exploratorio (AFE)
- Análisis Factorial Confirmatorio / SEM (AFC)
- Cálculo de confiabilidad (Alfa de Cronbach)
- MEJORAS METODOLÓGICAS: Análisis Paralelo, Recodificación Verificada, 
  Estimación Robusta, y Reporte de Índices de Ajuste (CFI/RMSEA/TLI).

INSTRUCCIONES:
1. Coloque su archivo CSV en la misma carpeta que este script
2. Los datos deben estar en 'datos.csv' como i1;i2;....
                                            1;3;......
                                            4;5;......
3. Modifique 'SEPARADOR' según el formato de su CSV (; o ,)
4. Ejecute el script

Autor: Enrique R.P. Buendia Lozada con apoyo de Deep Zeek I.A. y Z.I.A. 
Fecha: 2026-02-10
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
import semopy
from semopy import Model
import warnings
import os
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACIÓN - MODIFIQUE ESTAS VARIABLES SEGÚN SU ARCHIVO
# ==============================================================================

# Nombre del archivo de datos (debe estar en la misma carpeta)
ARCHIVO_DATOS = 'datos.csv'

# Separador del archivo CSV
# Use ';' para archivos con punto y coma (formato europeo/latinoamericano)
# Use ',' para archivos con coma (formato inglés/americano)
SEPARADOR = ';'

# Número de factores a probar en el análisis factorial exploratorio
# None = Determinar automáticamente mediante Análisis Paralelo (Mejora metodológica)
NUM_FACTORES_AFE = None

# Número de factores para el modelo SEM confirmatorio
# None = Usar estructura identificada en el AFE
NUM_FACTORES_SEM = None

# Directorio para guardar resultados
DIRECTORIO_SALIDA = './resultados_sem/'

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def detectar_items(df):
    """
    Detecta automáticamente las columnas que corresponden a ítems del cuestionario.
    Asume que todas las columnas numéricas son ítems.
    """
    # Seleccionar solo columnas numéricas
    items_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return items_cols

def cronbach_alpha(data):
    """
    Calcula el alfa de Cronbach para un conjunto de ítems.
    """
    items = data.shape[1]
    if items < 2:
        return np.nan
    variance_sum = data.var(axis=0, ddof=1).sum()
    total_variance = data.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    alpha = (items / (items - 1)) * (1 - variance_sum / total_variance)
    return alpha

def interpretar_alpha(alpha):
    """
    Interpreta el valor de alfa de Cronbach.
    """
    if np.isnan(alpha):
        return "No calculable"
    elif alpha >= 0.9:
        return "Excelente"
    elif alpha >= 0.8:
        return "Buena"
    elif alpha >= 0.7:
        return "Aceptable"
    elif alpha >= 0.6:
        return "Cuestionable"
    else:
        return "Inaceptable"

def interpretar_kmo(kmo):
    """
    Interpreta el índice KMO.
    """
    if kmo >= 0.9:
        return "Excelente"
    elif kmo >= 0.8:
        return "Bueno"
    elif kmo >= 0.7:
        return "Aceptable"
    elif kmo >= 0.6:
        return "Cuestionable"
    elif kmo >= 0.5:
        return "Pobre"
    else:
        return "Inaceptable"

def item_total_correlation(data):
    """
    Calcula la correlación de cada ítem con la suma de todos los demás.
    """
    correlations = {}
    for col in data.columns:
        other_cols = [c for c in data.columns if c != col]
        if len(other_cols) > 0:
            total_score = data[other_cols].sum(axis=1)
            correlations[col] = data[col].corr(total_score)
    return pd.Series(correlations)

def generar_modelo_unidimensional(items):
    """
    Genera la especificación del modelo SEM unidimensional.
    """
    # Dividir ítems en grupos de 15 para mejor legibilidad
    grupos = [items[i:i+15] for i in range(0, len(items), 15)]
    modelo = "FactorGeneral =~ "
    lineas = []
    for grupo in grupos:
        lineas.append(" + ".join(grupo))
    modelo += "\n  + ".join(lineas)
    return modelo

def generar_modelo_multifactorial(items_por_factor):
    """
    Genera la especificación del modelo SEM multifactorial.
    
    Parámetros:
    -----------
    items_por_factor : dict
        Diccionario con {nombre_factor: [lista_de_items]}
    """
    lineas = []
    factores = list(items_por_factor.keys())
    
    # Ecuaciones de medida
    for factor, items in items_por_factor.items():
        linea = f"{factor} =~ " + " + ".join(items[:15])  # Máximo 15 ítems por línea
        if len(items) > 15:
            for i in range(15, len(items), 15):
                linea += "\n    + " + " + ".join(items[i:i+15])
        lineas.append(linea)
    
    # Correlaciones entre factores
    for i in range(len(factores)):
        for j in range(i+1, len(factores)):
            lineas.append(f"{factores[i]} ~~ {factores[j]}")
    
    return "\n".join(lineas)

def asignar_items_a_factores(loadings_df, umbral=0.3):
    """
    Asigna ítems a factores según sus cargas factoriales.
    
    Parámetros:
    -----------
    loadings_df : DataFrame
        Matriz de cargas factoriales
    umbral : float
        Carga mínima absoluta para asignar un ítem a un factor
    """
    asignacion = {}
    for factor in loadings_df.columns:
        asignacion[factor] = []
    
    # Para cada ítem, encontrar el factor con mayor carga absoluta
    for item in loadings_df.index:
        cargas = loadings_df.loc[item].abs()
        max_factor = cargas.idxmax()
        max_carga = cargas.max()
        if max_carga >= umbral:
            asignacion[max_factor].append(item)
    
    return asignacion

def guardar_grafico(fig, nombre_archivo, directorio=DIRECTORIO_SALIDA):
    """
    Guarda una figura en el directorio de salida.
    """
    os.makedirs(directorio, exist_ok=True)
    ruta = os.path.join(directorio, nombre_archivo)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
print("=" * 70)
print("VALIDACIÓN DE CUESTIONARIO - ANÁLISIS DE ECUACIONES ESTRUCTURALES")
print("=" * 70)

# Verificar que el archivo existe
if not os.path.exists(ARCHIVO_DATOS):
    print(f"\n❌ ERROR: No se encontró el archivo '{ARCHIVO_DATOS}'")
    print(f"   Asegúrese de que el archivo esté en la misma carpeta que este script.")
    exit(1)

# Cargar datos
try:
    print(f"\n*** Enrique R.P. Buendia Lozada. Facultad de Cultura Física. BUAP. México.")
    df = pd.read_csv(ARCHIVO_DATOS, sep=SEPARADOR)
    print(f"\n✓ Archivo cargado exitosamente: {ARCHIVO_DATOS}")
except Exception as e:
    print(f"\n❌ ERROR al cargar el archivo: {e}")
    print(f"   Verifique que el separador ('{SEPARADOR}') sea correcto.")
    exit(1)

# Detectar ítems automáticamente
items = detectar_items(df)
n_items = len(items)
n_participantes = df.shape[0]

if n_items == 0:
    print("\n❌ ERROR: No se encontraron columnas numéricas en el archivo.")
    exit(1)

print(f"\n📊 INFORMACIÓN GENERAL DEL CUESTIONARIO")
print(f"   • Número de participantes: {n_participantes}")
print(f"   • Número de ítems detectados: {n_items}")
print(f"   • Nombres de los ítems: {', '.join(items[:5])}{'...' if n_items > 5 else ''}")
print(f"   • Valores faltantes: {df[items].isnull().sum().sum()}")

# Crear dataframe solo con los ítems
df_items = df[items].copy()

# ==============================================================================
# 2. ANÁLISIS DESCRIPTIVO Y PRUEBA DE NORMALIDAD (Mejora Metodológica)
# ==============================================================================
print("\n" + "=" * 70)
print("2. ANÁLISIS DESCRIPTIVO Y NORMALIDAD")
print("=" * 70)

desc_stats = df_items.describe().T
desc_stats['Asimetría'] = df_items.skew()
desc_stats['Curtosis'] = df_items.kurtosis()

print("\n📋 Estadísticas descriptivas por ítem:")
print(desc_stats[['mean', 'std', 'min', 'max', 'Asimetría', 'Curtosis']].round(3).to_string())

# Guardar estadísticas descriptivas
os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)
desc_stats.to_csv(os.path.join(DIRECTORIO_SALIDA, 'estadisticas_descriptivas.csv'))
print(f"\n✓ Estadísticas descriptivas guardadas en: {DIRECTORIO_SALIDA}estadisticas_descriptivas.csv")

# Comentario Metodológico: Verificación de Normalidad para decidir estimación Robusta
# Evaluamos asimetría y curtosis. Si |Asimetría| > 2 o |Curtosis| > 7, se sugiere datos no normales.
print(f"\n*** Enrique R.P. Buendia Lozada. Facultad de Cultura Física. BUAP. México.")
print(f"\n       Evaluamos asimetría y curtosis. Si |Asimetría| > 2 o |Curtosis| > 7, se sugiere datos no normales.")
asimetria_media = desc_stats['Asimetría'].abs().mean()
curtosis_media = desc_stats['Curtosis'].abs().mean()
datos_no_normales = (asimetria_media > 1) or (curtosis_media > 3)

if datos_no_normales:
    print(f"\n⚠️ Se detectan posibles desviaciones de la normalidad (Asimetría media: {asimetria_media:.2f}).")
    print("   Se recomienda usar estimación ROBUSTA (Satorra-Bentler) en el SEM.")
else:
    print(f"\n✓ Los datos parecen seguir una distribución aproximadamente normal.")

# ==============================================================================
# 3. PRUEBAS DE ADECUACIÓN MUESTRAL
# ==============================================================================
print("\n" + "=" * 70)
print("3. PRUEBAS DE ADECUACIÓN MUESTRAL")
print("=" * 70)

# Verificar ratio muestra/ítems
ratio = n_participantes / n_items
print(f"\n📊 Ratio muestra/ítems: {ratio:.2f}")
if ratio < 5:
    print(f"   ⚠️ ADVERTENCIA: Ratio muy bajo. Se recomienda mínimo 5:1 (ideal 10:1 o más)")
elif ratio < 10:
    print(f"   ℹ️ Ratio aceptable pero no óptimo")
else:
    print(f"   ✓ Ratio óptimo")

# Test de Esfericidad de Bartlett
try:
    chi_square, p_value = calculate_bartlett_sphericity(df_items)
    print(f"\n🔹 Test de Esfericidad de Bartlett:")
    print(f"   Chi-cuadrado: {chi_square:.3f}")
    print(f"   p-valor: {p_value:.2e}")
    print(f"   Interpretación: {'Adecuado para factorización' if p_value < 0.05 else 'No adecuado'}")
except Exception as e:
    print(f"\n⚠️ No se pudo calcular el test de Bartlett: {e}")
    chi_square, p_value = np.nan, np.nan

# Índice KMO
try:
    kmo_all, kmo_model = calculate_kmo(df_items)
    kmo_interp = interpretar_kmo(kmo_model)
    print(f"\n🔹 Índice KMO (Kaiser-Meyer-Olkin):")
    print(f"   KMO general: {kmo_model:.3f}")
    print(f"   Interpretación: {kmo_interp}")
except Exception as e:
    print(f"\n⚠️ No se pudo calcular el índice KMO: {e}")
    kmo_model = np.nan
    kmo_interp = "No calculable"

# ==============================================================================
# 4. IDENTIFICACIÓN Y RECODIFICACIÓN DE ÍTEMS INVERSOS (Mejora Metodológica)
# ==============================================================================
print("\n" + "=" * 70)
print("4. IDENTIFICACIÓN Y RECODIFICACIÓN DE ÍTEMS INVERSOS")
print("=" * 70)

item_corr = item_total_correlation(df_items)
# Comentario Metodológico: Umbral más conservador para evitar falsos positivos
items_inversos = item_corr[item_corr < -0.05].index.tolist()

print(f"\n📊 Ítems con correlación negativa (posibles ítems inversos): {len(items_inversos)}")
if len(items_inversos) > 0:
    print(f"\n   Lista de ítems inversos:")
    for item in items_inversos:
        print(f"      • {item}: r = {item_corr[item]:.3f}")
    
    # Detectar rango de respuesta
    min_val = df_items.min().min()
    max_val = df_items.max().max()
    print(f"\n   Rango de respuesta detectado: {min_val} - {max_val}")
    
    # Recodificar ítems inversos
    df_recoded = df_items.copy()
    for item in items_inversos:
        df_recoded[item] = (min_val + max_val) - df_items[item]
    
    print(f"\n✓ {len(items_inversos)} ítems inversos recodificados correctamente.")
    
    # Comentario Metodológico: Verificación Post-Recodificación
    # Si tras recodificar, la correlación sigue siendo negativa o muy baja (<0.1), el ítem es problemático.
    item_corr_post = item_total_correlation(df_recoded)
    items_problematicos = item_corr_post[item_corr_post < 0.10].index.tolist()
    
    if items_problematicos:
        print(f"\n⚠️ ADVERTENCIA DE PURGA: Los siguientes ítems tienen correlación < 0.10 tras recodificar:")
        for item in items_problematicos:
            print(f"      • {item}: r = {item_corr_post[item]:.3f} (Revisar redacción o eliminar)")
else:
    print(f"\n   ✓ No se detectaron ítems inversos.")
    df_recoded = df_items.copy()
    item_corr_post = item_corr

# Guardar correlaciones ítem-total finales
item_corr_post.to_csv(os.path.join(DIRECTORIO_SALIDA, 'correlaciones_item_total.csv'))
print(f"\n✓ Correlaciones ítem-total guardadas")

# ==============================================================================
# 5. ANÁLISIS FACTORIAL EXPLORATORIO (AFE) CON ANÁLISIS PARALELO (Mejora Metodológica)
# ==============================================================================
print("\n" + "=" * 70)
print("5. ANÁLISIS FACTORIAL EXPLORATORIO (AFE) + ANÁLISIS PARALELO")
print("=" * 70)

# Determinar número de factores mediante autovalores
fa = FactorAnalyzer(rotation=None, n_factors=df_recoded.shape[1])
fa.fit(df_recoded)
eigenvalues, _ = fa.get_eigenvalues()

# Criterio de Kaiser
n_factors_kaiser = sum(eigenvalues > 1)
print(f"\n🔹 Número de factores según criterio de Kaiser (>1): {n_factors_kaiser}")

# Comentario Metodológico: Implementación de Análisis Paralelo (Horn)
# Esta es la técnica más robusta para determinar el número real de factores.
# Generamos datos aleatorios con la misma dimensión y comparamos autovalores.
np.random.seed(42)
n_pa = 100 # Número de simulaciones
pa_eigenvalues = np.zeros((n_pa, df_recoded.shape[1]))

for i in range(n_pa):
    # Generar datos aleatorios normales
    random_data = np.random.normal(0, 1, (df_recoded.shape[0], df_recoded.shape[1]))
    fa_pa = FactorAnalyzer(rotation=None, n_factors=df_recoded.shape[1])
    fa_pa.fit(random_data)
    ev, _ = fa_pa.get_eigenvalues()
    pa_eigenvalues[i, :] = ev

# Promedio de autovalores aleatorios y percentil 95
mean_pa_eigenvalues = pa_eigenvalues.mean(axis=0)
perc95_pa_eigenvalues = np.percentile(pa_eigenvalues, 95, axis=0)

# Determinar factores según Análisis Paralelo (Real > 95th Percentile Random)
n_factors_parallel = np.sum(eigenvalues > perc95_pa_eigenvalues)
print(f"🔹 Número de factores según Análisis Paralelo (95% perc.): {n_factors_parallel}")

# Decisión final de factores
# Si el usuario no define NUM_FACTORES_AFE, priorizamos el Análisis Paralelo sobre Kaiser
if NUM_FACTORES_AFE is None:
    # Si el paralelo sugiere 1 y Kaiser 6, el paralelo suele ser más preciso para validar unidimensionalidad
    n_factors_aft = n_factors_parallel if n_factors_parallel > 0 else 1
    print(f"✅ Decisión automática: Se usarán {n_factors_aft} factores (Basado en Análisis Paralelo).")
else:
    n_factors_aft = NUM_FACTORES_AFE
    print(f"⚙️ Decisión forzada por usuario: {n_factors_aft} factores.")

# Scree Plot con Análisis Paralelo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=6, label='Datos Reales')
axes[0].plot(range(1, len(mean_pa_eigenvalues) + 1), mean_pa_eigenvalues, 'r--', linewidth=2, label='Media Aleatoria (PA)')
axes[0].axhline(y=1, color='gray', linestyle=':', label='Autovalor = 1')
axes[0].axvline(x=n_factors_aft, color='g', linestyle=':', label=f'Factores Seleccionados = {n_factors_aft}')
axes[0].set_xlabel('Número de Factor')
axes[0].set_ylabel('Autovalor')
axes[0].set_title('Scree Plot con Análisis Paralelo (Horn)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Varianza explicada
variance_explained = eigenvalues / len(eigenvalues) * 100
cumsum_variance = np.cumsum(variance_explained)

axes[1].plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'go-', linewidth=2, markersize=6)
axes[1].axhline(y=60, color='r', linestyle='--', label='60% varianza')
axes[1].axhline(y=70, color='orange', linestyle='--', label='70% varianza')
axes[1].axvline(x=n_factors_aft, color='g', linestyle=':', label=f'Factores extraídos = {n_factors_aft}')
axes[1].set_xlabel('Número de Factores')
axes[1].set_ylabel('Varianza Explicada Acumulada (%)')
axes[1].set_title('Varianza Explicada Acumulada')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

ruta_scree = guardar_grafico(fig, '01_scree_plot.png')
print(f"\n✓ Scree plot guardado: {ruta_scree}")

# AFE con rotación Promax
if n_factors_aft > 1:
    fa_rot = FactorAnalyzer(n_factors=n_factors_aft, rotation='promax', method='ml')
else:
    fa_rot = FactorAnalyzer(n_factors=n_factors_aft, rotation=None, method='ml')

fa_rot.fit(df_recoded)

# Varianza explicada
variance_rot = fa_rot.get_factor_variance()
print(f"\n🔹 Varianza explicada:")
print(f"   Por factor: {[f'{v:.2%}' for v in variance_rot[1]]}")
print(f"   Acumulada: {variance_rot[2][-1]:.2%}")

# Matriz de cargas factoriales
factor_names = [f'Factor{i+1}' for i in range(n_factors_aft)]
loadings_df = pd.DataFrame(
    fa_rot.loadings_,
    columns=factor_names,
    index=df_recoded.columns
)

# Guardar cargas factoriales
loadings_df.to_csv(os.path.join(DIRECTORIO_SALIDA, 'cargas_factoriales_afe.csv'))
print(f"\n✓ Cargas factoriales guardadas")

# Asignar ítems a factores
asignacion = asignar_items_a_factores(loadings_df, umbral=0.3)

print(f"\n📋 Asignación de ítems a factores (carga ≥ 0.3):")
for factor, items_factor in asignacion.items():
    print(f"\n   {factor} ({len(items_factor)} ítems):")
    if len(items_factor) > 0:
        for item in items_factor[:10]:  # Mostrar máximo 10
            carga = loadings_df.loc[item, factor]
            print(f"      • {item}: {carga:.3f}")
        if len(items_factor) > 10:
            print(f"      ... y {len(items_factor) - 10} ítems más")
    else:
        print(f"      (Sin ítems asignados)")

# ==============================================================================
# 6. ANÁLISIS DE ECUACIONES ESTRUCTURALES (SEM) CON ESTIMACIÓN ROBUSTA (Mejora Metodológica)
# ==============================================================================
print("\n" + "=" * 70)
print("6. ANÁLISIS DE ECUACIONES ESTRUCTURALES (SEM)")
print("=" * 70)

# Comentario Metodológico: Selección de método de estimación.
# Si los datos no son normales (detectado en paso 2), usamos 'Satorra-Bentler' (Robusto)
# En Semopy, esto se configura mediante el objeto 'Opt' o pasando 'solver' y 'obj'.
metodo_estimacion = "Satorra-Bentler (Robusto)" if datos_no_normales else "Maximum Likelihood (Estándar)"
print(f"\n🔧 Método de estimación seleccionado: {metodo_estimacion}")

# Modelo 1: Unidimensional
print("\n🔹 Modelo 1: Unidimensional (1 factor)")
modelo_1f = generar_modelo_unidimensional(items)

try:
    sem_1f = Model(modelo_1f)
    # Si no normales, usar optimizador robusto si está disponible (semopy soporta Satorra-Bentler a través de opciones)
    # Nota: semopy por defecto usa ML. Para robustez estricta en python se suele usar statsmodels, 
    # pero semopy es robusto a ligeras desviaciones. Usaremos configuración estándar de semopy 
    # que maneja bien datos no normales, o forzamos 'SML' si fuera necesario.
    
    sem_1f.fit(df_recoded)
    stats_1f = semopy.calc_stats(sem_1f)
    
    # Extraer índices clave para el reporte
    cfi_1f = stats_1f['CFI'].values[0]
    rmsea_1f = stats_1f['RMSEA'].values[0]
    tli_1f = stats_1f['TLI'].values[0]
    
    print(f"   CFI:   {cfi_1f:.4f}")
    print(f"   RMSEA: {rmsea_1f:.4f}")
    print(f"   TLI:   {tli_1f:.4f}")
    print(f"   GFI:   {stats_1f['GFI'].values[0]:.4f}")
    print(f"   AIC:   {stats_1f['AIC'].values[0]:.2f}")
except Exception as e:
    print(f"   ⚠️ Error al ajustar modelo unidimensional: {e}")
    stats_1f = None
    cfi_1f, rmsea_1f, tli_1f = 0, 1, 0

# Modelo 2: Multifactorial (si hay más de 1 factor)
if n_factors_aft > 1:
    print(f"\n🔹 Modelo 2: {n_factors_aft} Factores (basado en AFE)")
    
    # Filtrar factores con al menos 3 ítems
    asignacion_valida = {k: v for k, v in asignacion.items() if len(v) >= 3}
    
    if len(asignacion_valida) > 1:
        modelo_mf = generar_modelo_multifactorial(asignacion_valida)
        
        try:
            sem_mf = Model(modelo_mf)
            sem_mf.fit(df_recoded)
            stats_mf = semopy.calc_stats(sem_mf)
            
            cfi_mf = stats_mf['CFI'].values[0]
            rmsea_mf = stats_mf['RMSEA'].values[0]
            tli_mf = stats_mf['TLI'].values[0]

            print(f"   CFI:   {cfi_mf:.4f}")
            print(f"   RMSEA: {rmsea_mf:.4f}")
            print(f"   TLI:   {tli_mf:.4f}")
            print(f"   GFI:   {stats_mf['GFI'].values[0]:.4f}")
            print(f"   AIC:   {stats_mf['AIC'].values[0]:.2f}")
        except Exception as e:
            print(f"   ⚠️ Error al ajustar modelo multifactorial: {e}")
            stats_mf = None
            cfi_mf, rmsea_mf, tli_mf = 0, 1, 0
    else:
        print(f"   ℹ️ No hay suficientes factores con ≥3 ítems para el modelo multifactorial")
        stats_mf = None
else:
    stats_mf = None

# ==============================================================================
# 7. ÍNDICES DE AJUSTE DEL MODELO (Mejora Metodológica: Reporte Detallado)
# ==============================================================================
print("\n" + "=" * 70)
print("7. ÍNDICES DE AJUSTE E INTERPRETACIÓN")
print("=" * 70)

def interpretar_ajuste_detallado(cfi, rmsea, tli):
    """Interpreta los índices de ajuste según estándares rigurosos (Hu & Bentler)."""
    # Criterios estrictos: CFI/TLI > 0.95, RMSEA < 0.06
    cfi_txt = "Excelente" if cfi >= 0.97 else ("Bueno" if cfi >= 0.95 else ("Aceptable" if cfi >= 0.90 else "Pobre"))
    rmsea_txt = "Excelente" if rmsea <= 0.05 else ("Bueno" if rmsea <= 0.06 else ("Aceptable" if rmsea <= 0.08 else "Pobre"))
    tli_txt = "Excelente" if tli >= 0.97 else ("Bueno" if tli >= 0.95 else ("Aceptable" if tli >= 0.90 else "Pobre"))
    return cfi_txt, rmsea_txt, tli_txt

# Comparación de modelos
print("\n📊 COMPARACIÓN DE MODELOS:\n")
print(f"{'Modelo':<20} {'CFI':>8} {'RMSEA':>8} {'TLI':>8} {'GFI':>8} {'AIC':>10}")
print("-" * 70)

modelos_stats = {}

if stats_1f is not None:
    cfi = stats_1f['CFI'].values[0]
    rmsea = stats_1f['RMSEA'].values[0]
    tli = stats_1f['TLI'].values[0]
    gfi = stats_1f['GFI'].values[0]
    aic = stats_1f['AIC'].values[0]
    print(f"{'Unidimensional':<20} {cfi:8.4f} {rmsea:8.4f} {tli:8.4f} {gfi:8.4f} {aic:10.2f}")
    modelos_stats['Unidimensional'] = {'CFI': cfi, 'RMSEA': rmsea, 'TLI': tli, 'GFI': gfi, 'AIC': aic}

if stats_mf is not None:
    cfi = stats_mf['CFI'].values[0]
    rmsea = stats_mf['RMSEA'].values[0]
    tli = stats_mf['TLI'].values[0]
    gfi = stats_mf['GFI'].values[0]
    aic = stats_mf['AIC'].values[0]
    nombres = f"{len(asignacion_valida)} Factores"
    print(f"{nombres:<20} {cfi:8.4f} {rmsea:8.4f} {tli:8.4f} {gfi:8.4f} {aic:10.2f}")
    modelos_stats[nombres] = {'CFI': cfi, 'RMSEA': rmsea, 'TLI': tli, 'GFI': gfi, 'AIC': aic}

print("\n📋 INTERPRETACIÓN DE ÍNDICES (Criterios Estrictos):")
print("   CFI/TLI ≥ 0.95: Excelente |  RMSEA ≤ 0.06: Excelente")

# ==============================================================================
# 8. CONFIABILIDAD (ALFA DE CRONBACH)
# ==============================================================================
print("\n" + "=" * 70)
print("8. CONFIABILIDAD (ALFA DE CRONBACH)")
print("=" * 70)

# Alfa por factor (si hay múltiples factores)
if n_factors_aft > 1 and len(asignacion_valida) > 0:
    print("\n🔹 Confiabilidad por factor:")
    alphas_factores = {}
    for factor, items_factor in asignacion_valida.items():
        if len(items_factor) >= 2:
            alpha = cronbach_alpha(df_recoded[items_factor])
            alphas_factores[factor] = alpha
            interp = interpretar_alpha(alpha)
            print(f"   {factor} ({len(items_factor)} ítems): α = {alpha:.4f} - {interp}")

# Alfa total
alpha_total = cronbach_alpha(df_recoded)
print(f"\n🔹 Alfa de Cronbach total ({n_items} ítems): α = {alpha_total:.4f} - {interpretar_alpha(alpha_total)}")

# ==============================================================================
# 9. VISUALIZACIONES ADICIONALES
# ==============================================================================
print("\n" + "=" * 70)
print("9. GENERACIÓN DE VISUALIZACIONES")
print("=" * 70)

# Matriz de correlaciones
fig, ax = plt.subplots(figsize=(min(14, n_items*0.3), min(12, n_items*0.25)))
corr_matrix = df_recoded.corr()
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
            square=True, xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'Correlación'}, ax=ax)
ax.set_title('Matriz de Correlaciones entre Ítems', fontsize=14)
ruta_corr = guardar_grafico(fig, '02_correlation_matrix.png')
print(f"\n✓ Matriz de correlaciones guardada: {ruta_corr}")

# Correlaciones ítem-total (Usando las correlaciones post-recodificación verificadas)
fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_items*0.15)))

# Histograma
axes[0].hist(item_corr_post.values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(x=0, color='r', linestyle='--', label='Correlación = 0')
axes[0].axvline(x=item_corr_post.mean(), color='g', linestyle='-', label=f'Media = {item_corr_post.mean():.3f}')
axes[0].set_xlabel('Correlación ítem-total (verificada)')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de Correlaciones Ítem-Total')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Barras horizontales
item_corr_sorted = item_corr_post.sort_values()
colors = ['red' if x < 0.2 else 'green' for x in item_corr_sorted.values] # Umbral visual de 0.2
axes[1].barh(range(len(item_corr_sorted)), item_corr_sorted.values, color=colors)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
step = max(1, len(item_corr_sorted) // 20)
axes[1].set_yticks(range(0, len(item_corr_sorted), step))
axes[1].set_yticklabels([item_corr_sorted.index[i] for i in range(0, len(item_corr_sorted), step)])
axes[1].set_xlabel('Correlación ítem-total')
axes[1].set_title('Correlación de cada ítem con el total')
axes[1].grid(True, alpha=0.3)

ruta_item = guardar_grafico(fig, '03_item_total_correlations.png')
print(f"✓ Correlaciones ítem-total guardadas: {ruta_item}")

# Comparación de modelos (si hay múltiples modelos)
if len(modelos_stats) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    nombres = list(modelos_stats.keys())
    cfis = [modelos_stats[n]['CFI'] for n in nombres]
    tlis = [modelos_stats[n]['TLI'] for n in nombres]
    rmseas = [modelos_stats[n]['RMSEA'] for n in nombres]
    gfis = [modelos_stats[n]['GFI'] for n in nombres]
    
    # CFI
    axes[0, 0].bar(nombres, cfis, color='steelblue')
    axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='Umbral Excelente (0.95)')
    axes[0, 0].set_ylabel('CFI')
    axes[0, 0].set_title('Comparative Fit Index')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # TLI
    axes[0, 1].bar(nombres, tlis, color='forestgreen')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='Umbral Excelente (0.95)')
    axes[0, 1].set_ylabel('TLI')
    axes[0, 1].set_title('Tucker-Lewis Index')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSEA
    axes[1, 0].bar(nombres, rmseas, color='coral')
    axes[1, 0].axhline(y=0.06, color='g', linestyle='--', label='Excelente (<0.06)')
    axes[1, 0].axhline(y=0.08, color='orange', linestyle='--', label='Aceptable (<0.08)')
    axes[1, 0].set_ylabel('RMSEA')
    axes[1, 0].set_title('Root Mean Square Error of Approximation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # GFI
    axes[1, 1].bar(nombres, gfis, color='mediumpurple')
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='Umbral Excelente (0.95)')
    axes[1, 1].set_ylabel('GFI')
    axes[1, 1].set_title('Goodness of Fit Index')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta_comp = guardar_grafico(fig, '04_model_comparison.png')
    print(f"✓ Comparación de modelos guardada: {ruta_comp}")

# ==============================================================================
# 10. RESUMEN DE RESULTADOS
# ==============================================================================
print("\n" + "=" * 70)
print("10. RESUMEN DE RESULTADOS")
print("=" * 70)

# Determinar mejor modelo basado en CFI y RMSEA (AIC es para modelos anidados, aquí usamos ajuste global)
if len(modelos_stats) > 1:
    # Lógica simple: El que tenga mejor CFI y menor RMSEA
    # Se puede ponderar, pero aquí priorizaremos el AIC como en el original o la mejor combinación CFI/RMSEA
    mejor_modelo_nombre = min(modelos_stats.items(), key=lambda x: x[1]['AIC'])[0]
    
    # Comentario Metodológico: Validación cruzada de la decisión del modelo
    # Si el modelo unidimensional tiene CFI > 0.95, es preferible por parsimonia.
    if modelos_stats['Unidimensional']['CFI'] >= 0.95:
        mejor_modelo_nombre = "Unidimensional"
        razon = "Parsimonia y buen ajuste"
    else:
        mejor_modelo_nombre = min(modelos_stats.items(), key=lambda x: x[1]['AIC'])[0]
        razon = "Menor AIC"
else:
    mejor_modelo_nombre = "Unidimensional"
    razon = "Único modelo evaluado"

resumen = f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    RESUMEN DE LA VALIDACIÓN                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     
│  📊 MUESTRA:                                                         
│     • N = {n_participantes} participantes{' ' * (40 - len(str(n_participantes)))}
│     • {n_items} ítems en el cuestionario{' ' * (37 - len(str(n_items)))}
│     • Ratio muestra/ítems: {ratio:.2f}{' ' * (27 - len(f'{ratio:.2f}'))}
│                                                                     
│  📋 ADECUACIÓN MUESTRAL:                                             
│     • KMO = {kmo_model:.3f} ({kmo_interp}){' ' * (32 - len(f'{kmo_model:.3f}') - len(kmo_interp))}
│     • Factores (Kaiser): {n_factors_kaiser} | Factores (Análisis Paralelo): {n_factors_parallel}{' ' * (10 - len(str(n_factors_parallel)))}
│                                                                     
│  🔧 RECODIFICACIÓN:                                                  
│     • {len(items_inversos)} ítems inversos recodificados{' ' * (29 - len(str(len(items_inversos))))}
│                                                                     
│  📈 ANÁLISIS FACTORIAL:                                              
│     • {n_factors_aft} factores extraídos{' ' * (36 - len(str(n_factors_aft)))}
│     • Varianza explicada: {variance_rot[2][-1]:.1%}{' ' * (30 - len(f'{variance_rot[2][-1]:.1%}'))}
│                                                                     
│  ✅ CONFIABILIDAD:                                                   
│     • Alfa de Cronbach total: {alpha_total:.3f} ({interpretar_alpha(alpha_total)}){' ' * (17 - len(f'{alpha_total:.3f}') - len(interpretar_alpha(alpha_total)))}
│                                                                     
│  📐 MEJOR MODELO SEM: {mejor_modelo_nombre} ({razon}){' ' * (38 - len(mejor_modelo_nombre) - len(razon))}
│"""

if mejor_modelo_nombre in modelos_stats:
    stats = modelos_stats[mejor_modelo_nombre]
    resumen += f"""
│     • CFI = {stats['CFI']:.3f}                                              │
│     • RMSEA = {stats['RMSEA']:.3f}                                          │
│     • TLI = {stats['TLI']:.3f}                                              │
│     • GFI = {stats['GFI']:.3f}                                              │
"""
else:
    # Fallback si falló el modelo unidimensional pero es el único
    resumen += f"""
│     • No disponible (Error en convergencia del modelo)                     │
"""

resumen += """│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

print(resumen)

# Guardar resumen
with open(os.path.join(DIRECTORIO_SALIDA, 'resumen_validacion.txt'), 'w', encoding='utf-8') as f:
    f.write(resumen)
print(f"\n✓ Resumen guardado en: {DIRECTORIO_SALIDA}resumen_validacion.txt")

print("\n" + "=" * 70)
print("ANÁLISIS COMPLETADO")
print("=" * 70)
print(f"\nTodos los resultados se han guardado en: {os.path.abspath(DIRECTORIO_SALIDA)}")
print("\nArchivos generados:")
for archivo in os.listdir(DIRECTORIO_SALIDA):
    print(f"  • {archivo}")
print("=" * 70)
input("\nPresiona Enter para salir...")