# pip install factor-analyzer
# pip install seaborn
# pip install semopy
# pip install scipy
# pip install statsmodels


"""
================================================================================
VALIDACIÓN DE CUESTIONARIO MEDIANTE ECUACIONES ESTRUCTURALES (SEM)
Versión Mejorada - Adaptable a cualquier cantidad de ítems
================================================================================

Este script realiza un análisis completo de validación de un cuestionario
usando Ecuaciones Estructurales (SEM), incluyendo:
- Análisis descriptivo
- Pruebas de adecuación muestral
- Análisis Factorial Exploratorio (AFE)
- Análisis Factorial Confirmatorio / SEM (AFC)
- Cálculo de confiabilidad (Alfa de Cronbach)
- Análisis de validez discriminante y convergente
- Comparación exhaustiva de modelos

INSTRUCCIONES:
1. Coloque su archivo CSV en la misma carpeta que este script
2. Modifique la variable 'ARCHIVO_DATOS' con el nombre de su archivo
3. Modifique 'SEPARADOR' según el formato de su CSV (; o ,)
4. Ejecute el script

Autor: Enrique R.P. Buendia Lozada apoyo de KIMI I.A, Z i.A. 
Fecha: 2026-02-10
Versión Mejorada: 2026-02-17
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
from scipy import stats
from scipy.stats import shapiro, normaltest
import statsmodels.api as sm
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
# None = Determinar automáticamente según criterio de Kaiser
NUM_FACTORES_AFE = None

# Número de factores para el modelo SEM confirmatorio
# None = Usar estructura identificada en el AFE
NUM_FACTORES_SEM = None

# Directorio para guardar resultados
DIRECTORIO_SALIDA = './resultados_sem/'

# ==============================================================================
# GLOSARIO DE ABREVIACIONES Y TÉRMINOS ESTADÍSTICOS
# ==============================================================================

def imprimir_glosario():
    """Imprime el glosario completo de abreviaciones y términos."""
    glosario = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                           GLOSARIO DE ABREVIACIONES                             ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  ANÁLISIS DESCRIPTIVO:                                                          ║
║  • N          = Tamaño de la muestra (número de participantes)                 ║
║  • M / Mean   = Media aritmética (promedio)                                      ║
║  • SD / Std   = Desviación estándar (dispersión de los datos)                  ║
║  • Min        = Valor mínimo observado                                          ║
║  • Max        = Valor máximo observado                                          ║
║  • Skewness   = Asimetría (0 = simétrico, >0 = cola derecha, <0 = cola izq)    ║
║  • Kurtosis   = Curtosis (forma de la distribución, 3 = normal)                ║
║                                                                                 ║
║  PRUEBAS DE ADECUACIÓN MUESTRAL:                                                ║
║  • KMO        = Kaiser-Meyer-Olkin (mide adecuación para factorización)        ║
║                Rango: 0-1. Valores >0.7 son aceptables, >0.9 excelentes         ║
║  • Bartlett   = Test de esfericidad de Bartlett                                 ║
║                Prueba de si la matriz de correlaciones es una matriz identidad  ║
║                H0: Las variables no están correlacionadas                       ║
║                p < 0.05 indica que SÍ hay correlaciones significativas         ║
║                                                                                 ║
║  ANÁLISIS FACTORIAL:                                                            ║
║  • AFE        = Análisis Factorial Exploratorio                                 ║
║                Descubre la estructura subyacente sin hipótesis previas          ║
║  • AFC/SEM    = Análisis Factorial Confirmatorio / Ecuaciones Estructurales    ║
║                Prueba hipótesis específicas sobre la estructura factorial       ║
║  • λ / Lambda = Cargas factoriales (correlación ítem-factor)                   ║
║                Valores >0.3 son aceptables, >0.5 buenos, >0.7 excelentes        ║
║  • h²         = Comunalidad (varianza explicada por los factores)              ║
║  • EV / Eigen = Autovalor (varianza explicada por cada factor)                 ║
║                Kaiser: conservar factores con EV > 1                            ║
║  • %VE        = Porcentaje de varianza explicada                               ║
║                                                                                 ║
║  ROTACIONES:                                                                    ║
║  • Varimax    = Rotación ortogonal (factores independientes)                   ║
║  • Promax     = Rotación oblicua (permite correlación entre factores)          ║
║  • Oblimin    = Otra rotación oblicua                                           ║
║                                                                                 ║
║  ÍNDICES DE AJUSTE DEL MODELO (GOF - Goodness of Fit):                          ║
║  • χ² / Chi²  = Chi-cuadrado (prueba de ajuste absoluto)                       ║
║                p > 0.05 indica buen ajuste, pero sensible al tamaño muestral    ║
║  • df         = Grados de libertad                                              ║
║  • χ²/df      = Chi-cuadrado dividido por grados de libertad                   ║
║                Valores < 2 son excelentes, < 3 aceptables, < 5 tolerables       ║
║  • CFI        = Comparative Fit Index (índice de ajuste comparativo)           ║
║                Rango: 0-1. >0.95 excelente, >0.90 aceptable, >0.80 marginal     ║
║  • TLI / NNFI = Tucker-Lewis Index / Non-Normed Fit Index                      ║
║                Similar a CFI, penaliza modelos complejos. >0.95 excelente       ║
║  • RMSEA      = Root Mean Square Error of Approximation                        ║
║                Error de aproximación. <0.05 excelente, <0.08 aceptable         ║
║                RMSEA 90% CI = Intervalo de confianza del 90%                   ║
║  • SRMR       = Standardized Root Mean Square Residual                         ║
║                Residual estandarizado. <0.05 excelente, <0.08 aceptable        ║
║  • GFI        = Goodness of Fit Index (índice de bondad de ajuste)             ║
║                Rango: 0-1. >0.95 excelente, >0.90 aceptable                     ║
║  • AGFI       = Adjusted GFI (ajustado por grados de libertad)                 ║
║                Similar a GFI pero penaliza complejidad                          ║
║  • NFI        = Normed Fit Index (índice de ajuste normado)                    ║
║                Rango: 0-1. >0.95 excelente                                      ║
║  • AIC        = Akaike Information Criterion (criterio de información)         ║
║                Penaliza complejidad. MENOR es MEJOR para comparar modelos       ║
║  • BIC        = Bayesian Information Criterion                                  ║
║                Similar a AIC pero penaliza más la complejidad                   ║
║                                                                                 ║
║  CONFIABILIDAD:                                                                 ║
║  • α / Alpha  = Alfa de Cronbach (consistencia interna)                        ║
║                Rango: 0-1. >0.9 excelente, >0.8 bueno, >0.7 aceptable          ║
║  • α si se elimina = Alfa si se elimina cada ítem                              ║
║                Ayuda a identificar ítems problemáticos                          ║
║  • CR / ρc    = Composite Reliability (confiabilidad compuesta)                ║
║                Alternativa al alfa, no asume tau-equivalencia. >0.7 aceptable  ║
║  • AVE        = Average Variance Extracted (varianza extraída promedio)        ║
║                >0.5 indica validez convergente                                  ║
║                                                                                 ║
║  VALIDEZ:                                                                       ║
║  • MSV        = Maximum Shared Variance (máxima varianza compartida)           ║
║  • ASV        = Average Shared Variance (varianza compartida promedio)         ║
║  • HTMT       = Heterotrait-Monotrait ratio (ratio heterotraz-mono)            ║
║                <0.85 indica validez discriminante                               ║
║  • Fornell-Larcker = Criterio de Fornell-Larcker                               ║
║                AVE > correlaciones al cuadrado entre factores                   ║
║                                                                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""
    print(glosario)
    return glosario

# ==============================================================================
# FUNCIONES AUXILIARES MEJORADAS
# ==============================================================================

def detectar_items(df):
    """
    Detecta automáticamente las columnas que corresponden a ítems del cuestionario.
    Asume que todas las columnas numéricas son ítems.
    """
    items_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return items_cols

def cronbach_alpha(data):
    """
    Calcula el alfa de Cronbach para un conjunto de ítems.
    Fórmula: α = (k/(k-1)) * (1 - Σσ²ᵢ/σ²ₜ)
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

def cronbach_alpha_detallado(data):
    """
    Calcula el alfa de Cronbach detallado con estadísticas por ítem.
    """
    items = data.columns.tolist()
    n_items = len(items)

    if n_items < 2:
        return None, None

    alpha_total = cronbach_alpha(data)
    alphas_eliminados = {}

    for item in items:
        items_restantes = [c for c in items if c != item]
        alpha_sin_item = cronbach_alpha(data[items_restantes])
        alphas_eliminados[item] = alpha_sin_item

    return alpha_total, alphas_eliminados

def interpretar_alpha(alpha):
    """Interpreta el valor de alfa de Cronbach."""
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
    """Interpreta el índice KMO."""
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
    """Calcula la correlación de cada ítem con la suma de todos los demás."""
    correlations = {}
    for col in data.columns:
        other_cols = [c for c in data.columns if c != col]
        if len(other_cols) > 0:
            total_score = data[other_cols].sum(axis=1)
            correlations[col] = data[col].corr(total_score)
    return pd.Series(correlations)

def generar_modelo_unidimensional(items):
    """Genera la especificación del modelo SEM unidimensional."""
    grupos = [items[i:i+15] for i in range(0, len(items), 15)]
    modelo = "FactorGeneral =~ "
    lineas = []
    for grupo in grupos:
        lineas.append(" + ".join(grupo))
    modelo += "\n  + ".join(lineas)
    return modelo

def generar_modelo_multifactorial(items_por_factor):
    """Genera la especificación del modelo SEM multifactorial."""
    lineas = []
    factores = list(items_por_factor.keys())

    for factor, items in items_por_factor.items():
        linea = f"{factor} =~ " + " + ".join(items[:15])
        if len(items) > 15:
            for i in range(15, len(items), 15):
                linea += "\n    + " + " + ".join(items[i:i+15])
        lineas.append(linea)

    for i in range(len(factores)):
        for j in range(i+1, len(factores)):
            lineas.append(f"{factores[i]} ~~ {factores[j]}")

    return "\n".join(lineas)

def asignar_items_a_factores(loadings_df, umbral=0.3):
    """Asigna ítems a factores según sus cargas factoriales máximas."""
    asignacion = {factor: [] for factor in loadings_df.columns}

    for item in loadings_df.index:
        cargas = loadings_df.loc[item].abs()
        max_factor = cargas.idxmax()
        max_carga = cargas.max()
        if max_carga >= umbral:
            asignacion[max_factor].append(item)

    return asignacion

def guardar_grafico(fig, nombre_archivo, directorio=DIRECTORIO_SALIDA):
    """Guarda una figura en el directorio de salida."""
    os.makedirs(directorio, exist_ok=True)
    ruta = os.path.join(directorio, nombre_archivo)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta

def calcular_composite_reliability(loadings):
    """
    Calcula la confiabilidad compuesta (CR).
    Fórmula: CR = (Σλ)² / [(Σλ)² + Σ(1-λ²)]
    """
    if len(loadings) == 0:
        return np.nan

    suma_loadings = np.sum(loadings)
    suma_loadings_sq = suma_loadings ** 2
    suma_error = np.sum(1 - loadings**2)

    cr = suma_loadings_sq / (suma_loadings_sq + suma_error)
    return cr

def calcular_ave(loadings):
    """
    Calcula el Average Variance Extracted (AVE).
    Fórmula: AVE = Σλ² / [Σλ² + Σ(1-λ²)]
    """
    if len(loadings) == 0:
        return np.nan

    suma_loadings_sq = np.sum(loadings**2)
    suma_error = np.sum(1 - loadings**2)

    ave = suma_loadings_sq / (suma_loadings_sq + suma_error)
    return ave

def interpretar_ajuste(cfi, rmsea, tli, gfi, srmr=None, chi2_df=None):
    """Interpreta los índices de ajuste."""
    resultados = {}

    if cfi >= 0.95:
        resultados['CFI'] = 'Excelente'
    elif cfi >= 0.90:
        resultados['CFI'] = 'Bueno'
    elif cfi >= 0.80:
        resultados['CFI'] = 'Aceptable'
    else:
        resultados['CFI'] = 'Pobre'

    if rmsea <= 0.05:
        resultados['RMSEA'] = 'Excelente'
    elif rmsea <= 0.08:
        resultados['RMSEA'] = 'Aceptable'
    else:
        resultados['RMSEA'] = 'Pobre'

    if tli >= 0.95:
        resultados['TLI'] = 'Excelente'
    elif tli >= 0.90:
        resultados['TLI'] = 'Bueno'
    elif tli >= 0.80:
        resultados['TLI'] = 'Aceptable'
    else:
        resultados['TLI'] = 'Pobre'

    if gfi >= 0.95:
        resultados['GFI'] = 'Excelente'
    elif gfi >= 0.90:
        resultados['GFI'] = 'Bueno'
    elif gfi >= 0.80:
        resultados['GFI'] = 'Aceptable'
    else:
        resultados['GFI'] = 'Pobre'

    if srmr is not None:
        if srmr <= 0.05:
            resultados['SRMR'] = 'Excelente'
        elif srmr <= 0.08:
            resultados['SRMR'] = 'Aceptable'
        else:
            resultados['SRMR'] = 'Pobre'

    if chi2_df is not None:
        if chi2_df <= 2:
            resultados['Chi²/df'] = 'Excelente'
        elif chi2_df <= 3:
            resultados['Chi²/df'] = 'Bueno'
        elif chi2_df <= 5:
            resultados['Chi²/df'] = 'Aceptable'
        else:
            resultados['Chi²/df'] = 'Pobre'

    return resultados

def evaluar_normalidad(data):
    """Evalúa la normalidad multivariada de los datos."""
    resultados = {}

    if data.shape[0] <= 5000:
        shapiro_stats = []
        shapiro_pvals = []
        for col in data.columns[:10]:
            stat, pval = shapiro(data[col].dropna())
            shapiro_stats.append(stat)
            shapiro_pvals.append(pval)

        resultados['shapiro_w'] = np.mean(shapiro_stats)
        resultados['shapiro_p'] = np.mean(shapiro_pvals)
        resultados['shapiro_normal'] = np.mean(shapiro_pvals) > 0.05

    try:
        stat, pval = normaltest(data.values.flatten())
        resultados['dagostino_stat'] = stat
        resultados['dagostino_p'] = pval
        resultados['dagostino_normal'] = pval > 0.05
    except:
        resultados['dagostino_normal'] = None

    return resultados

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
print("=" * 80)
print("VALIDACIÓN DE CUESTIONARIO - ANÁLISIS DE ECUACIONES ESTRUCTURALES (SEM)")
print("=" * 80)

# Imprimir glosario al inicio
imprimir_glosario()

# Verificar que el archivo existe
if not os.path.exists(ARCHIVO_DATOS):
    print(f"\n❌ ERROR: No se encontró el archivo '{ARCHIVO_DATOS}'")
    print(f"   Asegúrese de que el archivo esté en la misma carpeta que este script.")
    exit(1)

# Cargar datos
try:
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
print(f"   • Número de participantes (N): {n_participantes}")
print(f"   • Número de ítems (k): {n_items}")
print(f"   • Nombres de los ítems: {', '.join(items[:5])}{'...' if n_items > 5 else ''}")
print(f"   • Valores faltantes: {df[items].isnull().sum().sum()}")
print(f"   • Porcentaje de datos completos: {((n_participantes * n_items - df[items].isnull().sum().sum()) / (n_participantes * n_items) * 100):.2f}%")

# Crear dataframe solo con los ítems
df_items = df[items].copy()

# ==============================================================================
# 2. ANÁLISIS DESCRIPTIVO DETALLADO
# ==============================================================================
print("\n" + "=" * 80)
print("2. ANÁLISIS DESCRIPTIVO DETALLADO")
print("=" * 80)

desc_stats = df_items.describe().T
desc_stats['Asimetría (Skewness)'] = df_items.skew()
desc_stats['Curtosis'] = df_items.kurtosis()
desc_stats['Mediana'] = df_items.median()
desc_stats['Moda'] = df_items.mode().iloc[0] if not df_items.mode().empty else np.nan
desc_stats['Rango'] = desc_stats['max'] - desc_stats['min']
desc_stats['IQR'] = df_items.quantile(0.75) - df_items.quantile(0.25)
desc_stats['Coef. Variación (%)'] = (desc_stats['std'] / desc_stats['mean'] * 100)

print("\n📋 ESTADÍSTICAS DESCRIPTIVAS POR ÍTEM:")
print("-" * 80)
print(desc_stats[['mean', 'std', 'min', 'max', 'Mediana', 'Asimetría (Skewness)', 'Curtosis', 'Rango', 'IQR']].round(3).to_string())

print("\n📊 INTERPRETACIÓN DE ASIMETRÍA:")
print("   • |Skewness| < 0.5: Distribución aproximadamente simétrica")
print("   • 0.5 ≤ |Skewness| < 1: Distribución moderadamente asimétrica")
print("   • |Skewness| ≥ 1: Distribución altamente asimétrica")

print("\n📊 INTERPRETACIÓN DE CURTOSIS:")
print("   • Curtosis ≈ 3 (o Exceso ≈ 0): Distribución normal (mesocúrtica)")
print("   • Curtosis > 3 (Exceso > 0): Distribución leptocúrtica (colas pesadas)")
print("   • Curtosis < 3 (Exceso < 0): Distribución platicúrtica (colas ligeras)")

# Guardar estadísticas descriptivas
os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)
desc_stats.to_csv(os.path.join(DIRECTORIO_SALIDA, 'estadisticas_descriptivas.csv'))
print(f"\n✓ Estadísticas descriptivas guardadas en: {DIRECTORIO_SALIDA}estadisticas_descriptivas.csv")

# ==============================================================================
# 2.1 ANÁLISIS DE NORMALIDAD
# ==============================================================================
print("\n" + "-" * 80)
print("2.1 ANÁLISIS DE NORMALIDAD MULTIVARIADA")
print("-" * 80)

print("\n📋 EXPLICACIÓN:")
print("   La normalidad es importante para el estimador de Máxima Verosimilitud (ML)")
print("   utilizado en el SEM. Desviaciones moderadas son tolerables con N > 200.")

norm_stats = evaluar_normalidad(df_items)

print("\n🔹 Test de Normalidad:")
if 'shapiro_w' in norm_stats:
    print(f"   • Test de Shapiro-Wilk (promedio de 10 ítems):")
    print(f"     W = {norm_stats['shapiro_w']:.4f}, p = {norm_stats['shapiro_p']:.4f}")
    print(f"     Interpretación: {'Datos aproximadamente normales' if norm_stats['shapiro_normal'] else 'Datos NO normales (p<0.05)'}")

if norm_stats['dagostino_normal'] is not None:
    print(f"   • Test de D'Agostino-Pearson:")
    print(f"     Estadístico = {norm_stats['dagostino_stat']:.2f}, p = {norm_stats['dagostino_p']:.2e}")
    print(f"     Interpretación: {'Datos aproximadamente normales' if norm_stats['dagostino_normal'] else 'Datos NO normales (p<0.05)'}")

print("\n📋 NOTA SOBRE NORMALIDAD Y SEM:")
print("   • El ML (Maximum Likelihood) asume normalidad multivariada")
print("   • Si los datos no son normales, considere usar el estimador MLR (Robusto)")
print("   • Las desviaciones moderadas de la normalidad son tolerables con N > 200")

# ==============================================================================
# 3. PRUEBAS DE ADECUACIÓN MUESTRAL
# ==============================================================================
print("\n" + "=" * 80)
print("3. PRUEBAS DE ADECUACIÓN MUESTRAL")
print("=" * 80)

print("\n📋 EXPLICACIÓN:")
print("   Estas pruebas determinan si los datos son adecuados para análisis factorial.")
print("   Un KMO bajo o un Bartlett no significativo indican que el AFE no es apropiado.")

# Verificar ratio muestra/ítems
ratio = n_participantes / n_items
print(f"\n📊 RATIO MUESTRA/ÍTEMS:")
print(f"   • Ratio calculado: {ratio:.2f}:1")
print(f"   • Recomendación mínima: 5:1 (Hair et al., 2010)")
print(f"   • Recomendación óptima: 10:1 o 20:1")

if ratio < 5:
    print(f"   ⚠️ ADVERTENCIA: Ratio muy bajo. Riesgo de sobreajuste y soluciones inestables.")
    print(f"      Se recomienda aumentar la muestra o reducir ítems.")
elif ratio < 10:
    print(f"   ℹ️ Ratio aceptable pero no óptimo. Resultados pueden ser sensibles.")
else:
    print(f"   ✓ Ratio óptimo. Tamaño muestral adecuado para el análisis.")

# Test de Esfericidad de Bartlett
print("\n" + "-" * 80)
print("3.1 TEST DE ESFERICIDAD DE BARTLETT")
print("-" * 80)

print("\n📋 EXPLICACIÓN:")
print("   El test de Bartlett evalúa si la matriz de correlaciones es una matriz")
print("   identidad (es decir, si las variables están incorreladas).")
print("   • H0: Las variables no están correlacionadas (matriz identidad)")
print("   • H1: Las variables SÍ están correlacionadas")
print("   • p < 0.05: Rechazamos H0, indicando que el AFE es apropiado")

try:
    chi_square, p_value = calculate_bartlett_sphericity(df_items)
    print(f"\n🔹 Resultados:")
    print(f"   • Chi-cuadrado (χ²): {chi_square:.3f}")
    print(f"   • Grados de libertad (df): {int(n_items * (n_items - 1) / 2)}")
    print(f"   • p-valor: {p_value:.2e}")
    print(f"   • Interpretación: {'✓ Adecuado para factorización (p<0.05)' if p_value < 0.05 else '✗ No adecuado (p≥0.05)'}")
except Exception as e:
    print(f"\n⚠️ No se pudo calcular el test de Bartlett: {e}")
    chi_square, p_value = np.nan, np.nan

# Índice KMO
print("\n" + "-" * 80)
print("3.2 ÍNDICE KMO (KAISER-MEYER-OLKIN)")
print("-" * 80)

print("\n📋 EXPLICACIÓN:")
print("   El índice KMO mide la adecuación de la muestra para análisis factorial.")
print("   Evalúa la proporción de varianza entre variables que podría ser varianza")
print("   común (compartida por los factores latentes).")
print("   • KMO > 0.90: Excelente")
print("   • KMO > 0.80: Bueno")
print("   • KMO > 0.70: Aceptable")
print("   • KMO > 0.60: Cuestionable")
print("   • KMO > 0.50: Pobre")
print("   • KMO < 0.50: Inaceptable")

try:
    kmo_all, kmo_model = calculate_kmo(df_items)
    kmo_interp = interpretar_kmo(kmo_model)
    print(f"\n🔹 Resultados:")
    print(f"   • KMO general: {kmo_model:.3f}")
    print(f"   • Interpretación: {kmo_interp}")

    # KMO por ítem
    print(f"\n   • KMO por ítem (muestra de 10 ítems):")
    for i, item in enumerate(items[:10]):
        print(f"     {item}: {kmo_all[i]:.3f}")

    if kmo_model >= 0.8:
        print(f"\n   ✓ El KMO indica que el análisis factorial es apropiado.")
    elif kmo_model >= 0.7:
        print(f"\n   ℹ️ El KMO es aceptable, pero considere revisar ítems con baja correlación.")
    else:
        print(f"\n   ⚠️ El KMO es bajo. El análisis factorial puede no ser apropiado.")

except Exception as e:
    print(f"\n⚠️ No se pudo calcular el índice KMO: {e}")
    kmo_model = np.nan
    kmo_interp = "No calculable"

# ==============================================================================
# 4. IDENTIFICACIÓN DE ÍTEMS INVERSOS
# ==============================================================================
print("\n" + "=" * 80)
print("4. IDENTIFICACIÓN Y TRATAMIENTO DE ÍTEMS INVERSOS")
print("=" * 80)

print("\n📋 EXPLICACIÓN:")
print("   Los ítems inversos (negativos) están formulados de manera que una puntuación")
print("   alta indica la ausencia del constructo medido. Es crucial identificarlos y")
print("   recodificarlos para evitar correlaciones negativas espurias.")
print("   • Correlación ítem-total negativa: Indica posible ítem inverso")
print("   • Umbral típico: r < -0.05 sugiere inversión")

item_corr = item_total_correlation(df_items)
items_inversos = item_corr[item_corr < -0.05].index.tolist()

print(f"\n🔹 Ítems con correlación negativa (posibles ítems inversos): {len(items_inversos)}")
if len(items_inversos) > 0:
    print(f"\n   Lista de ítems inversos detectados:")
    for item in items_inversos:
        print(f"      • {item}: r = {item_corr[item]:.3f}")

    # Detectar rango de respuesta
    min_val = df_items.min().min()
    max_val = df_items.max().max()
    print(f"\n   Rango de respuesta detectado: {min_val} - {max_val}")
    print(f"   Fórmula de recodificación: Nuevo valor = ({min_val} + {max_val}) - Valor original")

    # Recodificar ítems inversos
    df_recoded = df_items.copy()
    for item in items_inversos:
        df_recoded[item] = (min_val + max_val) - df_items[item]

    print(f"\n✓ {len(items_inversos)} ítems inversos recodificados correctamente.")
else:
    print(f"\n   ✓ No se detectaron ítems inversos.")
    df_recoded = df_items.copy()

# Guardar correlaciones ítem-total
item_corr.to_csv(os.path.join(DIRECTORIO_SALIDA, 'correlaciones_item_total.csv'))
print(f"\n✓ Correlaciones ítem-total guardadas")

# ==============================================================================
# 5. ANÁLISIS FACTORIAL EXPLORATORIO (AFE)
# ==============================================================================
print("\n" + "=" * 80)
print("5. ANÁLISIS FACTORIAL EXPLORATORIO (AFE)")
print("=" * 80)

print("\n📋 EXPLICACIÓN DEL AFE:")
print("   El Análisis Factorial Exploratorio (AFE) es una técnica estadística que")
print("   permite identificar la estructura subyacente de un conjunto de variables.")
print("   Objetivos:")
print("   • Reducir la dimensionalidad de los datos")
print("   • Identificar factores latentes no observables")
print("   • Agrupar ítems que miden el mismo constructo")
print("\n   Método de extracción: Máxima Verosimilitud (ML)")
print("   Rotación: Promax (oblicua, permite correlación entre factores)")

# Determinar número de factores mediante autovalores
fa = FactorAnalyzer(rotation=None, n_factors=df_recoded.shape[1])
fa.fit(df_recoded)
eigenvalues, _ = fa.get_eigenvalues()

# Criterio de Kaiser
n_factors_kaiser = sum(eigenvalues > 1)
print(f"\n" + "-" * 80)
print("5.1 CRITERIO DE KAISER (AUTVALORES > 1)")
print("-" * 80)
print("\n📋 EXPLICACIÓN:")
print("   El criterio de Kaiser conserva solo los factores con autovalores > 1.")
print("   Un autovalor representa la varianza explicada por cada factor.")
print("   Un factor con autovalor < 1 explica menos varianza que un solo ítem.")
print(f"\n🔹 Resultados:")
print(f"   • Número de factores según Kaiser: {n_factors_kaiser}")
print(f"   • Autovalores: {[f'{ev:.3f}' for ev in eigenvalues[:n_factors_kaiser+2]]}")

# Determinar número de factores a extraer
if NUM_FACTORES_AFE is None:
    diffs = np.diff(eigenvalues)
    elbow_idx = np.where(diffs > -0.3)[0]
    if len(elbow_idx) > 0:
        n_factors_scree = elbow_idx[-1] + 1
    else:
        n_factors_scree = n_factors_kaiser

    n_factors_aft = min(n_factors_kaiser, n_factors_scree, 10)
    n_factors_aft = max(n_factors_aft, 1)
else:
    n_factors_aft = NUM_FACTORES_AFE

print(f"\n" + "-" * 80)
print("5.2 SCREE PLOT Y CRITERIO DEL CODO")
print("-" * 80)
print("\n📋 EXPLICACIÓN:")
print("   El Scree Plot muestra los autovalores en orden decreciente.")
print("   El 'codo' es el punto donde la pendiente cambia drásticamente,")
print("   indicando que factores adicionales aportan poca varianza adicional.")
print(f"\n🔹 Factores a extraer: {n_factors_aft}")

# Scree Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=6)
axes[0].axhline(y=1, color='r', linestyle='--', label='Autovalor = 1 (Kaiser)')
axes[0].axvline(x=n_factors_aft, color='g', linestyle=':', label=f'Factores extraídos = {n_factors_aft}')
axes[0].set_xlabel('Número de Factor')
axes[0].set_ylabel('Autovalor')
axes[0].set_title('Scree Plot - Criterio del Codo')
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
print(f"\n" + "-" * 80)
print("5.3 EXTRACCIÓN DE FACTORES CON ROTACIÓN PROMAX")
print("-" * 80)
print("\n📋 EXPLICACIÓN:")
print("   La rotación Promax es una rotación oblicua que permite que los factores")
print("   estén correlacionados entre sí, lo cual es más realista en ciencias sociales.")

if n_factors_aft > 1:
    fa_rot = FactorAnalyzer(n_factors=n_factors_aft, rotation='promax', method='ml')
else:
    fa_rot = FactorAnalyzer(n_factors=n_factors_aft, rotation=None, method='ml')

fa_rot.fit(df_recoded)

# Varianza explicada
variance_rot = fa_rot.get_factor_variance()
print(f"\n🔹 Varianza explicada por el modelo:")
print(f"   • Por factor: {[f'{v:.2%}' for v in variance_rot[1]]}")
print(f"   • Varianza acumulada: {variance_rot[2][-1]:.2%}")

if variance_rot[2][-1] < 0.50:
    print(f"   ⚠️ La varianza explicada es baja (<50%). Considere revisar los ítems.")
elif variance_rot[2][-1] < 0.60:
    print(f"   ℹ️ La varianza explicada es moderada (50-60%). Aceptable pero mejorable.")
else:
    print(f"   ✓ La varianza explicada es buena (>60%).")

# Matriz de cargas factoriales
factor_names = [f'Factor{i+1}' for i in range(n_factors_aft)]
loadings_df = pd.DataFrame(
    fa_rot.loadings_,
    columns=factor_names,
    index=df_recoded.columns
)

# Calcular comunalidades
comunalidades = pd.DataFrame({
    'Item': df_recoded.columns,
    'Comunalidad (h²)': fa_rot.get_communalities()
})
comunalidades = comunalidades.sort_values('Comunalidad (h²)', ascending=False)

print(f"\n" + "-" * 80)
print("5.4 COMUNALIDADES (h²)")
print("-" * 80)
print("\n📋 EXPLICACIÓN:")
print("   La comunalidad (h²) representa la proporción de varianza de cada ítem")
print("   que es explicada por los factores extraídos.")
print("   • h² > 0.50: El ítem está bien representado por los factores")
print("   • h² < 0.30: El ítem puede no pertenecer al constructo medido")

print(f"\n🔹 Comunalidades por ítem:")
print(comunalidades.to_string(index=False))

items_baja_comunalidad = comunalidades[comunalidades['Comunalidad (h²)'] < 0.30]['Item'].tolist()
if items_baja_comunalidad:
    print(f"\n   ⚠️ Ítems con comunalidad < 0.30 (revisar): {items_baja_comunalidad}")

# Guardar cargas factoriales
loadings_df.to_csv(os.path.join(DIRECTORIO_SALIDA, 'cargas_factoriales_afe.csv'))
comunalidades.to_csv(os.path.join(DIRECTORIO_SALIDA, 'comunalidades.csv'), index=False)
print(f"\n✓ Cargas factoriales y comunalidades guardadas")

# Asignar ítems a factores
asignacion = asignar_items_a_factores(loadings_df, umbral=0.3)

print(f"\n" + "-" * 80)
print("5.5 ASIGNACIÓN DE ÍTEMS A FACTORES")
print("-" * 80)
print("\n📋 EXPLICACIÓN:")
print("   Se asigna cada ítem al factor donde tiene la mayor carga factorial.")
print("   Umbral de asignación: |λ| ≥ 0.30 (criterio mínimo aceptable)")

print(f"\n🔹 Asignación de ítems a factores (carga ≥ 0.3):")
for factor, items_factor in asignacion.items():
    print(f"\n   {factor} ({len(items_factor)} ítems):")
    if len(items_factor) > 0:
        for item in items_factor[:10]:
            carga = loadings_df.loc[item, factor]
            print(f"      • {item}: λ = {carga:.3f}")
        if len(items_factor) > 10:
            print(f"      ... y {len(items_factor) - 10} ítems más")
    else:
        print(f"      (Sin ítems asignados)")

# Identificar ítems problemáticos (cargas cruzadas)
print(f"\n" + "-" * 80)
print("5.6 IDENTIFICACIÓN DE ÍTEMS CON CARGAS CRUZADAS")
print("-" * 80)
print("\n📋 EXPLICACIÓN:")
print("   Los ítems con cargas cruzadas cargan significativamente (>0.30) en más")
print("   de un factor, lo que dificulta la interpretación.")

items_cruzados = []
for item in loadings_df.index:
    cargas_altas = (loadings_df.loc[item].abs() >= 0.30).sum()
    if cargas_altas > 1:
        items_cruzados.append(item)
        cargas = loadings_df.loc[item].abs().sort_values(ascending=False)
        print(f"\n   ⚠️ {item}:")
        print(f"      Carga 1: {cargas.index[0]} = {cargas.iloc[0]:.3f}")
        print(f"      Carga 2: {cargas.index[1]} = {cargas.iloc[1]:.3f}")

if not items_cruzados:
    print(f"\n   ✓ No se detectaron ítems con cargas cruzadas significativas.")

# ==============================================================================
# 6. ANÁLISIS DE ECUACIONES ESTRUCTURALES (SEM)
# ==============================================================================
print("\n" + "=" * 80)
print("6. ANÁLISIS DE ECUACIONES ESTRUCTURALES (SEM)")
print("=" * 80)

print("\n📋 EXPLICACIÓN DEL SEM:")
print("   Las Ecuaciones Estructurales (SEM) permiten probar hipótesis específicas")
print("   sobre la estructura factorial del cuestionario. A diferencia del AFE,")
print("   el SEM es confirmatorio: se especifica a priori qué ítems pertenecen a")
print("   qué factores y se evalúa el ajuste del modelo propuesto.")

# Modelo 1: Unidimensional
print("\n" + "-" * 80)
print("6.1 MODELO 1: UNIDIMENSIONAL (1 FACTOR GENERAL)")
print("-" * 80)
print("\n📋 EXPLICACIÓN:")
print("   Este modelo asume que todos los ítems miden un único constructo general.")
print("   Es el modelo más parsimonioso (simple) pero puede no capturar la")
print("   complejidad real si el cuestionario tiene subescalas distintas.")

modelo_1f = generar_modelo_unidimensional(items)
print(f"\n🔹 Especificación del modelo:")
print(f"   {modelo_1f}")

try:
    sem_1f = Model(modelo_1f)
    sem_1f.fit(df_recoded)
    stats_1f = semopy.calc_stats(sem_1f)

    print(f"\n🔹 ÍNDICES DE AJUSTE DEL MODELO UNIDIMENSIONAL:")
    print(f"   • CFI (Comparative Fit Index):        {stats_1f['CFI'].values[0]:.4f}")
    print(f"   • RMSEA (Root Mean Square Error):     {stats_1f['RMSEA'].values[0]:.4f}")
    print(f"   • TLI (Tucker-Lewis Index):           {stats_1f['TLI'].values[0]:.4f}")
    print(f"   • GFI (Goodness of Fit Index):        {stats_1f['GFI'].values[0]:.4f}")
    print(f"   • AIC (Akaike Information Criterion): {stats_1f['AIC'].values[0]:.2f}")

    # Interpretación
    interp_1f = interpretar_ajuste(
        stats_1f['CFI'].values[0],
        stats_1f['RMSEA'].values[0],
        stats_1f['TLI'].values[0],
        stats_1f['GFI'].values[0]
    )

    print(f"\n   INTERPRETACIÓN:")
    for indice, calificacion in interp_1f.items():
        print(f"      • {indice}: {calificacion}")

    # Verificar si el modelo es aceptable
    cfi_ok = stats_1f['CFI'].values[0] >= 0.90
    rmsea_ok = stats_1f['RMSEA'].values[0] <= 0.08
    tli_ok = stats_1f['TLI'].values[0] >= 0.90

    if cfi_ok and rmsea_ok and tli_ok:
        print(f"\n   ✓ El modelo unidimensional muestra un ajuste ACEPTABLE.")
    else:
        print(f"\n   ⚠️ El modelo unidimensional muestra un ajuste DEFICIENTE.")
        print(f"      Considere un modelo multifactorial.")

except Exception as e:
    print(f"\n   ⚠️ Error al ajustar modelo unidimensional: {e}")
    stats_1f = None
    interp_1f = None

# Modelo 2: Multifactorial (si hay más de 1 factor)
stats_mf = None
interp_mf = None

if n_factors_aft > 1:
    print(f"\n" + "-" * 80)
    print(f"6.2 MODELO 2: MULTIFACTORIAL ({n_factors_aft} FACTORES)")
    print("-" * 80)
    print(f"\n📋 EXPLICACIÓN:")
    print(f"   Este modelo asume que los ítems agrupan en {n_factors_aft} factores")
    print(f"   correlacionados, según los resultados del AFE.")

    # Filtrar factores con al menos 3 ítems
    asignacion_valida = {k: v for k, v in asignacion.items() if len(v) >= 3}

    if len(asignacion_valida) > 1:
        modelo_mf = generar_modelo_multifactorial(asignacion_valida)
        print(f"\n🔹 Especificación del modelo:")
        print(f"   {modelo_mf}")

        try:
            sem_mf = Model(modelo_mf)
            sem_mf.fit(df_recoded)
            stats_mf = semopy.calc_stats(sem_mf)

            print(f"\n🔹 ÍNDICES DE AJUSTE DEL MODELO MULTIFACTORIAL:")
            print(f"   • CFI (Comparative Fit Index):        {stats_mf['CFI'].values[0]:.4f}")
            print(f"   • RMSEA (Root Mean Square Error):     {stats_mf['RMSEA'].values[0]:.4f}")
            print(f"   • TLI (Tucker-Lewis Index):           {stats_mf['TLI'].values[0]:.4f}")
            print(f"   • GFI (Goodness of Fit Index):        {stats_mf['GFI'].values[0]:.4f}")
            print(f"   • AIC (Akaike Information Criterion): {stats_mf['AIC'].values[0]:.2f}")

            # Interpretación
            interp_mf = interpretar_ajuste(
                stats_mf['CFI'].values[0],
                stats_mf['RMSEA'].values[0],
                stats_mf['TLI'].values[0],
                stats_mf['GFI'].values[0]
            )

            print(f"\n   INTERPRETACIÓN:")
            for indice, calificacion in interp_mf.items():
                print(f"      • {indice}: {calificacion}")

            # Verificar si el modelo es aceptable
            cfi_ok = stats_mf['CFI'].values[0] >= 0.90
            rmsea_ok = stats_mf['RMSEA'].values[0] <= 0.08
            tli_ok = stats_mf['TLI'].values[0] >= 0.90

            if cfi_ok and rmsea_ok and tli_ok:
                print(f"\n   ✓ El modelo multifactorial muestra un ajuste ACEPTABLE.")
            else:
                print(f"\n   ⚠️ El modelo multifactorial muestra un ajuste DEFICIENTE.")

        except Exception as e:
            print(f"\n   ⚠️ Error al ajustar modelo multifactorial: {e}")
            stats_mf = None
    else:
        print(f"\n   ℹ️ No hay suficientes factores con ≥3 ítems para el modelo multifactorial")

# ==============================================================================
# 7. ÍNDICES DE AJUSTE DEL MODELO - COMPARACIÓN DETALLADA
# ==============================================================================
print("\n" + "=" * 80)
print("7. COMPARACIÓN DETALLADA DE ÍNDICES DE AJUSTE")
print("=" * 80)

print("\n📋 EXPLICACIÓN DE LOS ÍNDICES DE AJUSTE:")
print("-" * 80)
print("""
Los índices de ajuste evalúan qué tan bien el modelo propuesto reproduce las
matrices de covarianza observadas en los datos. Se clasifican en:

1. ÍNDICES ABSOLUTOS: Evalúan el ajuste global sin comparar con otros modelos
   • χ² (Chi-cuadrado): Prueba estadística exacta. p > 0.05 indica buen ajuste.
     Limitación: Muy sensible al tamaño muestral (con N grande, casi siempre p < 0.05)

   • RMSEA: Error de aproximación. Insensible al tamaño muestral.
     <0.05 Excelente, <0.08 Aceptable, >0.10 Inaceptable

   • SRMR: Residual estandarizado. <0.05 Excelente, <0.08 Aceptable

   • GFI: Bondad de ajuste absoluta. Similar a R² en regresión.
     >0.95 Excelente, >0.90 Aceptable

2. ÍNDICES INCREMENTALES: Comparan el modelo con un modelo nulo (independencia)
   • CFI: Índice de ajuste comparativo. Más robusto que χ².
     >0.95 Excelente, >0.90 Aceptable, penaliza complejidad

   • TLI: Índice de Tucker-Lewis. Similar a CFI pero penaliza más la complejidad.
     >0.95 Excelente, >0.90 Aceptable

3. ÍNDICES DE PARSIMONIA: Penalizan la complejidad del modelo
   • AIC: Criterio de información de Akaike. MENOR es MEJOR.
     Útil para comparar modelos no anidados en la misma muestra.

RECOMENDACIÓN PRÁCTICA:
Para aceptar un modelo, se recomienda que AL MENOS CUMPLA:
• CFI ≥ 0.90 (idealmente ≥ 0.95)
• RMSEA ≤ 0.08 (idealmente ≤ 0.05)
• TLI ≥ 0.90 (idealmente ≥ 0.95)
• SRMR ≤ 0.08 (idealmente ≤ 0.05)
""")

# Comparación de modelos
print("\n" + "-" * 80)
print("7.1 TABLA COMPARATIVA DE MODELOS")
print("-" * 80)

print("\n📊 COMPARACIÓN DE MODELOS:\n")
print(f"{'Modelo':<25} {'CFI':>8} {'RMSEA':>8} {'TLI':>8} {'GFI':>8} {'AIC':>12}")
print("-" * 75)

modelos_stats = {}

if stats_1f is not None:
    cfi = stats_1f['CFI'].values[0]
    rmsea = stats_1f['RMSEA'].values[0]
    tli = stats_1f['TLI'].values[0]
    gfi = stats_1f['GFI'].values[0]
    aic = stats_1f['AIC'].values[0]
    print(f"{'Unidimensional':<25} {cfi:8.4f} {rmsea:8.4f} {tli:8.4f} {gfi:8.4f} {aic:12.2f}")
    modelos_stats['Unidimensional'] = {'CFI': cfi, 'RMSEA': rmsea, 'TLI': tli, 'GFI': gfi, 'AIC': aic}

if stats_mf is not None:
    cfi = stats_mf['CFI'].values[0]
    rmsea = stats_mf['RMSEA'].values[0]
    tli = stats_mf['TLI'].values[0]
    gfi = stats_mf['GFI'].values[0]
    aic = stats_mf['AIC'].values[0]
    nombres = f"{len(asignacion_valida)} Factores"
    print(f"{nombres:<25} {cfi:8.4f} {rmsea:8.4f} {tli:8.4f} {gfi:8.4f} {aic:12.2f}")
    modelos_stats[nombres] = {'CFI': cfi, 'RMSEA': rmsea, 'TLI': tli, 'GFI': gfi, 'AIC': aic}

# ==============================================================================
# 8. CONFIABILIDAD (ALFA DE CRONBACH) DETALLADA
# ==============================================================================
print("\n" + "=" * 80)
print("8. CONFIABILIDAD (ALFA DE CRONBACH) DETALLADA")
print("=" * 80)

print("\n📋 EXPLICACIÓN:")
print("   El Alfa de Cronbach mide la consistencia interna del cuestionario.")
print("   Indica el grado en que los ítems miden el mismo constructo.")
print("   Valores de referencia:")
print("   • α ≥ 0.90: Excelente (ideal para decisiones individuales)")
print("   • α ≥ 0.80: Bueno (aceptable para investigación)")
print("   • α ≥ 0.70: Aceptable (mínimo recomendado)")
print("   • α ≥ 0.60: Cuestionable (bajo, revisar ítems)")
print("   • α < 0.60: Inaceptable (no usar)")

# Alfa por factor (si hay múltiples factores)
if n_factors_aft > 1 and len(asignacion_valida) > 0:
    print("\n" + "-" * 80)
    print("8.1 CONFIABILIDAD POR FACTOR")
    print("-" * 80)

    alphas_factores = {}
    for factor, items_factor in asignacion_valida.items():
        if len(items_factor) >= 2:
            alpha, alphas_elim = cronbach_alpha_detallado(df_recoded[items_factor])
            alphas_factores[factor] = alpha
            interp = interpretar_alpha(alpha)
            print(f"\n🔹 {factor} ({len(items_factor)} ítems):")
            print(f"   • Alfa de Cronbach: α = {alpha:.4f} - {interp}")

            # Mostrar ítems que si se eliminan mejoran el alfa
            if alphas_elim:
                mejora = {k: v for k, v in alphas_elim.items() if v > alpha}
                if mejora:
                    print(f"   • Ítems que si se eliminan MEJORAN el alfa:")
                    for item, alpha_sin in sorted(mejora.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"      - {item}: α sería {alpha_sin:.4f} (+{alpha_sin-alpha:.4f})")

# Alfa total
print("\n" + "-" * 80)
print("8.2 CONFIABILIDAD TOTAL DEL CUESTIONARIO")
print("-" * 80)

alpha_total, alphas_elim_total = cronbach_alpha_detallado(df_recoded)
print(f"\n🔹 Alfa de Cronbach total ({n_items} ítems):")
print(f"   • α = {alpha_total:.4f} - {interpretar_alpha(alpha_total)}")

# Análisis de ítems que afectan el alfa
if alphas_elim_total:
    print(f"\n   • Análisis 'si se elimina' cada ítem:")

    # Ítems que mejoran el alfa si se eliminan
    mejora_alfa = {k: v for k, v in alphas_elim_total.items() if v > alpha_total}
    if mejora_alfa:
        print(f"\n     Ítems que DEBERÍAN CONSIDERARSE PARA ELIMINACIÓN:")
        for item, alpha_sin in sorted(mejora_alfa.items(), key=lambda x: x[1], reverse=True):
            diferencia = alpha_sin - alpha_total
            print(f"      - {item}: α = {alpha_sin:.4f} (mejora +{diferencia:.4f})")

    # Ítems esenciales (el alfa baja mucho si se eliminan)
    esenciales = {k: v for k, v in alphas_elim_total.items() if v < alpha_total - 0.05}
    if esenciales:
        print(f"\n     Ítems ESENCIALES (el alfa baja significativamente sin ellos):")
        for item, alpha_sin in sorted(esenciales.items(), key=lambda x: x[1])[:5]:
            diferencia = alpha_total - alpha_sin
            print(f"      - {item}: α = {alpha_sin:.4f} (baja -{diferencia:.4f})")

# ==============================================================================
# 9. VISUALIZACIONES ADICIONALES
# ==============================================================================
print("\n" + "=" * 80)
print("9. GENERACIÓN DE VISUALIZACIONES")
print("=" * 80)

# Matriz de correlaciones
fig, ax = plt.subplots(figsize=(min(14, n_items*0.3), min(12, n_items*0.25)))
corr_matrix = df_recoded.corr()
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
            square=True, xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'Correlación'}, ax=ax)
ax.set_title('Matriz de Correlaciones entre Ítems', fontsize=14)
ruta_corr = guardar_grafico(fig, '02_correlation_matrix.png')
print(f"\n✓ Matriz de correlaciones guardada: {ruta_corr}")

# Correlaciones ítem-total
fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_items*0.15)))

# Histograma
axes[0].hist(item_corr.values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(x=0, color='r', linestyle='--', label='Correlación = 0')
axes[0].axvline(x=item_corr.mean(), color='g', linestyle='-', label=f'Media = {item_corr.mean():.3f}')
axes[0].set_xlabel('Correlación ítem-total')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de Correlaciones Ítem-Total')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Barras horizontales
item_corr_sorted = item_corr.sort_values()
colors = ['red' if x < 0 else 'green' for x in item_corr_sorted.values]
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
    axes[0, 0].axhline(y=0.90, color='r', linestyle='--', label='Umbral buen ajuste (0.90)')
    axes[0, 0].set_ylabel('CFI')
    axes[0, 0].set_title('Comparative Fit Index')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)

    # TLI
    axes[0, 1].bar(nombres, tlis, color='forestgreen')
    axes[0, 1].axhline(y=0.90, color='r', linestyle='--', label='Umbral buen ajuste (0.90)')
    axes[0, 1].set_ylabel('TLI')
    axes[0, 1].set_title('Tucker-Lewis Index')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    # RMSEA
    axes[1, 0].bar(nombres, rmseas, color='coral')
    axes[1, 0].axhline(y=0.05, color='g', linestyle='--', label='Buen ajuste (<0.05)')
    axes[1, 0].axhline(y=0.08, color='orange', linestyle='--', label='Aceptable (<0.08)')
    axes[1, 0].set_ylabel('RMSEA')
    axes[1, 0].set_title('Root Mean Square Error of Approximation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # GFI
    axes[1, 1].bar(nombres, gfis, color='mediumpurple')
    axes[1, 1].axhline(y=0.90, color='r', linestyle='--', label='Umbral buen ajuste (0.90)')
    axes[1, 1].set_ylabel('GFI')
    axes[1, 1].set_title('Goodness of Fit Index')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    ruta_comp = guardar_grafico(fig, '04_model_comparison.png')
    print(f"✓ Comparación de modelos guardada: {ruta_comp}")

# ==============================================================================
# 10. CONCLUSIÓN COMPARATIVA Y RECOMENDACIÓN FINAL
# ==============================================================================
print("\n" + "=" * 80)
print("10. CONCLUSIÓN COMPARATIVA Y RECOMENDACIÓN FINAL")
print("=" * 80)

print("\n" + "📊 ANÁLISIS COMPARATIVO DE MODELOS".center(80))
print("-" * 80)

# Determinar mejor modelo
if len(modelos_stats) > 1:
    print("\n📋 COMPARACIÓN ESTADÍSTICA:")
    print("\n   Criterios de evaluación:")
    print("   1. AIC (Akaike Information Criterion): MENOR es MEJOR")
    print("      Penaliza la complejidad del modelo.")
    print("   2. CFI (Comparative Fit Index): MAYOR es MEJOR")
    print("      Debe ser ≥ 0.90 para ajuste aceptable, ≥ 0.95 para excelente.")
    print("   3. RMSEA: MENOR es MEJOR")
    print("      Debe ser ≤ 0.08 para ajuste aceptable, ≤ 0.05 para excelente.")
    print("   4. Principio de Parsimonia:")
    print("      Entre modelos con ajuste similar, preferir el más simple.")

    # Encontrar mejor modelo según AIC
    mejor_modelo_aic = min(modelos_stats.items(), key=lambda x: x[1]['AIC'])
    nombre_mejor_aic = mejor_modelo_aic[0]

    # Encontrar mejor modelo según CFI
    mejor_modelo_cfi = max(modelos_stats.items(), key=lambda x: x[1]['CFI'])
    nombre_mejor_cfi = mejor_modelo_cfi[0]

    print(f"\n📈 RESULTADOS:")
    print(f"   • Mejor modelo según AIC: {nombre_mejor_aic} (AIC = {mejor_modelo_aic[1]['AIC']:.2f})")
    print(f"   • Mejor modelo según CFI: {nombre_mejor_cfi} (CFI = {mejor_modelo_cfi[1]['CFI']:.4f})")

    # Análisis de diferencia de AIC
    aics = [stats['AIC'] for stats in modelos_stats.values()]
    delta_aic = max(aics) - min(aics)

    print(f"\n📊 Diferencia de AIC entre modelos: {delta_aic:.2f}")
    if delta_aic < 2:
        print("   • La diferencia es < 2: Ambos modelos son prácticamente equivalentes.")
        print("   • Se recomienda elegir el modelo más parsimonioso (simple).")
    elif delta_aic < 7:
        print("   • La diferencia es 2-7: Hay evidencia a favor del mejor modelo.")
    else:
        print("   • La diferencia es > 7: Fuerte evidencia a favor del mejor modelo.")

    # Determinar modelo recomendado
    if nombre_mejor_aic == nombre_mejor_cfi:
        modelo_recomendado = nombre_mejor_aic
        print(f"\n✅ MODELO RECOMENDADO: {modelo_recomendado}")
        print(f"   Justificación: Es el mejor según AIC y CFI simultáneamente.")
    else:
        # Comparar calidad de ajuste
        aic_model = modelos_stats[nombre_mejor_aic]
        cfi_model = modelos_stats[nombre_mejor_cfi]

        # Si el modelo con mejor CFI tiene AIC razonablemente cercano
        if cfi_model['CFI'] - aic_model['CFI'] > 0.05:
            modelo_recomendado = nombre_mejor_cfi
            print(f"\n✅ MODELO RECOMENDADO: {modelo_recomendado}")
            print(f"   Justificación: Aunque {nombre_mejor_aic} tiene menor AIC,")
            print(f"   {nombre_mejor_cfi} muestra un ajuste sustancialmente mejor (CFI diferencia > 0.05).")
        else:
            modelo_recomendado = nombre_mejor_aic
            print(f"\n✅ MODELO RECOMENDADO: {modelo_recomendado}")
            print(f"   Justificación: Tiene el menor AIC y la diferencia de CFI")
            print(f"   con {nombre_mejor_cfi} no es sustancial (< 0.05).")

    # Estadísticas del modelo recomendado
    stats_recomendado = modelos_stats[modelo_recomendado]
    print(f"\n📋 ESTADÍSTICAS DEL MODELO RECOMENDADO:")
    print(f"   • CFI:  {stats_recomendado['CFI']:.4f} ({'Excelente' if stats_recomendado['CFI'] >= 0.95 else 'Bueno' if stats_recomendado['CFI'] >= 0.90 else 'Aceptable'})")
    print(f"   • RMSEA: {stats_recomendado['RMSEA']:.4f} ({'Excelente' if stats_recomendado['RMSEA'] <= 0.05 else 'Aceptable' if stats_recomendado['RMSEA'] <= 0.08 else 'Pobre'})")
    print(f"   • TLI:  {stats_recomendado['TLI']:.4f} ({'Excelente' if stats_recomendado['TLI'] >= 0.95 else 'Bueno' if stats_recomendado['TLI'] >= 0.90 else 'Aceptable'})")
    print(f"   • GFI:  {stats_recomendado['GFI']:.4f} ({'Excelente' if stats_recomendado['GFI'] >= 0.95 else 'Bueno' if stats_recomendado['GFI'] >= 0.90 else 'Aceptable'})")
    print(f"   • AIC:  {stats_recomendado['AIC']:.2f}")

else:
    # Solo hay un modelo
    if len(modelos_stats) == 1:
        modelo_recomendado = list(modelos_stats.keys())[0]
        stats_recomendado = modelos_stats[modelo_recomendado]

        print(f"\n📋 ÚNICO MODELO AJUSTADO: {modelo_recomendado}")
        print(f"\n📈 ESTADÍSTICAS:")
        print(f"   • CFI:  {stats_recomendado['CFI']:.4f}")
        print(f"   • RMSEA: {stats_recomendado['RMSEA']:.4f}")
        print(f"   • TLI:  {stats_recomendado['TLI']:.4f}")
        print(f"   • GFI:  {stats_recomendado['GFI']:.4f}")

        # Evaluar si es aceptable
        if stats_recomendado['CFI'] >= 0.90 and stats_recomendado['RMSEA'] <= 0.08:
            print(f"\n✅ El modelo muestra un ajuste ACEPTABLE.")
        else:
            print(f"\n⚠️ El modelo muestra un ajuste DEFICIENTE.")
            print(f"   Se recomienda revisar la estructura factorial o considerar")
            print(f"   modelos alternativos (ej. bifactor, ESEM).")
    else:
        print(f"\n⚠️ No se pudo ajustar ningún modelo.")
        modelo_recomendado = None

# ==============================================================================
# RESUMEN EJECUTIVO FINAL
# ==============================================================================
print("\n" + "=" * 80)
print("RESUMEN EJECUTIVO DE LA VALIDACIÓN")
print("=" * 80)

resumen = f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    RESUMEN DE LA VALIDACIÓN                                  │
│   Enrique R.P Buendia Lozada.                                               │
│   Benemérita Universidad Autónoma de Puebla, BUAP, México.                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📊 MUESTRA:                                                                 │
│     • N = {n_participantes} participantes{' ' * (50 - len(str(n_participantes)))}│
│     • {n_items} ítems en el cuestionario{' ' * (47 - len(str(n_items)))}│
│     • Ratio muestra/ítems: {ratio:.2f}{' ' * (37 - len(f'{ratio:.2f}'))}│
│                                                                              │
│  📋 ADECUACIÓN MUESTRAL:                                                     │
│     • KMO = {kmo_model:.3f} ({kmo_interp}){' ' * (42 - len(f'{kmo_model:.3f}') - len(kmo_interp))}│
│     • Factores según Kaiser: {n_factors_kaiser}{' ' * (37 - len(str(n_factors_kaiser)))}│
│                                                                              │
│  🔧 RECODIFICACIÓN:                                                          │
│     • {len(items_inversos)} ítems inversos identificados{' ' * (39 - len(str(len(items_inversos))))}│
│                                                                              │
│  📈 ANÁLISIS FACTORIAL:                                                      │
│     • {n_factors_aft} factores extraídos{' ' * (46 - len(str(n_factors_aft)))}│
│     • Varianza explicada: {variance_rot[2][-1]:.1%}{' ' * (40 - len(f'{variance_rot[2][-1]:.1%}'))}│
│                                                                              │
│  ✅ CONFIABILIDAD:                                                           │
│     • Alfa de Cronbach total: {alpha_total:.3f} ({interpretar_alpha(alpha_total)}){' ' * (27 - len(f'{alpha_total:.3f}') - len(interpretar_alpha(alpha_total)))}│
│                                                                              │
│  📐 MODELO RECOMENDADO: {modelo_recomendado if modelo_recomendado else 'Ninguno'}{' ' * (48 - len(str(modelo_recomendado)))}│
"""

if modelo_recomendado and modelo_recomendado in modelos_stats:
    stats = modelos_stats[modelo_recomendado]
    resumen += f"""│                                                                              │
│     • CFI = {stats['CFI']:.3f}                                                    │
│     • RMSEA = {stats['RMSEA']:.3f}                                                │
│     • TLI = {stats['TLI']:.3f}                                                    │
│     • GFI = {stats['GFI']:.3f}                                                    │
│     • AIC = {stats['AIC']:.2f}                                                  │
"""

resumen += """│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""

print(resumen)

# Guardar resumen
with open(os.path.join(DIRECTORIO_SALIDA, 'resumen_validacion.txt'), 'w', encoding='utf-8') as f:
    f.write(resumen)
print(f"\n✓ Resumen guardado en: {DIRECTORIO_SALIDA}resumen_validacion.txt")

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"\nTodos los resultados se han guardado en: {os.path.abspath(DIRECTORIO_SALIDA)}")
print("\nArchivos generados:")
for archivo in os.listdir(DIRECTORIO_SALIDA):
    print(f"  • {archivo}")
print("=" * 80)
input("Presiona Enter para continuar...")



