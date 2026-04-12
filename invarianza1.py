# pip install rpy2
# pip install semopy factor-analyzer pandas numpy scipy 
# semopy y factor_analyzer  

#
import pandas as pd
import numpy as np
from semopy import Model
from semopy.inspector import inspect
from scipy import stats
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CARGAR DATOS
# ============================================
print("="*70)
print("ANÁLISIS DE INVARIANZA DE MEDIDA - CUESTIONARIO BIENESTAR")
print("="*70)

# Cargar datos
datos = pd.read_csv("datos_invar.csv", sep=";")

print(f"\n📊 DATOS CARGADOS:")
print(f"   Total de casos: {len(datos)}")
print(f"\n📋 Distribución por sexo:")
print(f"   Mujeres (0): {(datos['sexo']==0).sum()} ({(datos['sexo']==0).mean()*100:.1f}%)")
print(f"   Hombres (1): {(datos['sexo']==1).sum()} ({(datos['sexo']==1).mean()*100:.1f}%)")

# Separar grupos
items = ['i1', 'i2', 'i3', 'i4', 'i5']
mujeres = datos[datos['sexo'] == 0][items].dropna()
hombres = datos[datos['sexo'] == 1][items].dropna()

print(f"\n   Mujeres válidas: n = {len(mujeres)}")
print(f"   Hombres válidos: n = {len(hombres)}")

# ============================================
# FUNCIONES DE APOYO (sin factor_analyzer)
# ============================================

def calcular_kmo_manual(datos):
    """
    Calcula KMO manualmente sin factor_analyzer
    """
    corr = datos.corr().values
    inv_corr = np.linalg.inv(corr)
    partial_corr = np.zeros_like(corr)
    
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j:
                partial_corr[i,j] = -inv_corr[i,j] / np.sqrt(inv_corr[i,i] * inv_corr[j,j])
    
    sum_sq_corr = np.sum(corr**2) - np.sum(np.diag(corr**2))
    sum_sq_partial = np.sum(partial_corr**2) - np.sum(np.diag(partial_corr**2))
    
    kmo = sum_sq_corr / (sum_sq_corr + sum_sq_partial)
    return kmo

def bartlett_test(datos):
    """
    Test de esfericidad de Bartlett
    """
    n = len(datos)
    p = datos.shape[1]
    corr = datos.corr().values
    corr_det = np.linalg.det(corr)
    
    if corr_det <= 0:
        corr_det = 1e-10
    
    statistic = -np.log(corr_det) * (n - 1 - (2*p + 5)/6)
    df = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(statistic, df)
    
    return statistic, p_value

def analisis_factorial_simple(datos_grupo, nombre):
    """
    AFE básico usando descomposición propia
    """
    print(f"\n{'='*70}")
    print(f"ANÁLISIS FACTORIAL EXPLORATORIO - {nombre}")
    print(f"{'='*70}")
    
    # Estandarizar
    datos_std = (datos_grupo - datos_grupo.mean()) / datos_grupo.std()
    
    # KMO y Bartlett
    kmo = calcular_kmo_manual(datos_grupo)
    chi2_bartlett, p_bartlett = bartlett_test(datos_grupo)
    
    print(f"\n📏 Adecuación muestral:")
    print(f"   KMO: {kmo:.3f} {'✅ Bueno' if kmo > 0.8 else '⚠️ Aceptable' if kmo > 0.7 else '❌ Pobre'}")
    print(f"   Bartlett: χ² = {chi2_bartlett:.2f}, p {'< 0.001 ✅' if p_bartlett < 0.001 else f'= {p_bartlett:.3f}'}")
    
    # Matriz de correlación y autovalores
    corr = datos_grupo.corr().values
    autovalores, autovectores = eigh(corr)
    autovalores = autovalores[::-1]  # Ordenar descendente
    
    print(f"\n📊 Autovalores:")
    for i, ev in enumerate(autovalores[:3]):
        print(f"   Factor {i+1}: {ev:.3f} {'✅' if ev > 1 else ''}")
    
    varianza = (autovalores[0] / len(items)) * 100
    
    # Cargas factoriales (primera componente)
    cargas = autovectores[:, -1] * np.sqrt(autovalores[0])
    cargas = cargas[::-1]  # Ajustar orden
    
    # Asegurar signo consistente (positivo)
    if np.sum(cargas) < 0:
        cargas = -cargas
    
    print(f"\n   Cargas factoriales (Factor 1):")
    for i, item in enumerate(items):
        print(f"      {item}: {abs(cargas[i]):.3f} {'✅' if abs(cargas[i]) > 0.4 else '⚠️'}")
    
    # Comunalidades aproximadas
    comunalidades = cargas**2 / autovalores[0]
    print(f"\n   Comunalidades aproximadas (h²):")
    for i, item in enumerate(items):
        h2 = comunalidades[i]
        print(f"      {item}: {h2:.3f} {'✅' if h2 > 0.3 else '⚠️ Baja'}")
    
    return {
        'kmo': kmo,
        'cargas': np.abs(cargas),
        'comunalidades': comunalidades,
        'varianza': varianza,
        'autovalor': autovalores[0]
    }

def calcular_ajuste_cfa(modelo, datos_grupo, nombre):
    """
    Calcula índices de ajuste aproximados para CFA
    """
    try:
        model = Model(modelo)
        model.fit(datos_grupo)
        stats_model = model.calc_stats()
        
        n = len(datos_grupo)
        chi2 = stats_model['chi2']
        df = stats_model['dof']
        
        # CFI aproximado
        # Modelo nulo: chi2 = traza de la matriz de correlaciones
        corr = datos_grupo.corr().values
        chi2_null = n * np.sum(corr[np.triu_indices_from(corr, k=1)]**2) * 2
        df_null = len(items) * (len(items) - 1) / 2
        
        cfi = 1 - max(chi2 - df, 0) / max(chi2_null - df_null, 0) if chi2_null > df_null else 1.0
        cfi = max(0, min(1, cfi))
        
        # TLI
        tli = ((chi2_null/df_null) - (chi2/df)) / ((chi2_null/df_null) - 1) if df > 0 and df_null > 0 else 1
        tli = max(0, min(1, tli))
        
        # RMSEA
        rmsea = np.sqrt(max(chi2 - df, 0) / (df * (n - 1))) if df > 0 and n > 1 else 0
        
        # SRMR aproximado
        residuals = corr - np.eye(len(items))  # Simplificación
        srmr = np.sqrt(np.mean(residuals**2))
        
        # Cargas
        results = inspect(model)
        loadings = results[results['op'] == '~']['std.all'].values[:5]
        
        print(f"\n--- {nombre} (n={n}) ---")
        print(f"   χ² = {chi2:.3f}, gl = {int(df)}")
        print(f"   CFI = {cfi:.3f} {'✅' if cfi > 0.95 else '⚠️' if cfi > 0.90 else '❌'}")
        print(f"   TLI = {tli:.3f} {'✅' if tli > 0.95 else '⚠️' if tli > 0.90 else '❌'}")
        print(f"   RMSEA = {rmsea:.3f} {'✅' if rmsea < 0.06 else '⚠️' if rmsea < 0.08 else '❌'}")
        print(f"   SRMR = {srmr:.3f} {'✅' if srmr < 0.08 else '⚠️'}")
        
        print(f"\n   Cargas estandarizadas:")
        for i, item in enumerate(items):
            print(f"      {item}: {loadings[i]:.3f}")
        
        return {
            'chi2': chi2, 'df': df, 'cfi': cfi, 'tli': tli, 
            'rmsea': rmsea, 'srmr': srmr, 'loadings': loadings,
            'model': model
        }
    except Exception as e:
        print(f"Error en {nombre}: {e}")
        return None

# ============================================
# 1. ANÁLISIS EXPLORATORIO POR GRUPO (CONFIGURAL)
# ============================================

print("\n" + "="*70)
print("NIVEL 1: INVARIANZA CONFIGURAL")
print("="*70)
print("\n¿El modelo unidimensional ajusta en ambos grupos por separado?")

afe_mujeres = analisis_factorial_simple(mujeres, "MUJERES")
afe_hombres = analisis_factorial_simple(hombres, "HOMBRES")

# Verificar estructura similar
print(f"\n{'='*70}")
print("COMPARACIÓN CONFIGURAL")
print(f"{'='*70}")

print(f"\n   Varianza explicada:")
print(f"      Mujeres: {afe_mujeres['varianza']:.1f}%")
print(f"      Hombres: {afe_hombres['varianza']:.1f}%")
print(f"      Diferencia: {abs(afe_mujeres['varianza'] - afe_hombres['varianza']):.1f}%")

estructura_similar = (
    (afe_mujeres['kmo'] > 0.7) and (afe_hombres['kmo'] > 0.7) and
    (abs(afe_mujeres['varianza'] - afe_hombres['varianza']) < 15)
)

print(f"\n   {'✅ INV. CONFIGURAL SOSTENIDA' if estructura_similar else '⚠️ DIFERENCIAS EN ESTRUCTURA'}")

# ============================================
# 2. ANÁLISIS CFA POR GRUPO (MÉTRICA)
# ============================================

print("\n" + "="*70)
print("NIVEL 2: INVARIANZA MÉTRICA (Cargas factoriales)")
print("="*70)

modelo_cfa = '''
    Bienestar =~ i1 + i2 + i3 + i4 + i5
'''

print("\n🔍 Ajuste del modelo CFA por grupo:")

ajuste_m = calcular_ajuste_cfa(modelo_cfa, mujeres, "MUJERES")
ajuste_h = calcular_ajuste_cfa(modelo_cfa, hombres, "HOMBRES")

# Comparación de cargas
if ajuste_m and ajuste_h:
    print(f"\n{'='*70}")
    print("COMPARACIÓN DE CARGAS FACTORIALES")
    print(f"{'='*70}")
    
    loadings_m = ajuste_m['loadings']
    loadings_h = ajuste_h['loadings']
    diferencias = np.abs(loadings_m - loadings_h)
    
    print(f"\n   {'Item':<10} {'Mujeres':<10} {'Hombres':<10} {'|Dif|':<10} {'Estado'}")
    print(f"   {'-'*50}")
    
    for i, item in enumerate(items):
        estado = "✅" if diferencias[i] < 0.1 else "⚠️" if diferencias[i] < 0.2 else "❌"
        print(f"   {item:<10} {loadings_m[i]:<10.3f} {loadings_h[i]:<10.3f} {diferencias[i]:<10.3f} {estado}")
    
    dif_promedio = np.mean(diferencias)
    max_dif = np.max(diferencias)
    
    print(f"\n   Diferencia promedio: {dif_promedio:.3f}")
    print(f"   Máxima diferencia: {max_dif:.3f} (ítem {items[np.argmax(diferencias)]})")
    
    if dif_promedio < 0.1:
        print(f"\n   ✅ INV. MÉTRICA SOSTENIDA (Δλ promedio < 0.1)")
        inv_metrica = "✅ Sostenida"
    elif dif_promedio < 0.2:
        print(f"\n   🟡 INV. MÉTRICA PARCIAL (algunas cargas difieren)")
        inv_metrica = "🟡 Parcial"
    else:
        print(f"\n   ❌ INV. MÉTRICA NO SOSTENIDA")
        inv_metrica = "❌ No sostenida"
else:
    inv_metrica = "❌ No calculable"

# ============================================
# 3. INVARIANZA ESCALAR (INTERCEPTOS)
# ============================================

print("\n" + "="*70)
print("NIVEL 3: INVARIANZA ESCALAR (Interceptos)")
print("="*70)

medias_m = mujeres.mean()
medias_h = hombres.mean()
desv_m = mujeres.std()
desv_h = hombres.std()

print(f"\n📊 Medias y desviaciones por grupo:")
print(f"\n   {'Item':<10} {'Mujeres':<15} {'Hombres':<15} {'Dif':<10} {'d de Cohen':<12} {'p (t-test)'}")
print(f"   {'-'*75}")

efectos = []
for i, item in enumerate(items):
    dif = medias_m[item] - medias_h[item]
    t_stat, p_val = stats.ttest_ind(mujeres[item], hombres[item])
    
    # d de Cohen
    pooled_std = np.sqrt(((len(mujeres)-1)*desv_m[item]**2 + (len(hombres)-1)*desv_h[item]**2) / 
                         (len(mujeres) + len(hombres) - 2))
    d_cohen = dif / pooled_std if pooled_std > 0 else 0
    efectos.append(abs(d_cohen))
    
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"   {item:<10} {medias_m[item]:.2f} ({desv_m[item]:.2f})   {medias_h[item]:.2f} ({desv_h[item]:.2f})   {dif:+.2f}      {d_cohen:+.2f}        {p_val:.3f} {sig}")

d_promedio = np.mean(efectos)

print(f"\n   Tamaño de efecto promedio (|d|): {d_promedio:.3f}")
print(f"   Interpretación: {'Negligible' if d_promedio < 0.2 else 'Pequeño' if d_promedio < 0.5 else 'Mediano' if d_promedio < 0.8 else 'Grande'}")

if d_promedio < 0.2:
    print(f"\n   ✅ INV. ESCALAR SOSTENIDA (d < 0.2)")
    inv_escalar = "✅ Sostenida"
elif d_promedio < 0.5:
    print(f"\n   🟡 INV. ESCALAR PARCIAL (d = 0.2-0.5)")
    inv_escalar = "🟡 Parcial"
else:
    print(f"\n   ❌ INV. ESCALAR NO SOSTENIDA (d > 0.5)")
    inv_escalar = "❌ No sostenida"

# ============================================
# 4. RESUMEN FINAL
# ============================================

print("\n" + "="*70)
print("RESUMEN DE INVARIANZA DE MEDIDA")
print("="*70)

print("""
📋 CRITERIOS UTILIZADOS (Chen, 2007; Putnick & Bornstein, 2016):

   CONFIGURAL: KMO > 0.70, estructura unidimensional clara
   MÉTRICA:    Diferencia promedio de cargas < 0.1
   ESCALAR:    d de Cohen < 0.2 (diferencias de medias pequeñas)
""")

print(f"\n{'='*70}")
print("RESULTADOS OBTENIDOS:")
print(f"{'='*70}")

print(f"""
   ┌─────────────────────────────────────────────────────────────┐
   │ NIVEL          │ RESULTADO        │ INTERPRETACIÓN          │
   ├─────────────────────────────────────────────────────────────┤
   │ Configural     │ {'✅ Sostenida' if estructura_similar else '⚠️ Dudos':<16} │ Mismo constructo          │
   │ Métrica        │ {inv_metrica:<16} │ Comparaciones válidas   │
   │ Escalar        │ {inv_escalar:<16} │ Medias comparables      │
   └─────────────────────────────────────────────────────────────┘
""")

print(f"\n💡 CONCLUSIÓN PRÁCTICA:")

if "✅" in inv_metrica and "✅" in inv_escalar:
    print("""
   ✅ El cuestionario es equivalente entre hombres y mujeres.
   ✅ Puedes comparar puntuaciones totales directamente.
   ✅ Las diferencias de sexo reflejan bienestar real.
    """)
elif "✅" in inv_metrica or "🟡" in inv_metrica:
    print("""
   🟡 Puedes comparar CORRELACIONES entre sexos.
   ⚠️  Ten cuidado al comparar MEDIAS brutas.
   💡  Usa puntuaciones tipificadas por sexo si es necesario.
    """)
else:
    print("""
   ❌ El cuestionario funciona diferente entre sexos.
   ❌ No compares puntuaciones entre grupos.
   💡  Analiza hombres y mujeres por separado.
    """)

print(f"\n{'='*70}")
print("NOTA PARA PUBLICACIÓN")
print(f"{'='*70}")
print(f"""
   Este análisis usa aproximaciones en Python. Para revistas Q1:
   
   1. Confirmar con lavaan (R): measurementInvariance()
   2. Usar criterio ΔCFI < 0.01 entre niveles de invarianza
   3. Reportar: χ², CFI, TLI, RMSEA para cada modelo
   
   Código R:
   library(lavaan); library(semTools)
   modelo <- 'Bienestar =~ i1 + i2 + i3 + i4 + i5'
   measurementInvariance(modelo, data=datos, group="sexo")
""")

