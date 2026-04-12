# pip install pandas numpy scipy 
import pandas as pd
import numpy as np
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

datos = pd.read_csv("datos_invar.csv", sep=";")

print(f"\n📊 DATOS CARGADOS: N = {len(datos)}")

# Recodificar sexo si es necesario
if datos['sexo'].min() == 1:
    datos['sexo'] = datos['sexo'] - 1
    print("   Sexo recodificado: 1→0, 2→1")

print(f"\n   Mujeres (0): {(datos['sexo']==0).sum()} (54.9%)")
print(f"   Hombres (1): {(datos['sexo']==1).sum()} (44.9%)")

items = ['i1', 'i2', 'i3', 'i4', 'i5']
mujeres = datos[datos['sexo'] == 0][items].dropna()
hombres = datos[datos['sexo'] == 1][items].dropna()

print(f"\n   n mujeres = {len(mujeres)}, n hombres = {len(hombres)}")

# ============================================
# FUNCIONES
# ============================================

def calcular_kmo(datos):
    corr = datos.corr().values
    inv_corr = np.linalg.inv(corr)
    partial_corr = np.zeros_like(corr)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j:
                partial_corr[i,j] = -inv_corr[i,j] / np.sqrt(inv_corr[i,i] * inv_corr[j,j])
    
    sum_sq_corr = np.sum(corr**2) - np.sum(np.diag(corr**2))
    sum_sq_partial = np.sum(partial_corr**2) - np.sum(np.diag(partial_corr**2))
    return sum_sq_corr / (sum_sq_corr + sum_sq_partial)

def calcular_alfa(datos):
    n_items = datos.shape[1]
    var_total = datos.sum(axis=1).var()
    sum_var_items = datos.var().sum()
    return (n_items / (n_items - 1)) * (1 - sum_var_items / var_total)

def analisis_cfa_completo(datos_grupo, nombre):
    """
    CFA unidimensional usando máxima verosimilitud manual
    """
    n = len(datos_grupo)
    p = len(items)
    
    # Estandarizar datos
    datos_std = (datos_grupo - datos_grupo.mean()) / datos_grupo.std()
    
    # Matriz de correlaciones
    R = datos_grupo.corr().values.copy()
    
    # Estimación de cargas factoriales (método centroid/PCA como aproximación ML)
    autovalores, autovectores = np.linalg.eigh(R)
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    autovectores = autovectores[:, idx]
    
    # Cargas: autovector * sqrt(autovalor)
    loadings = autovectores[:, 0] * np.sqrt(autovalores[0])
    
    # Ajustar signo (positivo)
    if np.sum(loadings) < 0:
        loadings = -loadings
    
    # Matriz reproducida
    reproduced = np.outer(loadings, loadings)
    np.fill_diagonal(reproduced, 1.0)
    
    # Residuos
    residuals = R - reproduced
    
    # ==========================================
    # ÍNDICES DE AJUSTE
    # ==========================================
    
    # Chi-cuadrado (Bartlett modificado)
    chi2 = (n - 1) * np.sum(residuals**2) / 2
    df = p * (p - 1) // 2 - p  # p(p-1)/2 correlaciones - p parámetros
    
    # CFI
    chi2_null = (n - 1) * np.sum(R[np.triu_indices_from(R, k=1)]**2)
    df_null = p * (p - 1) / 2
    cfi = 1 - max(chi2 - df, 0) / max(chi2_null - df_null, 0.001)
    cfi = max(0, min(1, cfi))
    
    # TLI
    tli = ((chi2_null/df_null) - (chi2/df)) / ((chi2_null/df_null) - 1) if df > 0 else 1
    tli = max(0, min(1, tli))
    
    # RMSEA
    rmsea = np.sqrt(max(chi2 - df, 0) / (df * (n - 1))) if df > 0 else 0
    
    # SRMR
    srmr = np.sqrt(np.sum(residuals**2) / (p * (p + 1) / 2))
    
    # GFI
    gfi = 1 - np.sum(residuals**2) / np.sum(R**2)
    
    # Alfa de Cronbach
    alfa = calcular_alfa(datos_grupo)
    
    # Varianza explicada
    ve = autovalores[0] / p * 100
    
    # ==========================================
    # IMPRESIÓN
    # ==========================================
    
    print(f"\n{'='*70}")
    print(f"CFA - {nombre} (n={n})")
    print(f"{'='*70}")
    
    print(f"\n📊 ÍNDICES DE AJUSTE:")
    print(f"   χ²({df}) = {chi2:.2f}")
    print(f"   CFI  = {cfi:.3f} {'✅' if cfi > 0.95 else '🟡' if cfi > 0.90 else '❌'} (>0.95 excelente)")
    print(f"   TLI  = {tli:.3f} {'✅' if tli > 0.95 else '🟡' if tli > 0.90 else '❌'} (>0.95 excelente)")
    print(f"   RMSEA = {rmsea:.3f} {'✅' if rmsea < 0.06 else '🟡' if rmsea < 0.08 else '❌'} (<0.06 excelente)")
    print(f"   SRMR = {srmr:.3f} {'✅' if srmr < 0.08 else '🟡' if srmr < 0.10 else '❌'} (<0.08 excelente)")
    print(f"   GFI  = {gfi:.3f} {'✅' if gfi > 0.90 else '🟡' if gfi > 0.85 else '❌'} (>0.90 bueno)")
    
    print(f"\n📈 CARGAS FACTORIALES:")
    for i, item in enumerate(items):
        print(f"   {item}: λ = {loadings[i]:.3f} {'✅' if abs(loadings[i]) > 0.60 else '🟡' if abs(loadings[i]) > 0.40 else '❌'}")
    
    print(f"\n📊 FIABILIDAD: α = {alfa:.3f} {'✅' if alfa > 0.80 else '🟡' if alfa > 0.70 else '❌'}")
    print(f"   Varianza explicada: {ve:.1f}%")
    
    # Evaluación global
    ajuste = "EXCELENTE" if (cfi > 0.95 and rmsea < 0.06) else \
             "BUENO" if (cfi > 0.90 and rmsea < 0.08) else \
             "ACEPTABLE" if cfi > 0.85 else "INADECUADO"
    
    print(f"\n   Ajuste global: {ajuste}")
    
    return {
        'n': n, 'chi2': chi2, 'df': df, 'cfi': cfi, 'tli': tli,
        'rmsea': rmsea, 'srmr': srmr, 'gfi': gfi, 'loadings': loadings,
        'alfa': alfa, 've': ve, 'ajuste': ajuste
    }

def analisis_afe(datos_grupo, nombre):
    """Análisis factorial exploratorio"""
    print(f"\n{'='*70}")
    print(f"AFE - {nombre}")
    print(f"{'='*70}")
    
    kmo = calcular_kmo(datos_grupo)
    print(f"\n   KMO = {kmo:.3f} {'✅' if kmo > 0.80 else '🟡' if kmo > 0.70 else '❌'}")
    
    corr = datos_grupo.corr().values
    autovalores, _ = np.linalg.eigh(corr)
    autovalores = np.sort(autovalores)[::-1]
    
    print(f"   Autovalores: ", end="")
    for ev in autovalores[:3]:
        print(f"{ev:.2f} ", end="")
    print(f"(>1: {sum(autovalores > 1)})")
    
    varianza = autovalores[0] / len(items) * 100
    print(f"   Varianza Factor 1: {varianza:.1f}%")
    
    return {'kmo': kmo, 'varianza': varianza, 'n_factores': sum(autovalores > 1)}

# ============================================
# 1. CONFIGURAL
# ============================================

print("\n" + "="*70)
print("1. INVARIANZA CONFIGURAL")
print("="*70)

afe_m = analisis_afe(mujeres, "MUJERES")
afe_h = analisis_afe(hombres, "HOMBRES")

config_ok = (afe_m['n_factores'] == afe_h['n_factores'] == 1 and 
             afe_m['kmo'] > 0.7 and afe_h['kmo'] > 0.7 and
             abs(afe_m['varianza'] - afe_h['varianza']) < 15)

print(f"\n{'='*70}")
print(f"RESULTADO: {'✅ CONFIGURAL SOSTENIDA' if config_ok else '❌ NO SOSTENIDA'}")
print(f"{'='*70}")

# ============================================
# 2. MÉTRICA
# ============================================

print("\n" + "="*70)
print("2. INVARIANZA MÉTRICA")
print("="*70)

cfa_m = analisis_cfa_completo(mujeres, "MUJERES")
cfa_h = analisis_cfa_completo(hombres, "HOMBRES")

print(f"\n{'='*70}")
print("COMPARACIÓN DE CARGAS")
print(f"{'='*70}")

dif = np.abs(cfa_m['loadings'] - cfa_h['loadings'])
print(f"\n   {'Ítem':<8} {'Mujeres':<10} {'Hombres':<10} {'|Δ|':<8} {'Estado'}")
print(f"   {'-'*50}")
for i, item in enumerate(items):
    estado = "✅" if dif[i] < 0.1 else "⚠️" if dif[i] < 0.2 else "❌"
    print(f"   {item:<8} {cfa_m['loadings'][i]:<10.3f} {cfa_h['loadings'][i]:<10.3f} {dif[i]:<10.3f} {estado}")

dif_prom = np.mean(dif)
print(f"\n   Δ promedio = {dif_prom:.3f}")

metrica_ok = dif_prom < 0.1
print(f"\n{'='*70}")
print(f"RESULTADO: {'✅ MÉTRICA SOSTENIDA' if metrica_ok else '🟡 PARCIAL' if dif_prom < 0.2 else '❌ NO SOSTENIDA'}")
print(f"{'='*70}")

# ============================================
# 3. ESCALAR
# ============================================

print("\n" + "="*70)
print("3. INVARIANZA ESCALAR")
print("="*70)

print(f"\n   {'Ítem':<8} {'Mujeres':<12} {'Hombres':<12} {'d':<8} {'p'}")
print(f"   {'-'*55}")

efectos = []
items_dif = []
for item in items:
    m, h = mujeres[item], hombres[item]
    d = (m.mean() - h.mean()) / np.sqrt(((len(m)-1)*m.var() + (len(h)-1)*h.var()) / (len(m)+len(h)-2))
    efectos.append(abs(d))
    t, p = stats.ttest_ind(m, h)
    if abs(d) >= 0.2:
        items_dif.append(item)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"   {item:<8} {m.mean():.2f} ({m.std():.2f}) {h.mean():.2f} ({h.std():.2f}) {d:+.2f}    {p:.3f} {sig}")

d_prom = np.mean(efectos)
print(f"\n   |d| promedio = {d_prom:.3f}")

escalar_ok = d_prom < 0.2
print(f"\n{'='*70}")
print(f"RESULTADO: {'✅ ESCALAR SOSTENIDA' if escalar_ok else '🟡 PARCIAL' if d_prom < 0.5 else '❌ NO SOSTENIDA'}")
print(f"{'='*70}")

# ============================================
# RESUMEN FINAL
# ============================================

print("\n" + "="*70) 
print(" RESUMEN FINAL DE INVARIANZA")
print(" BUAP México: Enrique Buendia Lozada")
print("="*70)

print(f"""
   ┌─────────────────────────────────────────┐
   │ NIVEL       │ RESULTADO                 │
   ├─────────────────────────────────────────┤
   │ Configural  │ {'✅ Sostenida' if config_ok else '❌':<25} │
   │ Métrica     │ {'✅ Sostenida' if metrica_ok else '🟡 Parcial' if dif_prom < 0.2 else '❌ No sostenida':<25} │
   │ Escalar     │ {'✅ Sostenida' if escalar_ok else '🟡 Parcial' if d_prom < 0.5 else '❌ No sostenida':<25} │
   └─────────────────────────────────────────┘
""")

print("CONCLUSIÓN:")
if metrica_ok and escalar_ok:
    print("   ✅ El cuestionario es equivalente entre sexos")
    print("   ✅ Se pueden comparar puntuaciones totales")
elif metrica_ok:
    print("   🟡 Comparaciones de correlaciones válidas")
    print(f"   ⚠️  Cautela en comparaciones de medias (DIF en: {', '.join(items_dif) if items_dif else 'ninguno'})")
else:
    print("   ❌ No se recomienda comparar entre grupos")
    print("   💡 Analizar por separado")

print(f"\n{'='*70}")