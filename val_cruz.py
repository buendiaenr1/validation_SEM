import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN
# ============================================
np.random.seed(42)
N_BOOTSTRAP = 1000

print("="*70)
print("VALIDACIÓN CRUZADA: BOOTSTRAP CFA + SPLIT-HALF")
print("Cuestionario de Bienestar - 5 ítems")
print("="*70)

# ============================================
# CARGAR DATOS
# ============================================
print(f"\n📂 Cargando datos...")
datos = pd.read_csv("datos.csv", sep=";")

items = ['i1', 'i2', 'i3', 'i4', 'i5']
n_total = len(datos)

print(f"   Total de casos: {n_total}")
print(f"   Variables: {items}")

# ============================================
# FUNCIONES DE APOYO
# ============================================

def calcular_cfa_manual(datos_input):
    """
    CFA unidimensional manual usando máxima verosimilitud
    Retorna cargas factoriales e índices de ajuste
    """
    n = len(datos_input)
    p = len(items)
    
    # Matriz de correlaciones
    R = datos_input.corr().values.copy()
    np.fill_diagonal(R, 1.0)
    
    # Estimación de cargas: método de factor principal (aproximación ML)
    autovalores, autovectores = np.linalg.eigh(R)
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    autovectores = autovectores[:, idx]
    
    # Cargas factoriales
    loadings = autovectores[:, 0] * np.sqrt(autovalores[0])
    
    # Ajustar signo (positivo)
    if np.sum(loadings) < 0:
        loadings = -loadings
    
    # Matriz reproducida
    reproduced = np.outer(loadings, loadings)
    reproduced = reproduced.copy()
    np.fill_diagonal(reproduced, 1.0)
    
    # Residuos
    residuals = R - reproduced
    
    # ==========================================
    # ÍNDICES DE AJUSTE
    # ==========================================
    
    # Chi-cuadrado
    chi2 = (n - 1) * np.sum(residuals**2) / 2
    df = p * (p - 1) // 2 - p
    
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
    alfa = calcular_alfa_cronbach(datos_input)
    
    return {
        'n': n,
        'chi2': chi2,
        'df': df,
        'cfi': cfi,
        'tli': tli,
        'rmsea': rmsea,
        'srmr': srmr,
        'gfi': gfi,
        'loadings': loadings,
        'alfa': alfa,
        'converged': True
    }

def calcular_alfa_cronbach(datos):
    """Calcula el alfa de Cronbach"""
    n_items = datos.shape[1]
    var_total = datos.sum(axis=1).var()
    sum_var_items = datos.var().sum()
    alfa = (n_items / (n_items - 1)) * (1 - sum_var_items / var_total)
    return max(0, min(1, alfa))

def calcular_ic(datos, confianza=0.95):
    """Intervalo de confianza percentil"""
    alpha = 1 - confianza
    return np.percentile(datos, alpha/2 * 100), np.percentile(datos, (1 - alpha/2) * 100)

# ============================================
# PARTE 1: AJUSTE EN MUESTRA COMPLETA
# ============================================
print(f"\n{'='*70}")
print("1. MUESTRA COMPLETA (estimación original)")
print(f"{'='*70}")

completa = calcular_cfa_manual(datos)

print(f"\n   n = {completa['n']}")
print(f"   χ²({completa['df']}) = {completa['chi2']:.2f}")
print(f"   CFI = {completa['cfi']:.3f} {'✅' if completa['cfi'] > 0.95 else '🟡' if completa['cfi'] > 0.90 else '❌'}")
print(f"   TLI = {completa['tli']:.3f} {'✅' if completa['tli'] > 0.95 else '🟡' if completa['tli'] > 0.90 else '❌'}")
print(f"   RMSEA = {completa['rmsea']:.3f} {'✅' if completa['rmsea'] < 0.06 else '🟡' if completa['rmsea'] < 0.08 else '❌'}")
print(f"   SRMR = {completa['srmr']:.3f} {'✅' if completa['srmr'] < 0.08 else '🟡' if completa['srmr'] < 0.10 else '❌'}")
print(f"   GFI = {completa['gfi']:.3f} {'✅' if completa['gfi'] > 0.90 else '🟡' if completa['gfi'] > 0.85 else '❌'}")
print(f"   α = {completa['alfa']:.3f} {'✅' if completa['alfa'] > 0.80 else '🟡' if completa['alfa'] > 0.70 else '❌'}")

print(f"\n   Cargas factoriales:")
for i, item in enumerate(items):
    print(f"      {item}: λ = {completa['loadings'][i]:.3f} {'✅' if abs(completa['loadings'][i]) > 0.60 else '🟡' if abs(completa['loadings'][i]) > 0.40 else '❌'}")

# ============================================
# PARTE 2: VALIDACIÓN CRUZADA SPLIT-HALF
# ============================================
print(f"\n{'='*70}")
print("2. VALIDACIÓN CRUZADA SPLIT-HALF")
print(f"{'='*70}")

print(f"\n   📋 Procedimiento:")
print(f"   • Dividir muestra aleatoriamente en 2 mitades")
print(f"   • Mitad A: n ≈ {n_total//2}")
print(f"   • Mitad B: n ≈ {n_total//2}")

# Dividir muestra aleatoriamente
indices = np.random.permutation(n_total)
mitad_a = datos.iloc[indices[:n_total//2]].reset_index(drop=True)
mitad_b = datos.iloc[indices[n_total//2:]].reset_index(drop=True)

print(f"\n   {'-'*50}")
print(f"   🔷 MITAD A (calibración)")
ajuste_a = calcular_cfa_manual(mitad_a)

print(f"   n = {ajuste_a['n']}")
print(f"   CFI = {ajuste_a['cfi']:.3f} | RMSEA = {ajuste_a['rmsea']:.3f} | SRMR = {ajuste_a['srmr']:.3f}")
print(f"   Cargas: ", end="")
for i, item in enumerate(items):
    print(f"{item}={ajuste_a['loadings'][i]:.2f} ", end="")
print()

print(f"\n   🔷 MITAD B (validación)")
ajuste_b = calcular_cfa_manual(mitad_b)

print(f"   n = {ajuste_b['n']}")
print(f"   CFI = {ajuste_b['cfi']:.3f} | RMSEA = {ajuste_b['rmsea']:.3f} | SRMR = {ajuste_b['srmr']:.3f}")
print(f"   Cargas: ", end="")
for i, item in enumerate(items):
    print(f"{item}={ajuste_b['loadings'][i]:.2f} ", end="")
print()

# Comparación split-half
print(f"\n   {'-'*50}")
print(f"   📊 COMPARACIÓN SPLIT-HALF:")

dif_cargas = np.abs(ajuste_a['loadings'] - ajuste_b['loadings'])
print(f"\n   {'Item':<8} {'Mitad A':<10} {'Mitad B':<10} {'|Dif|':<10} {'Estado'}")
print(f"   {'-'*50}")
for i, item in enumerate(items):
    estado = "✅" if dif_cargas[i] < 0.1 else "⚠️" if dif_cargas[i] < 0.2 else "❌"
    print(f"   {item:<8} {ajuste_a['loadings'][i]:<10.3f} {ajuste_b['loadings'][i]:<10.3f} {dif_cargas[i]:<10.3f} {estado}")

dif_media = np.mean(dif_cargas)
dif_max = np.max(dif_cargas)

print(f"\n   Diferencia promedio: {dif_media:.3f}")
print(f"   Máxima diferencia: {dif_max:.3f} (ítem {items[np.argmax(dif_cargas)]})")

print(f"\n   Concordancia de índices de ajuste:")
print(f"   CFI:   {ajuste_a['cfi']:.3f} vs {ajuste_b['cfi']:.3f} (Δ = {abs(ajuste_a['cfi']-ajuste_b['cfi']):.3f})")
print(f"   TLI:   {ajuste_a['tli']:.3f} vs {ajuste_b['tli']:.3f} (Δ = {abs(ajuste_a['tli']-ajuste_b['tli']):.3f})")
print(f"   RMSEA: {ajuste_a['rmsea']:.3f} vs {ajuste_b['rmsea']:.3f} (Δ = {abs(ajuste_a['rmsea']-ajuste_b['rmsea']):.3f})")
print(f"   SRMR:  {ajuste_a['srmr']:.3f} vs {ajuste_b['srmr']:.3f} (Δ = {abs(ajuste_a['srmr']-ajuste_b['srmr']):.3f})")

split_ok = (dif_media < 0.1 and 
            abs(ajuste_a['cfi']-ajuste_b['cfi']) < 0.05 and
            abs(ajuste_a['rmsea']-ajuste_b['rmsea']) < 0.05)

print(f"\n   {'✅ SPLIT-HALF: CONCORDANCIA ACEPTABLE' if split_ok else '⚠️ SPLIT-HALF: DIFERENCIAS IMPORTANTES'}")

# ============================================
# PARTE 3: BOOTSTRAP CFA COMPLETO
# ============================================
print(f"\n{'='*70}")
print("3. BOOTSTRAP CFA COMPLETO (Validación interna)")
print(f"{'='*70}")

print(f"\n   📋 Procedimiento:")
print(f"   • {N_BOOTSTRAP} remuestras bootstrap con reemplazo")
print(f"   • n = {n_total} en cada remuestra")
print(f"   • CFA unidimensional en cada iteración")
print(f"   • Estimación de distribución de parámetros")

boot_cargas = {item: [] for item in items}
boot_cfi = []
boot_rmsea = []
boot_srmr = []
boot_tli = []
converged = 0

print(f"\n   Ejecutando...")
print(f"   [{' '*50}] 0%", end='', flush=True)

for b in range(N_BOOTSTRAP):
    if b % 20 == 0:
        pct = int((b / N_BOOTSTRAP) * 100)
        bar = int((b / N_BOOTSTRAP) * 50)
        print(f"\r   [{'='*bar}{' '*(50-bar)}] {pct}%", end='', flush=True)
    
    # Remuestra bootstrap
    indices_boot = np.random.choice(n_total, size=n_total, replace=True)
    muestra_boot = datos.iloc[indices_boot].copy()
    
    # CFA en remuestra
    try:
        resultado_boot = calcular_cfa_manual(muestra_boot)
        
        # Guardar resultados
        for i, item in enumerate(items):
            boot_cargas[item].append(resultado_boot['loadings'][i])
        
        boot_cfi.append(resultado_boot['cfi'])
        boot_rmsea.append(resultado_boot['rmsea'])
        boot_srmr.append(resultado_boot['srmr'])
        boot_tli.append(resultado_boot['tli'])
        
        converged += 1
    except Exception as e:
        # Si falla, no guardar esta iteración
        pass

print(f"\r   [{'='*50}] 100%")
print(f"\n   Muestras convergidas: {converged}/{N_BOOTSTRAP} ({converged/N_BOOTSTRAP*100:.1f}%)")

# Convertir a arrays
for item in items:
    boot_cargas[item] = np.array(boot_cargas[item])
boot_cfi = np.array(boot_cfi)
boot_rmsea = np.array(boot_rmsea)
boot_srmr = np.array(boot_srmr)
boot_tli = np.array(boot_tli)

print(f"\n   {'-'*50}")
print(f"   📊 RESULTADOS BOOTSTRAP CFA:")

print(f"\n   {'Item':<8} {'Original':<10} {'Media':<10} {'SE':<8} {'IC 95%':<25} {'Sesgo'}")
print(f"   {'-'*70}")

sesgos = []
for i, item in enumerate(items):
    orig = completa['loadings'][i]
    media = np.mean(boot_cargas[item])
    se = np.std(boot_cargas[item], ddof=1)
    ic_low, ic_high = calcular_ic(boot_cargas[item])
    sesgo = media - orig
    sesgos.append(sesgo)
    
    # Evaluar sesgo
    sesgo_eval = "✅" if abs(sesgo) < 0.05 else "⚠️" if abs(sesgo) < 0.10 else "❌"
    
    print(f"   {item:<8} {orig:<10.3f} {media:<10.3f} {se:<8.3f} [{ic_low:.3f}, {ic_high:.3f}]  {sesgo:+.3f} {sesgo_eval}")

# Estadísticas de índices de ajuste
print(f"\n   📊 ÍNDICES DE AJUSTE (Bootstrap):")
print(f"   {'Índice':<10} {'Original':<10} {'Media':<10} {'SE':<8} {'IC 95%':<25}")
print(f"   {'-'*65}")

for nombre, original, boot_array in [
    ('CFI', completa['cfi'], boot_cfi),
    ('TLI', completa['tli'], boot_tli),
    ('RMSEA', completa['rmsea'], boot_rmsea),
    ('SRMR', completa['srmr'], boot_srmr)
]:
    media = np.mean(boot_array)
    se = np.std(boot_array, ddof=1)
    ic_low, ic_high = calcular_ic(boot_array)
    
    print(f"   {nombre:<10} {original:<10.3f} {media:<10.3f} {se:<8.3f} [{ic_low:.3f}, {ic_high:.3f}]")

# ============================================
# PARTE 4: SÍNTESIS Y EVALUACIÓN DE ROBUSTEZ
# ============================================
print(f"\n{'='*70}")
print("4. SÍNTESIS DE VALIDACIÓN CRUZADA")
print(" BUAP México:   Enrique Buendia Lozada")
print(f"{'='*70}")

# Criterios de estabilidad
criterios = {
    'Split-half (Δλ < 0.10)': dif_media < 0.10,
    'Bootstrap: Sesgo cargas < 0.05': all(abs(s) < 0.05 for s in sesgos),
    'Bootstrap: SE cargas < 0.10': all(np.std(boot_cargas[item], ddof=1) < 0.10 for item in items),
    'CFI estable (SE < 0.05)': np.std(boot_cfi, ddof=1) < 0.05,
    'Convergencia bootstrap > 95%': (converged/N_BOOTSTRAP) > 0.95
}

print(f"\n   📋 CRITERIOS DE ROBUSTEZ:")
puntos = 0
for criterio, cumple in criterios.items():
    estado = "✅" if cumple else "❌"
    print(f"   {estado} {criterio}")
    if cumple:
        puntos += 1

print(f"\n   EVIDENCIA: {puntos}/{len(criterios)} criterios")

# Evaluación global
if puntos == len(criterios):
    nivel_robustez = "ALTA ROBUSTEZ"
    interpretacion = "El modelo es altamente estable. Las estimaciones son confiables y replicables."
elif puntos >= len(criterios) * 0.6:
    nivel_robustez = "MODERADA ROBUSTEZ"
    interpretacion = "El modelo es razonablemente estable. Algunas estimaciones muestran variabilidad moderada."
else:
    nivel_robustez = "BAJA ROBUSTEZ"
    interpretacion = "El modelo muestra inestabilidad. Las estimaciones deben interpretarse con cautela."

print(f"\n{'='*70}")
print(f"   {nivel_robustez}")
print(f"{'='*70}")
print(f"\n   INTERPRETACIÓN:")
print(f"   {interpretacion}")

# Recomendaciones específicas
print(f"\n   💡 RECOMENDACIONES:")
if not criterios['Bootstrap: Sesgo cargas < 0.05']:
    items_sesgo = [items[i] for i, s in enumerate(sesgos) if abs(s) >= 0.05]
    print(f"   • Ítems con sesgo moderado: {', '.join(items_sesgo)}")
    print(f"     → Considerar estabilidad de estos ítems en futuras muestras")

if not criterios['Bootstrap: SE cargas < 0.10']:
    items_se = [item for item in items if np.std(boot_cargas[item], ddof=1) >= 0.10]
    print(f"   • Ítems con alta variabilidad: {', '.join(items_se)}")
    print(f"     → Las cargas de estos ítems son menos precisas")

if criterios['Split-half (Δλ < 0.10)'] and criterios['Bootstrap: SE cargas < 0.10']:
    print(f"   • Las cargas factoriales son estables entre muestras")
    print(f"   • Se puede confiar en la estructura factorial reportada")

print(f"\n{'='*70}")
print("ANÁLISIS COMPLETADO")
print(f"{'='*70}")
