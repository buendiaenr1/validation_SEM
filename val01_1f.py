import pandas as pd
import numpy as np
import pingouin as pg
from factor_analyzer import FactorAnalyzer
import warnings

# Silenciar advertencias técnicas para limpiar la salida
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    # 1. Cargar datos
    df = pd.read_csv('datos.csv', sep=';')
    
    # Limpieza básica
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # --- DETECCIÓN DE DATOS DICOTÓMICOS ---
    es_dicotomico = True
    columnas_no_dicotomicas = []

    for col in df.columns:
        # Obtenemos los valores únicos ordenados
        unicos = sorted(df[col].unique())
        # Verificamos si los únicos valores son exactamente [0, 1]
        if unicos != [0, 1]:
            es_dicotomico = False
            columnas_no_dicotomicas.append(col)
    
    print("\n\n")
    print("  Enrique R.P. Buendia Lozada   BUAP 2026")
    print("=" * 50)
    if es_dicotomico:
        print("NOTA IMPORTANTE: Los datos son DICOTÓMICOS (0 y 1).")
        print("Se ha detectado que todas las variables solo contienen valores 0 y 1.")
        print("Recomendación: El Omega de McDonald es preferible al Alfa de Cronbach")
        print("en escalas dicotómicas, ya que no asume normalidad multivariante estricta.")
    else:
        print("NOTA: Los datos NO son puramente dicotómicos.")
        if len(columnas_no_dicotomicas) > 0:
            print(f"Las siguientes columnas tienen otros valores además de 0 y 1: {columnas_no_dicotomicas[:5]}...")
    print("=" * 50)
    print("\n")



    # --- DETECCIÓN Y REPORTE DE ÍTEMS A ELIMINAR ---
    # Identificar columnas con varianza cero antes de borrarlas
    std_devs = df.std()
    columnas_a_borrar = std_devs[std_devs == 0].index.tolist()
    
    if columnas_a_borrar:
        print(f"\n[ELIMINACIÓN] Se detectaron {len(columnas_a_borrar)} ítems con varianza cero (todos 0 o todos 1).")
        print(f"Ítems eliminados del análisis: {columnas_a_borrar}")
    else:
        print("\n[TODOS LOS ÍTEMS] Todos los ítems tienen varianza (>0). No se elimina ninguno.")

    # Eliminar columnas constantes (varianza 0) que rompen el cálculo
    df = df.loc[:, df.std() > 0]
    n_sujetos, n_items = df.shape
    
    print(f"Procesando: {n_sujetos} sujetos, {n_items} ítems válidos.")

    if n_sujetos < 50:
        print("ADVERTENCIA: Muestra pequeña (<50). Los resultados pueden ser inestables.\n")

    # -----------------------------------------
    # CÁLCULO 1: ALFA DE CRONBACH (Pingouin)
    # -----------------------------------------
    try:
        alpha_val, ci_alpha = pg.cronbach_alpha(data=df)
        print(f"Coeficiente Alfa de Cronbach: {alpha_val:.4f}")
        print(f"Intervalo de Confianza (95%): [{ci_alpha[0][0]:.4f}, {ci_alpha[0][1]:.4f}]")
    except Exception as e:
        print(f"No se pudo calcular el Alfa de Cronbach: {e}")
    
    print("-" * 50)

    # -----------------------------------------
    # CÁLCULO 2: OMEGA DE MCDONALD (Factor Analyzer)
    # -----------------------------------------
    try:
        # Usamos 'uls' (Mínimos Cuadrados No Ponderados) que suele ser más estable 
        # para datos dicotómicos cuando no se usa matriz tetracórica explícita,
        # aunque 'ml' también funciona con muestras grandes (>200).
        fa = FactorAnalyzer(n_factors=1, rotation=None, method='uls')
        fa.fit(df)

        # Extracción de parámetros (Manejo correcto de NumPy arrays)
        loadings = fa.loadings_.flatten()
        uniquenesses = fa.get_uniquenesses()

        # Fórmula del Omega Total
        sum_loadings = np.sum(loadings)
        numerator = sum_loadings ** 2
        denominator = numerator + np.sum(uniquenesses)

        if denominator == 0:
            omega_total = 0.0
        else:
            omega_total = numerator / denominator

        print(f"Omega de McDonald Total: {omega_total:.4f}")
        
        # Interpretación
        if omega_total >= 0.9:
            nivel = "Excelente"
        elif omega_total >= 0.8:
            nivel = "Bueno"
        elif omega_total >= 0.7:
            nivel = "Aceptable"
        else:
            nivel = "Bajo"
        
        print(f"Interpretación: Consistencia interna {nivel}.")
        print("-" * 50)
        print("\n\n")
        print("\nPresiona Enter para salir...")
        input()

    except Exception as e:
        print(f"Error al calcular Omega: {e}")
        import traceback
        traceback.print_exc()

except FileNotFoundError:
    print("Error crítico: No se encuentra el archivo 'datos.csv'. Verifica la ruta.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")

