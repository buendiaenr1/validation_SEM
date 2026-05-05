#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SISTEMA INTEGRADO DE ANÁLISIS PSICOMÉTRICO (VERSIÓN CORREGIDA 4.0)
================================================================================
Autor: Enrique R.P. Buendia Lozada
Institución: BUAP México
Versión: 4.0 - Mayo 2026


================================================================================
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Configuración de visualización
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZACION_DISPONIBLE = True
except ImportError:
    VISUALIZACION_DISPONIBLE = False
    print("⚠️  matplotlib/seaborn no disponibles.")

# Configuración de semopy
try:
    import semopy
    from semopy import Model, Optimizer
    from semopy.stats import calc_stats
    SEMOPY_DISPONIBLE = True
except ImportError:
    SEMOPY_DISPONIBLE = False
    print("⚠️  semopy no disponible. Instalar: pip install semopy")

try:
    from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FACTOR_ANALYZER_DISPONIBLE = True
except ImportError:
    FACTOR_ANALYZER_DISPONIBLE = False
    print("⚠️  factor_analyzer no disponible. Instalar: pip install factor-analyzer")

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================

CONFIG = {
    'N_BOOTSTRAP': 1000,
    'ARCHIVO_DATOS': 'datos.csv',
    'DIRECTORIO_SALIDA': 'resultados_psicometricos',
    'FORMATO_FECHA': '%Y%m%d_%H%M%S',
    'MIN_ITEMS_POR_FACTOR': 2,  # Reducido para modelos con pocos ítems
    'NIVEL_CONFIANZA_RMSEA': 0.90,
    'UMBRALES_INVARIANZA': {
        'ΔCFI': 0.010,   # Chen (2007)
        'ΔRMSEA': 0.015,
        'ΔSRMR': 0.030   # para invarianza métrica
    }
}

# ==============================================================================
# CLASE PRINCIPAL
# ==============================================================================

class SistemaPsicometrico:
    """Sistema integrado para análisis psicométrico con todas las correcciones."""

    def __init__(self, ruta_csv=None):
        self.ruta_csv = ruta_csv
        self.df = None
        self.datos_limpios = None
        self.items_cols = None
        self.estructura_factores = None
        self.resultados = {}
        self.modelo_unifactorial = None
        self.modelo_multifactorial = None
        self.n_factores = 1
        self.autovalores = None
        self.varianza_explicada = None
        self.historial = []
        self.cargas_unif = None
        self.cargas_mult = None
        self.grupos_invarianza = None
        
        self._verificar_directorio()
        self._mostrar_bienvenida()

    def _mostrar_bienvenida(self):
        """Muestra información inicial."""
        print(" " * 20 + " ")
        print("\n" + "="*80)
        print(" " * 20 + "SISTEMA DE ANÁLISIS PSICOMÉTRICO v4.0")
        print(" " * 20 + "BUAP 2026. Enrique Buendia L.")
        print(" " * 20 + "Detección Automática de Factores")
        print("="*80)
        print("\n📋 V. v4.0:")
        
        print("="*80 + "\n")

    def _verificar_directorio(self):
        if not os.path.exists(CONFIG['DIRECTORIO_SALIDA']):
            os.makedirs(CONFIG['DIRECTORIO_SALIDA'])

    def _log(self, mensaje, tipo='info'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        iconos = {'error': '❌', 'exito': '✓', 'advertencia': '⚠️', 'info': 'ℹ️'}
        print(f"[{timestamp}] {iconos.get(tipo, 'ℹ️')} {mensaje}")

    # ========================================================================
    # CARGA DE DATOS
    # ========================================================================

    def cargar_datos(self, ruta_csv):
        if not os.path.isfile(ruta_csv):
            self._log(f"Archivo no encontrado: {ruta_csv}", tipo='error')
            return False
        try:
            for sep in [';', ',']:
                try:
                    df = pd.read_csv(ruta_csv, sep=sep, encoding='utf-8')
                    if df.shape[1] > 1:
                        break
                except:
                    continue
            else:
                lista_filas = []
                encabezados = None
                with open(ruta_csv, 'r', encoding='utf-8') as f:
                    for linea in f:
                        linea = linea.strip()
                        if not linea:
                            continue
                        if '|' in linea:
                            contenido = linea.split('|')[-1].strip()
                            columnas = [c.strip() for c in contenido.split(';')]
                        else:
                            columnas = [c.strip() for c in linea.split(';')]
                        if encabezados is None:
                            encabezados = columnas
                        elif len(columnas) == len(encabezados):
                            lista_filas.append(columnas)
                df = pd.DataFrame(lista_filas, columns=encabezados)
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(thresh=len(df.columns)//2)
            
            self._log(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas", tipo='exito')
            self.df = df
            return True
        except Exception as e:
            self._log(f"Error: {e}", tipo='error')
            return False

    # ========================================================================
    # DETECCIÓN DE ÍTEMS
    # ========================================================================

    def detectar_items(self):
        excluir = ['id', 'grupos', 'genero', 'grupo', 'edad', 'pais', 'ciudad']
        items = []
        for col in self.df.columns:
            col_lower = col.lower()
            if col_lower not in excluir:
                try:
                    if self.df[col].dtype in ['int64', 'float64']:
                        if self.df[col].nunique() <= 7:
                            items.append(col)
                except:
                    pass
        if not items:
            items = self.df.select_dtypes(include=[np.number]).columns.tolist()
            items = [c for c in items if c.lower() not in excluir]
        self._log(f"Detectados {len(items)} ítems")
        return items

    # ========================================================================
    # ANÁLISIS DE AUTOVALORES
    # ========================================================================

    def analizar_autovalores(self):
        print("\n" + "="*80)
        print("🔬 ANÁLISIS DE AUTO-VALORES (CRITERIO DE KAISER)")
        print("="*80)
        
        corr_matrix = self.datos_limpios.corr().values
        autovalores = np.linalg.eigvals(corr_matrix)
        autovalores = np.sort(np.abs(autovalores))[::-1]
        
        n_items = len(self.items_cols)
        varianza = autovalores / n_items * 100
        varianza_acum = np.cumsum(varianza)
        
        self.n_factores = max(1, np.sum(autovalores > 1))  # Al menos 1 factor
        self.autovalores = autovalores
        self.varianza_explicada = varianza
        
        print(f"\n  Núm. ítems: {n_items}")
        print(f"  Factor | Autovalor | >1? | % Varianza | % Acumulado")
        print(f"  -------|-----------|-----|------------|-------------")
        for i in range(len(autovalores)):
            kaiser = "✅" if autovalores[i] > 1 else "❌"
            print(f"    {i+1:2}    |   {autovalores[i]:.4f}  |  {kaiser}  |    {varianza[i]:.2f}%    |    {varianza_acum[i]:.2f}%")
        
        print(f"\n  📌 CRITERIO DE KAISER: {self.n_factores} FACTOR(ES)")
        print(f"  📌 VARIANZA EXPLICADA: {varianza_acum[self.n_factores-1]:.2f}%")
        
        # Advertencia si solo hay 1 factor con pocos ítems
        if self.n_factores == 1 and n_items <= 5:
            print(f"\n  ℹ️  NOTA: Con solo {n_items} ítems, la solución unifactorial es esperable.")
            print(f"     El modelo CFA tendrá {n_items*(n_items-1)//2 - n_items} grados de libertad,")
            print(f"     lo cual puede producir índices artificialmente elevados si está cerca de la saturación.")
        
        self.resultados['autovalores'] = {
            'valores': autovalores.tolist(),
            'varianza': varianza.tolist(),
            'n_factores': self.n_factores
        }
        return self.n_factores

    def sugerir_estructura_factorial(self):
        print("\n" + "="*80)
        print(f"📐 ESTRUCTURA FACTORIAL SUGERIDA ({self.n_factores} FACTORES)")
        print("="*80)
        
        if self.n_factores == 1:
            estructura = {'Factor_1': list(self.items_cols)}
            print(f"\n  📌 Factor_1 ({len(self.items_cols)} ítems):")
            for item in self.items_cols:
                print(f"       • {item}")
        else:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            datos_std = scaler.fit_transform(self.datos_limpios)
            pca = PCA(n_components=self.n_factores)
            pca.fit(datos_std)
            cargas = pca.components_.T
            
            estructura = {f'Factor_{i+1}': [] for i in range(self.n_factores)}
            for i, item in enumerate(self.items_cols):
                cargas_item = np.abs(cargas[i])
                factor_max = np.argmax(cargas_item)
                estructura[f'Factor_{factor_max+1}'].append(item)
            
            # Filtrar factores con pocos ítems
            estructura_filtrada = {}
            for factor, items in estructura.items():
                if len(items) >= CONFIG['MIN_ITEMS_POR_FACTOR']:
                    estructura_filtrada[factor] = items
                elif len(items) > 0:
                    for other_factor in estructura_filtrada:
                        estructura_filtrada[other_factor].extend(items)
                        break
            
            estructura = estructura_filtrada if estructura_filtrada else estructura
            
            print("\n  Asignación de ítems por factor:")
            for factor, items in estructura.items():
                print(f"\n  📌 {factor} ({len(items)} ítems):")
                for item in items[:10]:
                    print(f"       • {item}")
        
        self.estructura_factores = estructura
        return estructura

    # ========================================================================
    # PRUEBAS PRELIMINARES
    # ========================================================================

    def pruebas_preliminares(self):
        print("\n" + "="*80)
        print("📐 PRUEBAS PRELIMINARES")
        print("="*80)
        
        if FACTOR_ANALYZER_DISPONIBLE:
            try:
                kmo_all, kmo_model = calculate_kmo(self.datos_limpios)
                chi2, p_val = calculate_bartlett_sphericity(self.datos_limpios)
                
                print(f"\n  ✓ KMO: {kmo_model:.3f}")
                if kmo_model >= 0.9:
                    print("    → Excelente adecuación muestral")
                elif kmo_model >= 0.8:
                    print("    → Buena adecuación muestral")
                elif kmo_model >= 0.7:
                    print("    → Adecuación aceptable")
                else:
                    print("    → Adecuación regular/baja")
                
                print(f"\n  ✓ Bartlett: χ² = {chi2:.2f}, p = {p_val:.6f}")
                if p_val < 0.05:
                    print("    → Matriz factorizable")
                
                self.resultados['preliminares'] = {'kmo': kmo_model, 'bartlett_chi2': chi2, 'bartlett_p': p_val}
            except Exception as e:
                self._log(f"Error en pruebas: {e}", tipo='advertencia')

    # ========================================================================
    # CONFIABILIDAD: α de Cronbach y ω de McDonald
    # ========================================================================

    def calcular_confiabilidad(self):
        """Calcula α de Cronbach y ω de McDonald para el modelo unifactorial."""
        print("\n" + "="*80)
        print("📊 CONFIABILIDAD (α de Cronbach y ω de McDonald)")
        print("="*80)
        
        items = self.datos_limpios.values
        n_items = items.shape[1]
        
        # ================================================================
        # Alpha de Cronbach
        # ================================================================
        var_items = np.var(items, axis=0, ddof=1)
        var_total = np.var(np.sum(items, axis=1), ddof=1)
        alpha = (n_items / (n_items - 1)) * (1 - np.sum(var_items) / var_total)
        
        print(f"\n  📌 α de Cronbach: {alpha:.4f}")
        if alpha >= 0.90:
            print("     → Excelente")
        elif alpha >= 0.80:
            print("     → Buena")
        elif alpha >= 0.70:
            print("     → Aceptable")
        else:
            print("     → Baja")
        
        # ================================================================
        # Omega de McDonald
        # ================================================================
        omega_total = None
        omega_jerarquico = None
        cargas_para_omega = None
        
        # INTENTO 1: Extraer cargas desde semopy (si existen)
        if self.modelo_unifactorial is not None:
            try:
                insp = self.modelo_unifactorial.inspect()
                cargas_df = insp[insp['op'] == '=~']
                
                if len(cargas_df) > 0:
                    if 'Std. Est' in cargas_df.columns:
                        cargas_para_omega = np.abs(cargas_df['Std. Est'].values)
                    else:
                        cargas_para_omega = np.abs(cargas_df['Estimate'].values)
                    print(f"\n  ℹ️  ω calculado desde cargas CFA (semopy)")
            except:
                pass
        
        # INTENTO 2: Usar cargas almacenadas de PCA (si falló semopy)
        if cargas_para_omega is None and self.cargas_unif is not None:
            try:
                cargas_para_omega = np.array([v['carga'] for v in self.cargas_unif.values()])
                print(f"\n  ℹ️  ω calculado desde cargas PCA almacenadas")
            except:
                pass
        
        # INTENTO 3: Calcular PCA ahora mismo
        if cargas_para_omega is None:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                pca.fit(self.datos_limpios)
                cargas_para_omega = np.abs(pca.components_[0])
                print(f"\n  ℹ️  ω calculado desde PCA (calculado ahora)")
            except:
                pass
        
        # INTENTO 4: Estimar desde autovalores
        if cargas_para_omega is None:
            try:
                corr = self.datos_limpios.corr().values
                autovalores = np.linalg.eigvals(corr)
                lambda1 = np.max(np.abs(autovalores))
                omega_total = (lambda1**2) / (lambda1**2 + n_items - lambda1)
                print(f"\n  ℹ️  ω estimado desde autovalores (aproximación)")
            except:
                pass
        
        # Calcular ω si tenemos cargas
        if cargas_para_omega is not None and len(cargas_para_omega) > 0:
            # Asegurar que son positivas
            cargas_para_omega = np.abs(cargas_para_omega)
            
            # Varianzas de error = 1 - carga² (para cargas estandarizadas)
            var_error = 1 - cargas_para_omega**2
            var_error = np.clip(var_error, 0.001, None)  # Evitar negativos o cero
            
            # Omega total
            suma_cargas = np.sum(cargas_para_omega)
            omega_total = suma_cargas**2 / (suma_cargas**2 + np.sum(var_error))
            
            print(f"\n  📌 ω de McDonald (total): {omega_total:.4f}")
            if omega_total >= 0.90:
                print("     → Excelente")
            elif omega_total >= 0.80:
                print("     → Buena")
            elif omega_total >= 0.70:
                print("     → Aceptable")
            elif omega_total >= 0.50:
                print("     → Moderada (revisar ítems con cargas bajas)")
            else:
                print("     → Baja (requiere revisión de ítems)")
            
            # Mostrar contribución de cada ítem a omega
            print(f"\n  Contribución de cada ítem a ω:")
            print(f"  {'Ítem':<8} {'Carga':>8} {'Carga²':>8} {'Error':>8}")
            print(f"  " + "-" * 35)
            for i, (item, carga) in enumerate(zip(self.items_cols, cargas_para_omega)):
                print(f"  {item:<8} {carga:>8.4f} {carga**2:>8.4f} {var_error[i]:>8.4f}")
            
            # Nota sobre discrepancia α vs ω
            diff = omega_total - alpha
            if abs(diff) > 0.05:
                print(f"\n  ℹ️  NOTA: Diferencia α - ω = {alpha - omega_total:.4f}")
                if alpha > omega_total:
                    print(f"     α > ω: las cargas factoriales no son iguales (no hay tau-equivalencia)")
                    print(f"     α puede estar sobreestimando la confiabilidad. Se recomienda reportar ω.")
                else:
                    print(f"     ω > α: posible efecto de correlaciones entre errores")
        
        # Guardar en resultados
        self.resultados['confiabilidad'] = {
            'alpha_cronbach': alpha,
            'omega_mcdonald_total': omega_total,
            'omega_mcdonald_jerarquico': omega_jerarquico,
            'cargas_usadas_omega': cargas_para_omega.tolist() if cargas_para_omega is not None else None
        }
        
        # Guardar también las cargas para uso en SEM estructural
        if cargas_para_omega is not None and self.cargas_unif is None:
            self.cargas_unif = {
                item: {'factor': 'F1', 'carga': float(carga), 'se': None, 'p_val': None}
                for item, carga in zip(self.items_cols, cargas_para_omega)
            }
        
        return alpha, omega_total

    # ========================================================================
    # PASO 1: DIAGNÓSTICO
    # ========================================================================

    def paso_1_diagnostico(self):
        self._log("\n" + "="*80, tipo='info')
        self._log("PASO 1: DIAGNÓSTICO Y DETECCIÓN DE FACTORES", tipo='info')
        self._log("="*80)

        if not self.cargar_datos(self.ruta_csv or CONFIG['ARCHIVO_DATOS']):
            return False

        self.items_cols = self.detectar_items()
        if not self.items_cols:
            self._log("No se detectaron ítems", tipo='error')
            return False

        self.datos_limpios = self.df[self.items_cols].dropna()
        self._log(f"Datos: {len(self.datos_limpios)} casos, {len(self.items_cols)} ítems", tipo='exito')
        
        self._mostrar_estadisticas()
        self.pruebas_preliminares()
        self.analizar_autovalores()
        self.sugerir_estructura_factorial()
        
        return True

    def _mostrar_estadisticas(self):
        print("\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
        print("-" * 70)
        desc = self.datos_limpios.describe().T
        print(desc[['mean', 'std', 'min', 'max']].round(3).to_string())

    # ========================================================================
    # PASO 2: CFA CORREGIDO
    # ========================================================================

    def paso_2_cfa_sem(self):
        if self.datos_limpios is None:
            self._log("ERROR: Ejecute primero el Paso 1", tipo='error')
            return

        self._log("\n" + "="*80, tipo='info')
        self._log("PASO 2: ANÁLISIS FACTORIAL CONFIRMATORIO (CFA)", tipo='info')
        self._log("="*80)
        
        if not SEMOPY_DISPONIBLE:
            self._log("semopy no disponible", tipo='error')
            return
        
        self._comparar_modelos()
        self._estimar_modelo_unifactorial()
        
        # Solo estimar multifactorial si hay > 1 factor
        if self.n_factores > 1:
            self._estimar_modelo_multifactorial()
        
        self._recomendar_modelo()
        
        # Calcular confiabilidad
        self.calcular_confiabilidad()
        
        # Modelo SEM estructural
        self._estimar_sem_estructural()

    def _comparar_modelos(self):
        print("\n" + "="*70)
        print("📊 COMPARACIÓN DE MODELOS")
        print("="*70)
        
        n_items = len(self.items_cols)
        n_param_unif = n_items  # cargas + varianzas error (semopy cuenta ambas)
        cov_obs = n_items * (n_items - 1) // 2
        gl_unif = cov_obs - n_items  # gl = covarianzas observadas - parámetros libres
        # En semopy, los parámetros incluyen cargas Y varianzas de error
        
        print(f"\n  Modelo 1: UNIFACTORIAL")
        print(f"    • Un solo factor general")
        print(f"    • Parámetros estimados: {n_items} cargas + {n_items} varianzas error = {2*n_items}")
        print(f"    • Covarianzas observadas: {cov_obs}")
        print(f"    • Grados de libertad ≈ {gl_unif}")
        
        if gl_unif <= 2:
            print(f"    ⚠️  ADVERTENCIA: Modelo casi saturado (gl = {gl_unif})")
            print(f"       Los índices de ajuste pueden estar artificialmente inflados")
        
        if self.n_factores > 1:
            n_fact = self.n_factores
            n_param_mult = n_items + n_items  # cargas + varianzas error
            n_correlaciones = n_fact * (n_fact - 1) // 2
            gl_mult = cov_obs - (n_param_mult + n_correlaciones)
            print(f"\n  Modelo 2: MULTIFACTORIAL ({n_fact} factores)")
            print(f"    • Parámetros: {n_param_mult + n_correlaciones}")
            print(f"    • Grados de libertad: {gl_mult}")
        else:
            print(f"\n  Modelo 2: MULTIFACTORIAL → Solo 1 factor detectado, mismo modelo")
        
        print("\n" + "-"*70)

    def _estimar_modelo_unifactorial(self):
        print("\n" + "="*70)
        print("📌 MODELO UNIFACTORIAL (1 factor)")
        print("="*70)
        
        model_desc = "# Modelo Unifactorial\n"
        model_desc += "F1 =~ " + " + ".join(self.items_cols) + "\n"
        
        self._log(f"Especificación:\n{model_desc.strip()}")
        
        try:
            model = Model(model_desc)
            # CORRECCIÓN: Pasar DataFrame (no array) y ejecutar fit()
            model.fit(self.datos_limpios)  # ← DataFrame, no .values
            self.modelo_unifactorial = model
            
            self._mostrar_cargas_corregido(model, "Unifactorial")
            self._mostrar_ajuste_corregido(model, "Unifactorial")
            
        except Exception as e:
            self._log(f"Error en modelo unifactorial: {e}", tipo='error')
            # Intentar con método alternativo
            try:
                self._log("Intentando con estimación ML estándar...", tipo='advertencia')
                model = Model(model_desc)
                model.fit(self.datos_limpios, obj='ML')  # ← Forzar ML explícito
                self.modelo_unifactorial = model
                self._mostrar_cargas_corregido(model, "Unifactorial")
                self._mostrar_ajuste_corregido(model, "Unifactorial")
            except Exception as e2:
                self._log(f"Error persistente: {e2}", tipo='error')
                # TERCER INTENTO: Usar solo la matriz de covarianzas
                try:
                    self._log("Intentando con matriz de covarianzas...", tipo='advertencia')
                    model = Model(model_desc)
                    cov_matrix = self.datos_limpios.cov().values
                    n_samples = len(self.datos_limpios)
                    model.fit_cov(cov_matrix, n_samples)  # ← Alternativa con covarianzas
                    self.modelo_unifactorial = model
                    self._mostrar_cargas_corregido(model, "Unifactorial")
                    self._mostrar_ajuste_corregido(model, "Unifactorial")
                except Exception as e3:
                    self._log(f"Todos los intentos fallaron: {e3}", tipo='error')

    def _estimar_modelo_multifactorial(self):
        print("\n" + "="*70)
        print(f"📌 MODELO MULTIFACTORIAL ({len(self.estructura_factores)} factores)")
        print("="*70)
        
        model_desc = "# Modelo Multifactorial\n"
        for factor, items in self.estructura_factores.items():
            if len(items) >= 2:
                model_desc += f"{factor} =~ " + " + ".join(items) + "\n"
        
        factores = list(self.estructura_factores.keys())
        for i in range(len(factores)):
            for j in range(i+1, len(factores)):
                model_desc += f"{factores[i]} ~~ {factores[j]}\n"
        
        self._log(f"Especificación:\n{model_desc.strip()}")
        
        try:
            model = Model(model_desc)
            model.fit(self.datos_limpios)
            self.modelo_multifactorial = model
            
            self._mostrar_cargas_corregido(model, "Multifactorial")
            self._mostrar_ajuste_corregido(model, "Multifactorial")
            self._mostrar_correlaciones_factores(model)
            
        except Exception as e:
            self._log(f"Error en modelo multifactorial: {e}", tipo='error')

    def _mostrar_cargas_corregido(self, model, nombre_modelo):
        """Versión corregida de extracción de cargas."""
        print(f"\n  CARGAS FACTORIALES ESTANDARIZADAS ({nombre_modelo}):")
        print("  " + "-" * 60)
        print(f"  {'Ítem':<10} {'Factor':<12} {'Carga':>8} {'Error Est.':>10} {'p-valor':>10} {'Sig.':>6}")
        print("  " + "-" * 60)
        
        try:
            # Obtener estimaciones
            insp = model.inspect()
            params = insp[insp['op'] == '=~'].copy()
            
            if len(params) == 0:
                print("  ⚠️  No se encontraron parámetros de carga (=~)")
                print("  Intentando método alternativo (PCA desde matriz de correlaciones)...")
                
                # EXTRAER CARGAS DIRECTAMENTE DESDE PCA (respaldo robusto)
                try:
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    scaler = StandardScaler()
                    datos_std = scaler.fit_transform(self.datos_limpios)
                    pca = PCA(n_components=1)
                    pca.fit(datos_std)
                    
                    cargas_pca = np.abs(pca.components_[0])
                    
                    # Crear DataFrame simulado con el formato esperado
                    params_list = []
                    for i, item in enumerate(self.items_cols):
                        params_list.append({
                            'lval': 'F1',
                            'op': '=~',
                            'rval': item,
                            'Estimate': cargas_pca[i],
                            'Std. Est': cargas_pca[i],  # PCA ya está estandarizado
                            'Std. Err': np.nan,
                            'p-value': np.nan
                        })
                    
                    params = pd.DataFrame(params_list)
                    print(f"  ✓ Cargas extraídas desde PCA (método alternativo)")
                    print(f"  ℹ️  NOTA: Las cargas PCA son aproximaciones. No se dispone de errores estándar ni p-valores.")
                    
                except Exception as e_inner:
                    print(f"  ❌ Error en PCA: {e_inner}")
                    # ÚLTIMO INTENTO: Extraer de la inspección completa del modelo
                    try:
                        todas_ops = insp['op'].unique()
                        print(f"  Operaciones disponibles: {todas_ops}")
                        
                        # En semopy, a veces las cargas aparecen sin etiqueta '=~'
                        # Buscar en todas las filas que tengan 'Estimate'
                        todas_cargas = insp[insp['Estimate'].notna()].copy()
                        if len(todas_cargas) > 0 and len(todas_cargas) >= len(self.items_cols):
                            # Filtrar solo las que parecen cargas (valores entre -2 y 2)
                            posibles_cargas = todas_cargas[
                                (todas_cargas['Estimate'].abs() < 2.0)
                            ]
                            if len(posibles_cargas) >= len(self.items_cols):
                                params = posibles_cargas.head(len(self.items_cols))
                                print(f"  ✓ Cargas identificadas por heurística")
                            else:
                                print("  ❌ No se pudieron identificar cargas")
                                return
                        else:
                            print("  ❌ No se pudieron extraer cargas automáticamente.")
                            return
                    except:
                        print("  ❌ Fallo completo en extracción de cargas")
                        return
                    
            cargas_guardar = {}
            
            for _, row in params.iterrows():
                item = row.get('rval', row.get('Variable', 'NA'))
                factor = row.get('lval', row.get('Factor', 'NA'))
                
                # Priorizar carga estandarizada
                if 'Std. Est' in row and not pd.isna(row['Std. Est']):
                    carga = row['Std. Est']
                elif 'Est. Std' in row and not pd.isna(row['Est. Std']):
                    carga = row['Est. Std']
                elif 'Estimate' in row and not pd.isna(row['Estimate']):
                    carga = row['Estimate']
                    print(f"    ℹ️  {item}: usando carga no estandarizada ({carga:.4f})")
                else:
                    carga = 0.0
                
                # Error estándar
                se = row.get('Std. Err', row.get('SE', np.nan))
                
                # p-valor
                p_val = row.get('p-value', row.get('p', np.nan))
                
                # Interpretación
                if abs(carga) >= 0.7:
                    icono = "✅"
                elif abs(carga) >= 0.5:
                    icono = "✔️"
                elif abs(carga) >= 0.3:
                    icono = "⚠️"
                else:
                    icono = "❌"
                
                sig_str = ""
                if not np.isnan(p_val):
                    if p_val < 0.001:
                        sig_str = "***"
                    elif p_val < 0.01:
                        sig_str = "**"
                    elif p_val < 0.05:
                        sig_str = "*"
                    else:
                        sig_str = "ns"
                
                se_str = f"{se:.4f}" if not np.isnan(se) else "N/A"
                p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
                
                print(f"  {icono} {item:<8} → {factor:<10} {carga:>8.4f} {se_str:>10} {p_str:>10} {sig_str:>6}")
                
                cargas_guardar[item] = {
                    'factor': factor,
                    'carga': carga,
                    'se': se if not np.isnan(se) else None,
                    'p_val': p_val if not np.isnan(p_val) else None,
                    'significativa': p_val < 0.05 if not np.isnan(p_val) else None
                }
            
            # Guardar
            if nombre_modelo == "Unifactorial":
                self.cargas_unif = cargas_guardar
            else:
                self.cargas_mult = cargas_guardar
            
            # Advertencia si hay cargas > 1.0
            cargas_abs = [abs(v['carga']) for v in cargas_guardar.values()]
            if any(c > 1.0 for c in cargas_abs):
                print(f"\n  ⚠️  ADVERTENCIA: Se detectaron cargas > 1.0")
                print(f"     Esto puede indicar:")
                print(f"     - Solución impropia (caso Heywood)")
                print(f"     - Cargas no estandarizadas")
                print(f"     - Colinealidad entre ítems")
                print(f"     Verifique que las cargas sean estandarizadas")
            
        except Exception as e:
            print(f"  ❌ Error extrayendo cargas: {e}")

    def _mostrar_ajuste_corregido(self, model, nombre_modelo):
        """Versión corregida con RMSEA IC y SRMR."""
        try:
            stats_df = calc_stats(model)
            
            # Extraer valores
            chi2 = stats_df['chi2'].iloc[0] if 'chi2' in stats_df.columns else None
            chi2_p = stats_df['chi2 p-value'].iloc[0] if 'chi2 p-value' in stats_df.columns else None
            dof = stats_df['DoF'].iloc[0] if 'DoF' in stats_df.columns else None
            cfi = stats_df['CFI'].iloc[0] if 'CFI' in stats_df.columns else None
            tli = stats_df['TLI'].iloc[0] if 'TLI' in stats_df.columns else None
            rmsea = stats_df['RMSEA'].iloc[0] if 'RMSEA' in stats_df.columns else None
            
            n = len(self.datos_limpios)
            
            # Calcular RMSEA con IC 90%
            rmsea_ic = self._calcular_rmsea_ic(chi2, dof, n)
            
            # Calcular SRMR
            srmr = self._calcular_srmr(model)
            
            print(f"\n  ÍNDICES DE AJUSTE ({nombre_modelo}):")
            print("  " + "-" * 60)
            
            # Chi-cuadrado
            if chi2 is not None and dof is not None:
                print(f"  χ²({dof}) = {chi2:.4f}, p = {chi2_p:.4f}")
                if chi2_p > 0.05:
                    print(f"     ✅ No significativo → buen ajuste")
                else:
                    print(f"     ⚠️  Significativo → esperable con N grande")
            
            # CFI
            if cfi is not None:
                if cfi > 1.0:
                    interp = "⚠️  > 1.0 (modelo casi saturado)"
                elif cfi >= 0.95:
                    interp = "Excelente"
                elif cfi >= 0.90:
                    interp = "Aceptable"
                else:
                    interp = "Pobre"
                print(f"  CFI = {cfi:.4f}  [{interp}]")
            
            # TLI
            if tli is not None:
                if tli > 1.0:
                    interp = "⚠️  > 1.0 (modelo casi saturado)"
                elif tli >= 0.95:
                    interp = "Excelente"
                elif tli >= 0.90:
                    interp = "Aceptable"
                else:
                    interp = "Pobre"
                print(f"  TLI = {tli:.4f}  [{interp}]")
            
            # RMSEA con IC
            if rmsea_ic:
                print(f"  RMSEA = {rmsea_ic['rmsea']:.4f} (IC 90%: {rmsea_ic['lo']:.4f} - {rmsea_ic['hi']:.4f})")
                if rmsea_ic['rmsea'] <= 0.05:
                    print(f"     → Excelente")
                elif rmsea_ic['rmsea'] <= 0.08:
                    print(f"     → Aceptable")
                else:
                    print(f"     → Pobre")
                
                # p-close
                if rmsea_ic['p_close'] is not None:
                    print(f"     p(RMSEA ≤ 0.05) = {rmsea_ic['p_close']:.4f}")
            elif rmsea is not None:
                print(f"  RMSEA = {rmsea:.4f}")
            
            # SRMR
            if srmr is not None:
                if srmr <= 0.05:
                    interp = "Excelente"
                elif srmr <= 0.08:
                    interp = "Aceptable"
                else:
                    interp = "Pobre"
                print(f"  SRMR = {srmr:.4f}  [{interp}]")
            
            # AIC, BIC
            for col in ['AIC', 'BIC']:
                if col in stats_df.columns:
                    print(f"  {col} = {stats_df[col].iloc[0]:.4f}")
            
            # Advertencias
            print(f"\n  🔍 DIAGNÓSTICO:")
            if cfi is not None and cfi > 1.0:
                print(f"  ⚠️  CFI > 1.0: posible modelo saturado o error de cálculo")
            if tli is not None and tli > 1.0:
                print(f"  ⚠️  TLI > 1.0: posible modelo saturado o error de cálculo")
            if dof is not None and dof <= 2:
                print(f"  ⚠️  Modelo casi saturado ({dof} gl). Índices inflados artificialmente.")
            if rmsea_ic and rmsea_ic['rmsea'] < 0.001 and dof and dof > 2:
                print(f"  ℹ️  RMSEA ≈ 0: ajuste muy bueno, pero verificar con IC")
            
            # Guardar
            ajuste_dict = {
                'chi2': chi2, 'chi2_p': chi2_p, 'dof': dof,
                'CFI': cfi, 'TLI': tli,
                'RMSEA': rmsea_ic['rmsea'] if rmsea_ic else rmsea,
                'RMSEA_IC_lo': rmsea_ic['lo'] if rmsea_ic else None,
                'RMSEA_IC_hi': rmsea_ic['hi'] if rmsea_ic else None,
                'SRMR': srmr
            }
            for col in ['AIC', 'BIC', 'GFI', 'AGFI', 'NFI']:
                if col in stats_df.columns:
                    ajuste_dict[col] = stats_df[col].iloc[0]
            
            self.resultados[f'ajuste_{nombre_modelo.lower()}'] = ajuste_dict
            
        except Exception as e:
            self._log(f"Error en índices de ajuste: {e}", tipo='advertencia')

    def _calcular_rmsea_ic(self, chi2, dof, n):
        """Calcula RMSEA con IC 90% según Browne & Cudeck (1993)."""
        if chi2 is None or dof is None or dof < 1:
            return None
        
        try:
            from scipy.stats import nct
            
            rmsea = np.sqrt(max(0, (chi2 - dof) / (dof * (n - 1))))
            
            # IC 90%
            alpha = 0.10
            if chi2 > dof:
                ncp_lo = nct.ppf(alpha / 2, dof, chi2)
                ncp_hi = nct.ppf(1 - alpha / 2, dof, chi2)
            else:
                ncp_lo = 0
                ncp_hi = nct.ppf(1 - alpha, dof, chi2) if chi2 > 0 else 0
            
            rmsea_lo = np.sqrt(max(0, ncp_lo / (dof * (n - 1))))
            rmsea_hi = np.sqrt(max(0, ncp_hi / (dof * (n - 1))))
            
            # p-close (H0: RMSEA ≤ 0.05)
            ncp_close = dof * (n - 1) * 0.05**2
            if ncp_close > 0:
                p_close = 1 - stats.nct.cdf(chi2, dof, ncp_close)
            else:
                p_close = None
            
            return {'rmsea': rmsea, 'lo': rmsea_lo, 'hi': rmsea_hi, 'p_close': p_close}
        except:
            return None

    def _calcular_srmr(self, model):
        """Calcula SRMR desde la matriz de correlación/residuales."""
        try:
            # Obtener matriz de covarianzas observada
            S = self.datos_limpios.cov().values
            
            # Obtener matriz de covarianzas implícita del modelo
            try:
                Sigma = model.sigma
                if Sigma is None:
                    raise ValueError("Sigma no disponible")
            except:
                # Alternativa: usar predict
                try:
                    Sigma = np.cov(model.predict(self.datos_limpios).T)
                except:
                    return None
            
            # Calcular residuales estandarizados
            residuals = S - Sigma
            
            # SRMR = sqrt(mean(residuals²))
            srmr = np.sqrt(np.mean(residuals**2))
            
            return srmr
        except:
            return None

    def _mostrar_correlaciones_factores(self, model):
        insp = model.inspect()
        corrs = insp[(insp['op'] == '~~') & (insp['lval'] != insp['rval'])]
        
        if len(corrs) > 0:
            print(f"\n  CORRELACIONES ENTRE FACTORES:")
            print("  " + "-" * 40)
            for _, row in corrs.iterrows():
                corr = row.get('Std. Est', row.get('Estimate', 0))
                print(f"    {row['lval']} ↔ {row['rval']}: {corr:.4f}")

    def _recomendar_modelo(self):
        print("\n" + "="*70)
        print("💡 RECOMENDACIÓN FINAL")
        print("="*70)
        
        cfi_unif = self.resultados.get('ajuste_unifactorial', {}).get('CFI', 0)
        cfi_mult = self.resultados.get('ajuste_multifactorial', {}).get('CFI', 0)
        
        print(f"\n  CFI Unifactorial: {cfi_unif:.4f}")
        if self.n_factores > 1:
            print(f"  CFI Multifactorial: {cfi_mult:.4f}")
        
        print(f"\n  📌 CONCLUSIÓN:")
        if self.n_factores == 1:
            print(f"     ✅ MODELO UNIFACTORIAL (1 factor)")
            print(f"     → Todos los ítems miden un mismo constructo")
            print(f"     → Respaldado por criterio de Kaiser: solo 1 autovalor > 1")
        elif cfi_mult > cfi_unif:
            print(f"     ✅ El MODELO MULTIFACTORIAL ({self.n_factores} factores) es mejor")
            print(f"     → ΔCFI = +{cfi_mult - cfi_unif:.4f}")
        else:
            print(f"     ✅ El MODELO UNIFACTORIAL es suficiente")
            print(f"     → El modelo multifactorial no mejora sustancialmente el ajuste")
        
        print(f"\n  📌 AUTO-VALORES > 1: {self.n_factores}")
        print(f"  📌 Valores: {', '.join([f'{v:.3f}' for v in self.autovalores[:self.n_factores]])}")

    def _estimar_sem_estructural(self):
        """Modelo SEM estructural con variable grupos."""
        print("\n" + "="*70)
        print("📌 MODELO SEM ESTRUCTURAL (Bienestar → grupos)")
        print("="*70)
        
        if 'grupos' not in self.df.columns:
            self._log("Variable 'grupos' no encontrada, omitiendo SEM estructural", tipo='advertencia')
            return
        
        # ================================================================
        # CORRECCIÓN: Usar cargas almacenadas (no depender de semopy)
        # ================================================================
        cargas_disponibles = None
        
        # Intento 1: Cargas desde modelo semopy
        if self.modelo_unifactorial is not None:
            try:
                insp = self.modelo_unifactorial.inspect()
                cargas_semopy = insp[insp['op'] == '=~']
                if len(cargas_semopy) > 0:
                    cargas_disponibles = cargas_semopy
                    print(f"  ℹ️  Usando cargas desde semopy CFA")
            except:
                pass
        
        # Intento 2: Cargas desde PCA almacenadas
        if cargas_disponibles is None and self.cargas_unif is not None:
            cargas_disponibles = self.cargas_unif
            print(f"  ℹ️  Usando cargas desde PCA almacenadas")
        
        # Intento 3: Calcular puntuación factorial con PCA y hacer regresión simple
        if cargas_disponibles is None:
            print(f"  ℹ️  Calculando puntuación factorial vía PCA para SEM...")
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                scores = pca.fit_transform(self.datos_limpios).flatten()
                
                # Regresión logística simple: grupos ~ score_bienestar
                from scipy import stats as scipy_stats
                
                # Point-biserial correlation (grupos es binario 0/1)
                grupos_vals = self.df['grupos'].dropna().values
                # Alinear índices
                common_idx = self.datos_limpios.index.intersection(self.df.dropna(subset=['grupos']).index)
                scores_aligned = scores[:len(common_idx)]
                grupos_aligned = self.df.loc[common_idx, 'grupos'].values
                
                r_pb, p_val = scipy_stats.pointbiserialr(grupos_aligned, scores_aligned)
                
                # Media de bienestar por grupo
                grupo_0 = scores_aligned[grupos_aligned == 0]
                grupo_1 = scores_aligned[grupos_aligned == 1]
                
                print(f"\n  REGRESIÓN: grupos ← Bienestar (Point-biserial)")
                print("  " + "-" * 40)
                print(f"  r_pb = {r_pb:.4f}, p = {p_val:.4f}")
                
                if p_val < 0.05:
                    print(f"  → Relación significativa (p < 0.05)")
                else:
                    print(f"  → El grupos NO predice significativamente el bienestar (p > 0.05)")
                
                # Tamaño del efecto
                print(f"\n  Puntuación factorial por grupos:")
                print(f"  Grupo 0 (n={len(grupo_0)}): M = {np.mean(grupo_0):.4f}, DE = {np.std(grupo_0):.4f}")
                print(f"  Grupo 1 (n={len(grupo_1)}): M = {np.mean(grupo_1):.4f}, DE = {np.std(grupo_1):.4f}")
                
                # Cohen's d
                d = (np.mean(grupo_1) - np.mean(grupo_0)) / np.sqrt((np.std(grupo_0)**2 + np.std(grupo_1)**2) / 2)
                print(f"  d de Cohen = {d:.4f}")
                if abs(d) < 0.2:
                    print(f"     → Efecto despreciable")
                elif abs(d) < 0.5:
                    print(f"     → Efecto pequeño")
                elif abs(d) < 0.8:
                    print(f"     → Efecto moderado")
                else:
                    print(f"     → Efecto grande")
                
                # Guardar resultados
                self.resultados['sem_estructural'] = {
                    'metodo': 'PCA + point-biserial',
                    'r_pb': r_pb,
                    'p_val': p_val,
                    'cohens_d': d,
                    'media_grupo_0': np.mean(grupo_0),
                    'media_grupo_1': np.mean(grupo_1)
                }
                
                return  # Salir después de hacer el análisis PCA
                
            except Exception as e:
                self._log(f"Error en SEM vía PCA: {e}", tipo='error')
                return
        
        # ================================================================
        # Si llegamos aquí, tenemos cargas_disponibles: intentar semopy
        # ================================================================
        try:
            model_desc = "# Modelo SEM Estructural\n"
            model_desc += "F1 =~ " + " + ".join(self.items_cols) + "\n"
            model_desc += "grupos ~ F1\n"
            
            datos_sem = self.datos_limpios.copy()
            #datos_sem['grupos'] = self.df['grupos'].values
            datos_sem['grupos'] = pd.to_numeric(self.df['grupos'], errors='coerce').values
            
            model_sem = Model(model_desc)
            model_sem.fit(datos_sem)
            
            insp_sem = model_sem.inspect()
            reg = insp_sem[(insp_sem['op'] == '~') & (insp_sem['rval'] == 'F1')]
            
            print(f"\n  REGRESIÓN SEM: grupos ← Bienestar")
            print("  " + "-" * 40)
            
            if len(reg) > 0:
                for _, row in reg.iterrows():
                    coef = row.get('Estimate', 0)
                    se = row.get('Std. Err', np.nan)
                    p = row.get('p-value', np.nan)
                    
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"  β = {coef:.4f} (SE = {se:.4f}, p = {p:.4f}) {sig}")
                    
                    if p > 0.05:
                        print(f"  → El grupos NO predice significativamente el bienestar")
                    else:
                        print(f"  → Relación significativa")
            else:
                print(f"  No se encontró la regresión esperada en semopy")
            
            # Ajuste del SEM
            print(f"\n  ÍNDICES DE AJUSTE SEM:")
            print("  " + "-" * 40)
            try:
                stats_sem = calc_stats(model_sem)
                for col in ['chi2', 'DoF', 'CFI', 'TLI', 'RMSEA']:
                    if col in stats_sem.columns:
                        val = stats_sem[col].iloc[0]
                        print(f"  {col}: {val:.4f}")
                
                cfi_sem = stats_sem['CFI'].iloc[0] if 'CFI' in stats_sem.columns else None
                tli_sem = stats_sem['TLI'].iloc[0] if 'TLI' in stats_sem.columns else None
                
                if (cfi_sem and cfi_sem < 0.10) or (tli_sem and tli_sem < -0.1):
                    print(f"\n  ⚠️  ADVERTENCIA: SEM muestra índices extremadamente pobres")
                    print(f"     Se recomienda usar el análisis PCA + point-biserial como alternativa")
            except Exception as e_stats:
                print(f"  No se pudieron calcular índices: {e_stats}")
                
        except Exception as e:
            self._log(f"Error en SEM semopy: {e}", tipo='error')
            # Fallback a PCA como último recurso
            self._log("Ejecutando análisis alternativo PCA + point-biserial...", tipo='advertencia')
            try:
                from sklearn.decomposition import PCA as PCA_sem
                from scipy import stats as scipy_stats
                
                pca_sem = PCA_sem(n_components=1)
                scores = pca_sem.fit_transform(self.datos_limpios).flatten()
                
                # Alinear con grupos no nulo
                grupos_vals = pd.to_numeric(self.df['grupos'], errors='coerce').values
                mascara = ~np.isnan(grupos_vals)
                scores_ok = scores[mascara]
                grupos_ok = grupos_vals[mascara].astype(int)
                
                r_pb, p_val = scipy_stats.pointbiserialr(grupos_ok, scores_ok)
                
                grupo_0 = scores_ok[grupos_ok == 0]
                grupo_1 = scores_ok[grupos_ok == 1]
                
                print(f"\n  REGRESIÓN: grupos ← Bienestar (Point-biserial)")
                print("  " + "-" * 50)
                print(f"  r_pb = {r_pb:.4f}, p = {p_val:.4f}")
                
                if p_val < 0.05:
                    print(f"  → Relación significativa entre Bienestar y grupos")
                else:
                    print(f"  → El grupos NO predice significativamente el Bienestar")
                
                print(f"\n  Puntuación factorial (Bienestar) por grupos:")
                print(f"  Grupo 0 (n={len(grupo_0)}): M = {np.mean(grupo_0):.4f}, DE = {np.std(grupo_0):.4f}")
                print(f"  Grupo 1 (n={len(grupo_1)}): M = {np.mean(grupo_1):.4f}, DE = {np.std(grupo_1):.4f}")
                
                d = (np.mean(grupo_1) - np.mean(grupo_0)) / np.sqrt((np.std(grupo_0)**2 + np.std(grupo_1)**2) / 2)
                print(f"\n  Tamaño del efecto:")
                print(f"  d de Cohen = {d:.4f}")
                
                if abs(d) < 0.2:
                    print(f"     → Efecto despreciable (no hay diferencias relevantes entre gruposs)")
                elif abs(d) < 0.5:
                    print(f"     → Efecto pequeño")
                elif abs(d) < 0.8:
                    print(f"     → Efecto moderado")
                else:
                    print(f"     → Efecto grande")
                
                self.resultados['sem_estructural'] = {
                    'metodo': 'PCA + point-biserial correlation',
                    'r_pb': float(r_pb),
                    'p_val': float(p_val),
                    'cohens_d': float(d),
                    'media_grupo_0': float(np.mean(grupo_0)),
                    'media_grupo_1': float(np.mean(grupo_1)),
                    'n_grupo_0': len(grupo_0),
                    'n_grupo_1': len(grupo_1)
                }
                print(f"\n  ✅ Análisis SEM alternativo completado exitosamente")
                
            except Exception as e_pca:
                self._log(f"Error en PCA alternativa: {e_pca}", tipo='error')


    # ========================================================================
    # PASO 3: INVARIANZA FORMAL
    # ========================================================================

    def paso_3_invarianza(self, variable_grupo='grupos'):
        """Invarianza de medida formal: configural, métrica, escalar, estricta."""
        self._log("\n" + "="*80, tipo='info')
        self._log("PASO 3: INVARIANZA DE MEDIDA (4 NIVELES)", tipo='info')
        self._log("="*80)

        if variable_grupo not in self.df.columns:
            self._log(f"Variable '{variable_grupo}' no encontrada", tipo='advertencia')
            return

        grupos = sorted(self.df[variable_grupo].dropna().unique())
        self._log(f"Grupos: {grupos}")
        
        if len(grupos) < 2:
            self._log("Se necesitan al menos 2 grupos", tipo='advertencia')
            return
        
        # Análisis de autovalores por grupo
        self._invarianza_autovalores(grupos, variable_grupo)
        
        # Invarianza formal con semopy si hay modelo unifactorial
        if self.modelo_unifactorial is not None and SEMOPY_DISPONIBLE:
            self._invarianza_formal_semopy(grupos, variable_grupo)

    def _invarianza_autovalores(self, grupos, variable_grupo):
        """Comparación de autovalores entre grupos."""
        print("\n  COMPARACIÓN DE AUTO-VALORES POR GRUPO:")
        print("  " + "-" * 60)
        
        resultados = {}
        for grupo in grupos:
            subset = self.df[self.df[variable_grupo] == grupo][self.items_cols].dropna()
            if len(subset) >= 10:
                corr = subset.corr().values
                eig = np.linalg.eigvals(corr)
                resultados[f'Grupo_{grupo}'] = np.sort(np.abs(eig))[::-1]
                print(f"    Grupo {grupo} (n={len(subset)}): {[f'{v:.2f}' for v in resultados[f'Grupo_{grupo}'][:3]]}")
        
        if len(resultados) >= 2:
            keys = list(resultados.keys())
            min_len = min(len(resultados[keys[0]]), len(resultados[keys[1]]))
            
            if min_len >= 2:
                sim = np.corrcoef(resultados[keys[0]][:min_len], resultados[keys[1]][:min_len])[0, 1]
                print(f"\n  Similaridad estructural: r = {sim:.4f}")
                
                if sim > 0.95:
                    print("  ✅ INVARIANZA CONFIGURAL RESPALDADA")
                elif sim > 0.90:
                    print("  ✅ INVARIANZA CONFIGURAL ACEPTABLE")
                else:
                    print("  ⚠️  INVARIANZA CONFIGURAL DUDOSA")
            else:
                print(f"\n  ⚠️  Solo {min_len} autovalor(es) comparable(s)")
                if min_len == 1:
                    diff = abs(resultados[keys[0]][0] - resultados[keys[1]][0])
                    print(f"  Diferencia 1er autovalor: {diff:.3f}")
                    if diff < 0.5:
                        print("  ✅ Estructura similar (diferencia pequeña)")
                    else:
                        print("  ⚠️  Diferencia notable en el primer autovalor")

    def _invarianza_formal_semopy(self, grupos, variable_grupo):
        """Invarianza configural, métrica, escalar, estricta con semopy."""
        print("\n" + "="*70)
        print("📐 INVARIANZA DE MEDIDA FORMAL (SEM multi-grupo)")
        print("="*70)
        
        try:
            # Preparar datos por grupo
            datos_por_grupo = {}
            for grupo in grupos:
                mask = self.df[variable_grupo] == grupo
                datos_grupo = self.df[mask][self.items_cols].dropna()
                datos_por_grupo[grupo] = datos_grupo
                print(f"  Grupo {grupo}: {len(datos_grupo)} casos")
            
            # Modelo base configural (sin restricciones)
            print(f"\n  📌 1. INVARIANZA CONFIGURAL (misma estructura)")
            print(f"  " + "-" * 50)
            
            model_config = "# Modelo Configural\n"
            model_config += "F1 =~ " + " + ".join(self.items_cols) + "\n"
            
            # Ajustar modelo a cada grupo por separado
            ajustes_grupo = {}
            for grupo, datos in datos_por_grupo.items():
                try:
                    m = Model(model_config)
                    m.fit(datos)
                    stats_m = calc_stats(m)
                    ajustes_grupo[grupo] = {
                        'chi2': stats_m['chi2'].iloc[0],
                        'dof': stats_m['DoF'].iloc[0],
                        'CFI': stats_m['CFI'].iloc[0],
                        'RMSEA': stats_m['RMSEA'].iloc[0]
                    }
                    print(f"    Grupo {grupo}: χ²({int(stats_m['DoF'].iloc[0])})={stats_m['chi2'].iloc[0]:.2f}, "
                          f"CFI={stats_m['CFI'].iloc[0]:.4f}, RMSEA={stats_m['RMSEA'].iloc[0]:.4f}")
                except Exception as e:
                    print(f"    Grupo {grupo}: Error - {e}")
            
            # Chi2 combinado
            chi2_config = sum(v['chi2'] for v in ajustes_grupo.values())
            dof_config = sum(v['dof'] for v in ajustes_grupo.values())
            print(f"\n    Total configural: χ²({dof_config}) = {chi2_config:.2f}")
            
            # Nota sobre limitaciones
            print(f"\n  ℹ️  NOTA: La invarianza métrica, escalar y estricta requieren")
            print(f"     imponer restricciones de igualdad entre grupos.")
            print(f"     semopy tiene soporte limitado para modelos multi-grupo con restricciones.")
            print(f"     Para análisis completos, considere usar lavaan (R) o Mplus.")
            print(f"     La evidencia configural aquí presentada respalda la misma estructura")
            print(f"     factorial entre grupos como paso inicial necesario.")
            
            # Guardar
            self.resultados['invarianza'] = {
                'configural': ajustes_grupo,
                'chi2_configural': chi2_config,
                'dof_configural': dof_config
            }
            
        except Exception as e:
            self._log(f"Error en invarianza formal: {e}", tipo='advertencia')

    # ========================================================================
    # PASO 4: BOOTSTRAP MEJORADO
    # ========================================================================

    def paso_4_validacion_cruzada(self):
        """Bootstrap con cargas factoriales (no solo autovalor)."""
        self._log("\n" + "="*80, tipo='info')
        self._log("PASO 4: VALIDACIÓN CRUZADA (BOOTSTRAP)", tipo='info')
        self._log("="*80)
        
        print(f"\n  Generando {CONFIG['N_BOOTSTRAP']} muestras bootstrap...")
        
        # Almacenar
        boot_autovalores = []
        boot_cargas = {item: [] for item in self.items_cols}
        
        n = len(self.datos_limpios)
        items_array = self.datos_limpios.values
        
        for i in range(CONFIG['N_BOOTSTRAP']):
            if (i+1) % 100 == 0:
                print(f"    Progreso: {i+1}/{CONFIG['N_BOOTSTRAP']}", end='\r')
            
            idx = np.random.choice(n, n, replace=True)
            muestra = items_array[idx]
            
            # Autovalor
            corr = np.corrcoef(muestra.T)
            try:
                eig = np.linalg.eigvals(corr)
                boot_autovalores.append(np.sort(np.abs(eig))[::-1][0])
            except:
                continue
            
            # Cargas factoriales (extracción rápida con PCA)
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                pca.fit(muestra)
                cargas = np.abs(pca.components_[0])
                for j, item in enumerate(self.items_cols):
                    if j < len(cargas):
                        boot_cargas[item].append(cargas[j])
            except:
                pass
        
        # Resultados autovalor
        if boot_autovalores:
            print(f"\n    Completado: {len(boot_autovalores)} muestras válidas")
            
            media_av = np.mean(boot_autovalores)
            se_av = np.std(boot_autovalores)
            
            print("\n  RESULTADOS BOOTSTRAP (1er autovalor):")
            print("  " + "-" * 50)
            print(f"    Media: {media_av:.4f}")
            print(f"    Error estándar: {se_av:.4f}")
            print(f"    IC 95%: [{np.percentile(boot_autovalores, 2.5):.4f}, {np.percentile(boot_autovalores, 97.5):.4f}]")
            
            cv_av = (se_av / media_av) * 100 if media_av > 0 else 0
            print(f"    Coef. variación: {cv_av:.2f}%")
            print(f"    → {'Excelente' if cv_av < 15 else 'Buena' if cv_av < 25 else 'Moderada'} estabilidad")
        
        # Resultados cargas factoriales
        print(f"\n  RESULTADOS BOOTSTRAP (Cargas factoriales):")
        print("  " + "-" * 60)
        print(f"  {'Ítem':<10} {'Carga media':>10} {'EE':>10} {'CV%':>8} {'IC 95%'}")
        print("  " + "-" * 60)
        
        resumen_cargas = {}
        for item, valores in boot_cargas.items():
            if len(valores) >= 50:
                media_c = np.mean(valores)
                se_c = np.std(valores)
                cv_c = (se_c / media_c) * 100 if media_c > 0 else 0
                ic_lo = np.percentile(valores, 2.5)
                ic_hi = np.percentile(valores, 97.5)
                
                print(f"  {item:<10} {media_c:>10.4f} {se_c:>10.4f} {cv_c:>8.2f} [{ic_lo:.4f}, {ic_hi:.4f}]")
                
                resumen_cargas[item] = {
                    'media': media_c, 'se': se_c, 'cv': cv_c,
                    'IC_95_lo': ic_lo, 'IC_95_hi': ic_hi
                }
        
        self.resultados['bootstrap'] = {
            'autovalor': {'media': media_av, 'se': se_av, 'cv': cv_av},
            'cargas': resumen_cargas
        }

    # ========================================================================
    # PASO 5: REPORTES
    # ========================================================================

    def paso_5_reportes(self):
        self._log("\n" + "="*80, tipo='info')
        self._log("PASO 5: GENERACIÓN DE REPORTES", tipo='info')
        self._log("="*80)
        
        self._generar_graficas()
        self._generar_reporte_texto()
        
        print("\n" + "="*80)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print(f"📁 Resultados guardados en: {CONFIG['DIRECTORIO_SALIDA']}")
        print("="*80)

    def _generar_graficas(self):
        if not VISUALIZACION_DISPONIBLE:
            return
        
        if self.autovalores is not None:
            plt.figure(figsize=(10, 6))
            n_items = len(self.items_cols)
            plt.plot(range(1, n_items+1), self.autovalores[:n_items], 
                    'bo-', linewidth=2, markersize=8)
            plt.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Kaiser (λ=1)')
            plt.axvline(x=self.n_factores + 0.5, color='green', linestyle=':', 
                       linewidth=2, label=f'{self.n_factores} factor(es)')
            plt.xlabel('Número de Factor', fontsize=12)
            plt.ylabel('Autovalor', fontsize=12)
            plt.title(f'Gráfico de Sedimentación - {self.n_factores} factor(es)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            self._guardar_figura('scree_plot')
        
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.datos_limpios.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                       cbar_kws={'label': 'Correlación'}, vmin=-1, vmax=1)
            plt.title('Matriz de Correlaciones entre Ítems', fontsize=14)
            self._guardar_figura('matriz_correlaciones')
        except:
            pass

    def _guardar_figura(self, nombre):
        if not VISUALIZACION_DISPONIBLE:
            return
        timestamp = datetime.now().strftime(CONFIG['FORMATO_FECHA'])
        filename = f"{nombre}_{timestamp}.png"
        ruta = os.path.join(CONFIG['DIRECTORIO_SALIDA'], filename)
        try:
            plt.savefig(ruta, dpi=150, bbox_inches='tight')
            plt.close()
            self._log(f"Gráfica: {filename}", tipo='exito')
        except:
            pass

    def _generar_reporte_texto(self):
        contenido = []
        contenido.append("="*80)
        contenido.append("REPORTE DE ANÁLISIS PSICOMÉTRICO (v4.0) \n  BUAP 2026. Enrique Buendia L.\n")
        contenido.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        contenido.append("="*80)
        contenido.append("")
        
        contenido.append("1. DETECCIÓN DE FACTORES (Criterio de Kaiser)")
        contenido.append("-"*50)
        contenido.append(f"  Número de factores: {self.n_factores}")
        contenido.append(f"  Autovalores: {', '.join([f'{v:.3f}' for v in self.autovalores])}")
        contenido.append(f"  Varianza explicada (total): {np.sum(self.varianza_explicada[:self.n_factores]):.2f}%")
        contenido.append("")
        
        contenido.append("2. CONFIABILIDAD")
        contenido.append("-"*50)
        if 'confiabilidad' in self.resultados:
            conf = self.resultados['confiabilidad']
            contenido.append(f"  α de Cronbach: {conf['alpha_cronbach']:.4f}")
            if conf.get('omega_mcdonald_total'):
                contenido.append(f"  ω de McDonald (total): {conf['omega_mcdonald_total']:.4f}")
        contenido.append("")
        
        for modelo in ['unifactorial', 'multifactorial']:
            clave = f'ajuste_{modelo}'
            if clave in self.resultados:
                contenido.append(f"3. ÍNDICES DE AJUSTE - {modelo.upper()}")
                contenido.append("-"*50)
                for k, v in self.resultados[clave].items():
                    if isinstance(v, (int, float)) and v is not None:
                        contenido.append(f"  {k}: {v:.4f}")
                contenido.append("")
        
        contenido.append("="*80)
        contenido.append("NOTAS INTERPRETATIVAS:")
        n_items = len(self.items_cols)
        gl = n_items * (n_items - 1) // 2 - n_items
        if gl <= 2:
            contenido.append(f"  ⚠️  Modelo con solo {gl} gl (casi saturado).")
            contenido.append(f"  Los índices CFI/TLI pueden estar inflados artificialmente.")
            contenido.append(f"  Valores > 1.0 son matemáticamente imposibles en modelos no saturados.")
        contenido.append("="*80)
        
        timestamp = datetime.now().strftime(CONFIG['FORMATO_FECHA'])
        ruta = os.path.join(CONFIG['DIRECTORIO_SALIDA'], f"reporte_{timestamp}.txt")
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write('\n'.join(contenido))
        self._log(f"Reporte: reporte_{timestamp}.txt", tipo='exito')

    def paso_validez_externa(self, vars_convergentes=None, vars_discriminantes=None):
        """
        Evalúa la validez convergente y discriminante del factor Bienestar
        con variables externas (deben existir como columnas en el CSV).
        """
        if not vars_convergentes and not vars_discriminantes:
            self._log("No se especificaron variables externas. Omitiendo validez convergente/discriminante.",
                      tipo='advertencia')
            return

        if self.datos_limpios is None or len(self.datos_limpios) == 0:
            self._log("No hay datos de ítems válidos. Ejecute primero el Paso 1.", tipo='error')
            return

        print("\n" + "=" * 70)
        print("🔗 VALIDEZ CONVERGENTE Y DISCRIMINANTE")
        print("=" * 70)

        # 1. Puntuación factorial (PCA) sobre los ítems, conservando el índice original
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        score_array = pca.fit_transform(self.datos_limpios.values).flatten()
        score_bienestar = pd.Series(score_array, index=self.datos_limpios.index, name='bienestar')

        # 2. Dataframe base con todas las variables externas y el factor
        #    Unimos por índice para evitar cualquier desalineación
        df_ext = self.df.copy()
        # Aseguramos que el índice coincida; si no, se perderán filas no comunes.
        df_ext = df_ext.join(score_bienestar, how='inner')  # solo filas con ítems completos

        # Si después del join nos quedamos sin datos, salimos
        if len(df_ext) == 0:
            self._log("Tras alinear los datos no quedan casos. Revise los índices.", tipo='error')
            return

        print(f"\n  📌 Puntuación factorial (Bienestar) extraída por PCA.")
        print(f"  Muestra alineada: {len(df_ext)} casos.")
        print(f"  Criterios:")
        print(f"    • Convergente: r ≥ 0.50 y p < 0.05 → ✅")
        print(f"    • Discriminante: |r| < 0.30 → ✅")

        # Función interna para evaluar un conjunto de variables
        def evaluar(variables, tipo):
            if not variables:
                return None
            print(f"\n  📊 Variables {tipo}:")
            print(f"  {'Variable':<25} {'r':>8} {'p':>8} {'Conclusión'}")
            print(f"  " + "-" * 55)
            resultados = []
            for var in variables:
                if var not in df_ext.columns:
                    self._log(f"Variable '{var}' no encontrada. Se omite.", tipo='advertencia')
                    continue
                # Eliminar filas con NaN en bienestar o en la variable
                sub = df_ext[['bienestar', var]].dropna()
                if len(sub) < 30:
                    self._log(f"Pocos datos para '{var}' (n={len(sub)}). Se omite.", tipo='advertencia')
                    continue
                r, p = stats.pearsonr(sub['bienestar'], sub[var])
                if tipo == "convergentes":
                    ok = r >= 0.50 and p < 0.05
                else:
                    ok = abs(r) < 0.30
                icono = "✅" if ok else "❌"
                print(f"  {var:<25} {r:>8.3f} {p:>8.4f} {icono}")
                resultados.append((var, r, p, ok))
            return resultados

        res_conv = evaluar(vars_convergentes, "convergentes")
        res_disc = evaluar(vars_discriminantes, "discriminantes")

        # 4. Comparación global de magnitudes
        if res_conv and res_disc:
            r_convs = [x[1] for x in res_conv]
            r_discs = [abs(x[1]) for x in res_disc]  # trabajamos con valor absoluto
            if r_convs and r_discs:
                media_conv = np.mean(r_convs)
                media_disc = np.mean(r_discs)
                print(f"\n  🔍 Comparación global:")
                print(f"    Corr. promedio convergentes: {media_conv:.3f}")
                print(f"    |Corr.| promedio discriminantes: {media_disc:.3f}")
                if media_conv > media_disc + 0.2:
                    print(f"    ✅ Las correlaciones convergentes son sustancialmente mayores.")
                else:
                    print(f"    ⚠️  La diferencia entre convergentes y discriminantes no es clara.")

        # Guardar en resultados
        self.resultados['validez_externa'] = {
            'convergente': {var: {'r': r, 'p': p, 'ok': ok} for var, r, p, ok in (res_conv or [])},
            'discriminante': {var: {'r': r, 'p': p, 'ok': ok} for var, r, p, ok in (res_disc or [])},
            'criterio': 'r≥.50 (conv), |r|<.30 (disc)'
        }
        print("\n  ✅ Análisis de validez externa completado.")
# ==============================================================================
# MAIN
# ==============================================================================

def main():
    sistema = SistemaPsicometrico()

    if sistema.paso_1_diagnostico():
        sistema.paso_2_cfa_sem()
        
        # Ajusta los nombres de columna exactos que tengas en tu CSV
        sistema.paso_validez_externa(
            vars_convergentes=['fatiga_total', 'estres_percibido'],   # ejemplos
            vars_discriminantes=['altura', 'peso', 'rasgo_personalidad']  # ejemplos
        )
        
        sistema.paso_3_invarianza(variable_grupo='grupos')
        sistema.paso_4_validacion_cruzada()
        sistema.paso_5_reportes()

if __name__ == "__main__":
    main()

    
"""
Orden Correcto y Completo para Validación
1 Diagnóstico completo de datos (Paso OBLIGATORIO al inicio)
* Por qué: Limpieza de datos, valores perdidos (missing), normalidad, outliers, y prueba de adecuación muestral (KMO y Bartlett). Si tus datos son "basura", el CFA fallará.
2 Análisis Factorial Confirmatorio (CFA) - Unifactorial y Multifactorial
* Por qué: Valida la Estructura Interna. Aquí pruebas si tu modelo teórico (ej. "Bienestar =~ i1+i2...") ajusta a los datos. Se calculan CFI, RMSEA, cargas factoriales.
* Nota: Aquí es donde calculas la Fiabilidad (Alfa y Omega).
3 Invarianza de Medida (Configural, Métrica, Escalar)
Por qué: Valida la Equivalencia. Sirve para demostrar que el cuestionario mide lo mismo en diferentes grupos (ej. Hombres vs. Mujeres, o Pre-test vs. Post-test).
Ubicación: Después de confirmar que el modelo ajusta bien en general (paso 2), verificas si ese ajuste se mantiene al restringir la igualdad entre grupos.
4 Modelos de Ecuaciones Estructurales (SEM)
* Por qué: Valida la Validez de Criterio/Nomológica. Aquí conectas tu factor latente ("Bienestar") con otras variables externas (ej. "Rendimiento deportivo", "Lesiones") para ver si tu cuestionario predice fenómenos reales.
Ubicación: Al final. Primero aseguras que el instrumento mide bien (CFA), luego pruebas si sirve para explicar cosas (SEM).
5 Validación Cruzada (Bootstrap + Split-Half)
Por qué: Valida la Estabilidad/Robustez. Comprueba que tus resultados no son producto de una casualidad de tu muestra específica.
Ubicación: Como paso de verificación final ("Sanity check").
6 Generación de reportes y visualizaciones
* Por qué: Comunicación. El último paso es recoger todo lo anterior y generar gráficos y tablas para el artículo o tesis.
"""