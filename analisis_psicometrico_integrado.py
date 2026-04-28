#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SISTEMA INTEGRADO DE ANÁLISIS PSICOMÉTRICO (INVARIANZA CORREGIDA)
================================================================================
Autor: Enrique R.P. Buendia Lozada
Institución: BUAP México
Fecha: Marzo 2026

Correcciones Invarianza:
  - Filtrado automático para mantener solo grupos 0 y 1 (Binario).
  - Manejo de errores (Try/Except) para evitar que un grupo pequeño colapse el sistema.
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
    from semopy import Model
    SEMOPY_DISPONIBLE = True
except ImportError:
    SEMOPY_DISPONIBLE = False
    print("⚠️  semopy no disponible.")

try:
    from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FACTOR_ANALYZER_DISPONIBLE = True
except ImportError:
    FACTOR_ANALYZER_DISPONIBLE = False

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================

CONFIG = {
    'N_BOOTSTRAP': 1000, 
    'ARCHIVO_DATOS': 'datos_invar.csv', 
    'DIRECTORIO_SALIDA': 'resultados_psicometricos',
    'FORMATO_FECHA': '%Y%m%d_%H%M%S'
}

# ==============================================================================
# CLASE PRINCIPAL
# ==============================================================================

class SistemaPsicometrico:
    """Sistema integrado para análisis psicométrico."""

    def __init__(self, ruta_csv=None):
        self.ruta_csv = ruta_csv
        self.df = None
        self.datos_limpios = None
        self.items_cols = None
        self.estructura_factores = None
        self.resultados = {}
        self.modelo_semopy = None
        self.historial = []
        
        self._verificar_directorio()

    def _verificar_directorio(self):
        directorio = CONFIG['DIRECTORIO_SALIDA'] 
        if not os.path.exists(directorio):
            try:
                os.makedirs(directorio)
                print(f"✓ Directorio creado: {directorio}")
            except Exception as e:
                print(f"❌ Error crítico creando directorio '{directorio}': {e}")
                sys.exit(1)
        else:
            print(f"✓ Directorio verificado: {directorio}")

    def _log(self, mensaje, end='\n'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        msg = f"[{timestamp}] {mensaje}"
        if end == '\n':
            self.historial.append(msg)
        print(msg, end=end)

    # ==========================================================================
    # LECTOR INTELIGENTE
    # ==========================================================================

    def cargar_datos_inteligente(self, ruta_csv):
        if not os.path.isfile(ruta_csv):
            self._log(f"❌ Archivo no encontrado: {ruta_csv}")
            return False

        try:
            lista_filas = []
            encabezados = None

            with open(ruta_csv, 'r', encoding='utf-8') as f:
                for linea in f:
                    linea = linea.strip()
                    if not linea: continue

                    partes_barra = linea.split('|')
                    contenido = partes_barra[-1].strip()
                    columnas = [c.strip() for c in contenido.split(';')]

                    if encabezados is None:
                        encabezados = columnas
                    else:
                        if len(columnas) == len(encabezados):
                            lista_filas.append(columnas)

            df = pd.DataFrame(lista_filas, columns=encabezados)
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()

            self._log(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            self._log(f"  Valores únicos en 'sexo': {df['sexo'].unique() if 'sexo' in df.columns else 'N/A'}")
            
            self.df = df
            return True

        except Exception as e:
            self._log(f"❌ Error crítico en lectura: {e}")
            return False

    # ==========================================================================
    # GUARDADO
    # ==========================================================================

    def _guardar_figura_segura(self, nombre_base):
        if not VISUALIZACION_DISPONIBLE: return None
        self._verificar_directorio()
        timestamp = datetime.now().strftime(CONFIG['FORMATO_FECHA'])
        filename = f"{nombre_base}_{timestamp}.png"
        directorio = CONFIG['DIRECTORIO_SALIDA']
        ruta = os.path.join(directorio, filename)
        try:
            plt.savefig(ruta, dpi=150, bbox_inches='tight')
            plt.close()
            self._log(f"✓ Gráfica guardada: {filename}")
            return ruta
        except Exception as e:
            self._log(f"⚠ Error guardando gráfica: {e}")
            return None

    def _guardar_csv_seguro(self, nombre_base, df):
        self._verificar_directorio()
        timestamp = datetime.now().strftime(CONFIG['FORMATO_FECHA'])
        filename = f"{nombre_base}_{timestamp}.csv"
        directorio = CONFIG['DIRECTORIO_SALIDA']
        ruta = os.path.join(directorio, filename)
        try:
            df.to_csv(ruta, index=False, sep=';', decimal=',')
            self._log(f"✓ CSV guardado: {filename}")
            return ruta
        except Exception as e:
            self._log(f"⚠ Error guardando CSV: {e}")
            return None

    def _guardar_texto_seguro(self, nombre_base, contenido):
        self._verificar_directorio()
        timestamp = datetime.now().strftime(CONFIG['FORMATO_FECHA'])
        filename = f"{nombre_base}_{timestamp}.txt"
        directorio = CONFIG['DIRECTORIO_SALIDA']
        ruta = os.path.join(directorio, filename)
        try:
            with open(ruta, 'w', encoding='utf-8') as f:
                f.write(contenido)
            self._log(f"✓ Reporte guardado: {filename}")
            return ruta
        except Exception as e:
            self._log(f"⚠ Error guardando reporte: {e}")
            return None

    # ==========================================================================
    # PASO 1: DIAGNÓSTICO
    # ==========================================================================

    def paso_1_diagnostico(self, ruta_csv=None, items_cols=None, estructura_factores=None):
        self._log(f"\n{'='*81}")
        self._log(f"\n           BUAP Enrique R.P. Buendia Lozada 2026\n")
        self._log("PASO 1: DIAGNÓSTICO Y CARGA DE DATOS")
        self._log(f"{'='*70}")

        if ruta_csv:
            success = self.cargar_datos_inteligente(ruta_csv)
        else:
            success = self.cargar_datos_inteligente(CONFIG['ARCHIVO_DATOS'])
        
        if not success or self.df is None or self.df.empty:
            self._log("❌ No se pudieron cargar datos. Abortando.")
            return False

        self.diagnosticar_csv()
        
        if items_cols:
            self.estructura_factores = estructura_factores
        self.preparar_datos_likert(items_cols, auto_detectar=True)
        
        if self.datos_limpios is None or self.datos_limpios.empty:
            return False

        self.pruebas_preliminares()
        self.generar_visualizaciones(tipo='diagnostico')
        return True

    def diagnosticar_csv(self):
        df = self.df
        print(f"\n📊 Dimensiones: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe().T
            desc['asimetria'] = df[numeric_cols].skew()
            print("\n--- Estadísticas Descriptivas ---")
            print(desc.round(2))
            
            info = {'filas': df.shape[0], 'columnas': df.shape[1], 'numericas': numeric_cols}
            self.resultados['diagnostico'] = info
        return info

    def preparar_datos_likert(self, items_cols, escala_min=1, escala_max=5, auto_detectar=False):
        if not items_cols and auto_detectar:
            exclude = ['id', 'sexo', 'grupo', 'edad']
            candidates = [c for c in self.df.select_dtypes(include=[np.number]).columns 
                          if c.lower() not in exclude]
            items_cols = [c for c in candidates if c.startswith('i')]

        if not items_cols:
            self._log("❌ No se detectaron ítems.")
            return None

        validos = [c for c in items_cols if c in self.df.columns]
        datos = self.df[validos].dropna()
        
        self._log(f"✓ Datos preparados: {len(datos)} casos, {len(validos)} ítems")
        self.datos_limpios = datos
        self.items_cols = validos
        return datos

    def pruebas_preliminares(self):
        datos = self.datos_limpios
        res = {}
        if FACTOR_ANALYZER_DISPONIBLE:
            try:
                kmo_all, kmo_model = calculate_kmo(datos)
                chi2, p_val = calculate_bartlett_sphericity(datos)
                res['kmo'] = kmo_model
                res['bartlett_p'] = p_val
                print(f"\n--- Pruebas Preliminares ---")
                print(f"KMO: {kmo_model:.3f}")
                print(f"Bartlett: p={p_val:.5f} (esferificidad)\n")
            except Exception as e:
                self._log(f"Error en pruebas preliminares: {e}")
        self.resultados['preliminares'] = res

    # ==========================================================================
    # PASO 2: CFA / SEM
    # ==========================================================================

    def paso_2_cfa_sem(self, estructura_factores=None):
        if self.datos_limpios is None:
            self._log("❌ ERROR: Ejecute primero el Paso 1.")
            return

        self._log(f"\n{'='*70}")
        self._log("PASO 2: CFA & SEM")
        self._log(f"{'='*70}")

        estructura = estructura_factores or self.estructura_factores
        if not estructura:
            estructura = {'Bienestar': self.items_cols}
            self.estructura_factores = estructura

        if SEMOPY_DISPONIBLE:
            self.estimar_cfa_semopy(estructura)
            if self.modelo_semopy:
                self.calcular_ajuste_semopy(self.modelo_semopy)  # CORRECCIÓN
                self.calcular_confiabilidad(estructura_factores=estructura)
            
            # --- AGREGADO: SEM ESTRUCTURAL ---
            self.estimar_sem_estructural()
        else:
            self.estimar_cfa_manual(estructura)

        self.generar_visualizaciones(tipo='cfa')

    def crear_modelo_cfa(self, estructura):
        desc = "# Modelo CFA\n"
        for factor, items in estructura.items():
            validos = [i for i in items if i in self.items_cols]
            desc += f"{factor} =~ {' + '.join(validos)}\n"
        return desc

    def estimar_cfa_semopy(self, estructura):
        modelo_desc = self.crear_modelo_cfa(estructura)
        try:
            model = Model(modelo_desc)
            model.fit(self.datos_limpios)
            self.modelo_semopy = model
            insp = model.inspect()
            loadings = insp[(insp['op'] == '~')]
            print("\n--- Cargas Factoriales (Semopy) ---")
            print(loadings[['lval', 'rval', 'Estimate', 'p-value']])
            self.resultados['cfa_semopy'] = {'inspeccion': insp}
            return model
        except Exception as e:
            self._log(f"Error CFA Semopy: {e}")
            return None

    def calcular_ajuste_semopy(self, model):
        try:
            stats = semopy.calc_stats(model)

            self._log("\n\n--- Índices de Ajuste (Semopy) ---")

            # convertir a diccionario plano
            valores = stats.to_dict(orient="records")[0]
            for k, v in valores.items():
                self._log(f"{k:20}: {v:.6f}")

            self.resultados['ajuste'] = valores

        except Exception as e:
            self._log(f"Error cálculo ajuste: {e}")





    def calcular_confiabilidad(self, estructura_factores):
        datos = self.datos_limpios[self.items_cols]
        item_vars = datos.var(ddof=1)
        total_var = datos.sum(axis=1).var(ddof=1)
        alfa = (len(datos.columns) / (len(datos.columns)-1)) * (1 - (item_vars.sum()/total_var))
        print(f"\n--- Confiabilidad ---")
        print(f"Alfa de Cronbach: {alfa:.3f}")

        omega = 0
        if self.modelo_semopy:
            insp = self.modelo_semopy.inspect(std_est=True)
            params = insp[(insp['rval'] == list(estructura_factores.keys())[0]) & (insp['op'] == '~')]
            if not params.empty:
                col = 'Std. Est' if 'Std. Est' in params.columns else 'Estimate'
                loadings = params[col].values
                error_var = 1 - loadings**2
                sum_l = loadings.sum()
                sum_e = error_var.sum()
                omega = (sum_l**2) / ((sum_l**2) + sum_e)
                print(f"Omega de McDonald: {omega:.3f}")

        self.resultados['confiabilidad'] = {'alfa': alfa, 'omega': omega}

    def estimar_cfa_manual(self, estructura):
        datos = self.datos_limpios[self.items_cols]
        R = datos.corr().values
        autovalores, autovectores = np.linalg.eigh(R)
        idx = np.argsort(autovalores)[::-1]
        loadings = autovectores[:, idx[0]] * np.sqrt(autovalores[idx[0]])
        print("\n--- CFA Manual ---")
        for i, item in enumerate(self.items_cols):
            print(f"{item}: {loadings[i]:.3f}")
        self.resultados['cfa_manual'] = {'loadings': loadings}

    # --- NUEVO MÉTODO PARA SEM ESTRUCTURAL ---
    def estimar_sem_estructural(self):
        """
        Realiza un SEM Estructural (Path Analysis) prediciendo variables dependientes
        a partir del constructo latente calculado.
        """
        self._log(f"\n{'='*70}")
        self._log("PASO 2.5: SEM ESTRUCTURAL (PATH ANALYSIS)")
        self._log(f"{'='*70}")

        if self.df is None:
            self._log("❌ Error: DataFrame original no disponible para SEM estructural.")
            return

        # 1. Preparación de la variable latente observada
        # Usamos el dataframe original para incluir 'sexo'
        items = ['i1', 'i2', 'i3', 'i4', 'i5']
        # Verificamos que existan las columnas
        if not all(col in self.df.columns for col in items + ['sexo']):
            self._log("❌ Faltan columnas necesarias (ítems o sexo) para SEM estructural.")
            return

        self.df['Bienestar'] = self.df[items].mean(axis=1)

        # 2. Definición del Modelo SEM Estructural
        # Syntax de semopy: ~ indica regresión
        # La variable latente observada 'Bienestar' predice a las variables observadas dependientes.
        model_desc = """
            # Ecuaciones Estructurales
            sexo ~ Bienestar
            i1 ~ Bienestar
            i2 ~ Bienestar
            i3 ~ Bienestar
            i4 ~ Bienestar
            i5 ~ Bienestar
        """

        try:
            model = Model(model_desc)
            results = model.fit(self.df)
            
            # 3. Inspección de Resultados (Estadísticas de ajuste y parámetros)
            print("\n--- Resultados del SEM Estructural ---")
            insp = model.inspect()
            print(insp)
            
            # Guardamos los resultados del SEM estructural
            # self.resultados['sem_estructural'] = {'\n\ninspeccion': insp, 'ajuste': semopy.calc_stats(model)}
            self.resultados['sem_estructural'] = {
                'inspeccion': insp,                # guardamos el DataFrame original
                'ajuste': semopy.calc_stats(model) # también es DataFrame
            }

            # 4. Validación Cruzada con Bootstrap
            print("\n--- Resultados del Bootstrap (n=1000) ---")
            # Nota: semopy.run_bootstrap devuelve las estadísticas de los modelos remuestreados
            estimates = semopy.calc_stats(model)
            bootstrap_estimates = semopy.run_bootstrap(model, n_samples=1000)

            # Mostramos un resumen de los parámetros con Bootstrap (Media y SE)
            if 'parameters' in bootstrap_estimates:
                params_df = pd.DataFrame(bootstrap_estimates['parameters'])
                print(params_df[['lval', 'op', 'rval', 'Estimate', 'SE', 'p-value']].head(10))

            # Información sobre el ajuste del modelo estructural
            print("\n\n--- Índices de Bondad de Ajuste (SEM Estructural) ---")
            stats = semopy.calc_stats(model)
            for k, v in stats.items():
                if isinstance(v, (float, int)):
                    print(f"{k}: {v:.4f}")
            
            self._log("✓ SEM Estructural completado exitosamente.")

        except Exception as e:
            self._log(f"❌ Error en SEM Estructural: {e}")

    # ==========================================================================
    # PASO 3: INVARIANZA (CORREGIDA Y ROBUSTA)
    # ==========================================================================

    def paso_3_invarianza(self, variable_grupo='sexo'):
        if self.datos_limpios is None:
            self._log("❌ ERROR: Ejecute primero el Paso 1.")
            return

        self._log(f"\n{'='*70}")
        self._log("PASO 3: INVARIANZA DE MEDIDA")
        self._log(f"{'='*70}")

        if variable_grupo not in self.df.columns:
             self._log(f"❌ Variable '{variable_grupo}' no encontrada en df.")
             return

        # Filtrar df original para mantener grupo
        datos_invar = self.df.dropna(subset=self.items_cols)
        
        # Verificar valores únicos y filtrar automáticamente si hay más de 2 (ej: 0, 1, 2)
        valores_unicos = sorted(datos_invar[variable_grupo].unique())
        self._log(f"Valores únicos detectados en '{variable_grupo}': {valores_unicos}")

        # Lógica de filtrado para sexo binario
        if len(valores_unicos) > 2 and variable_grupo.lower() == 'sexo':
            self._log("⚠️ Se detectaron más de 2 grupos en una variable de 'sexo'.")
            self._log(f"   Filtrando para usar solo: {valores_unicos[:2]}")
            valores_usar = valores_unicos[:2]
            datos_invar = datos_invar[datos_invar[variable_grupo].isin(valores_usar)]
        else:
            valores_usar = valores_unicos

        self.analisis_invarianza(datos=datos_invar, items_cols=self.items_cols, variable_grupo=variable_grupo, valores_grupo=valores_usar)

    def analisis_invarianza(self, datos, items_cols, variable_grupo, valores_grupo):
        if variable_grupo not in datos.columns:
            self._log("Error: Variable grupo no encontrada.")
            return

        self._log(f"Analizando invarianza entre grupos: {valores_grupo}")
        resultados = {}
        
        for val in valores_grupo:
            subset = datos[datos[variable_grupo] == val][items_cols].dropna()
            
            if len(subset) < 3:
                self._log(f"⚠ Grupo {val}: Insuficientes datos ({len(subset)} casos). Saltando.")
                continue

            try:
                R = subset.corr().values
                # Verificar si la matriz es válida para calcular eigenvalores
                if np.isnan(R).any() or np.isinf(R).any():
                     self._log(f"⚠ Grupo {val}: Matriz de correlación inválida (NaN/Inf). Saltando.")
                     continue

                eig, vec = np.linalg.eigh(R)
                loadings = vec[:, np.argmax(eig)] * np.sqrt(np.max(eig))
                resultados[f'Grupo_{val}'] = loadings
            except Exception as e:
                self._log(f"⚠ Grupo {val}: Error calculando cargas ({e}). Saltando.")
                continue
        
        if len(resultados) > 0:
            self.resultados['invarianza'] = resultados
            self._log("Análisis de invarianza completado con éxito parcial.")
        else:
            self._log("❌ No se pudo calcular invarianza para ningún grupo.")

    # ==========================================================================
    # PASO 4: VALIDACIÓN CRUZADA
    # ==========================================================================

    def paso_4_validacion_cruzada(self):
        if self.datos_limpios is None:
            self._log("❌ ERROR: Ejecute primero el Paso 1.")
            return

        self._log(f"\n{'='*70}")
        self._log("PASO 4: VALIDACIÓN CRUZADA")
        self._log(f"{'='*70}")

        self.validacion_cruzada(
            datos=self.datos_limpios,
            items_cols=self.items_cols,
            n_bootstrap=CONFIG['N_BOOTSTRAP']
        )

    def validacion_cruzada(self, datos, items_cols, n_bootstrap):
        self._log("Iniciando Bootstrap...")
        boot_cargas = []
        n = len(datos)
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            muestra = datos.iloc[idx][items_cols]
            R = muestra.corr().values
            eig, vec = np.linalg.eigh(R)
            if len(eig) > 0:
                load = vec[:, np.argmax(eig)] * np.sqrt(np.max(eig))
                boot_cargas.append(load)
        
        if len(boot_cargas) > 0:
            boot_cargas = np.array(boot_cargas)
            media_boot = np.mean(boot_cargas, axis=0)
            se_boot = np.std(boot_cargas, axis=0, ddof=1)
            
            print("\n--- Resultados Bootstrap ---")
            for i, item in enumerate(items_cols):
                print(f"{item}: Media={media_boot[i]:.3f}, SE={se_boot[i]:.3f}")
            
            self.resultados['validacion_cruzada'] = {'media': media_boot, 'se': se_boot}
        else:
            self._log("⚠ No se generaron muestras bootstrap válidas.")

    # ==========================================================================
    # PASO 5: REPORTES
    # ==========================================================================

    def paso_5_reportes(self):
        self._log(f"\n{'='*70}")
        self._log("PASO 5: GENERACIÓN DE REPORTES")
        self._log(f"{'='*70}")

        self.generar_visualizaciones(tipo='final')
        self.generar_reporte_texto()
        self.exportar_resultados_csv()

    def generar_visualizaciones(self, tipo='final'):
        if not VISUALIZACION_DISPONIBLE: return
        
        # Matriz de correlaciones
        if tipo == 'diagnostico' or tipo == 'final':
            try:
                plt.figure(figsize=(10,8))
                sns.heatmap(self.datos_limpios.corr(), annot=True, cmap='coolwarm')
                plt.title('Figura 1. Matriz de Correlaciones')
                self._guardar_figura_segura('correlaciones')
            except: pass

        # Scree plot
        if tipo == 'cfa' or tipo == 'final':
            try:
                R = self.datos_limpios.corr().values
                eig = np.linalg.eigvals(R)
                plt.figure(figsize=(8,5))
                plt.plot(range(1, len(eig)+1), sorted(eig)[::-1], 'bo-')
                plt.axhline(y=1, color='r', linestyle='--')
                plt.title('Figura 2. Scree Plot de autovalores')
                self._guardar_figura_segura('scree_plot')
            except: pass

        # Índices de ajuste del modelo SEM
        if 'ajuste' in self.resultados:
            try:
                df = pd.DataFrame.from_dict(self.resultados['ajuste'], orient='index', columns=['Valor'])
                plt.figure(figsize=(8,5))
                df.plot(kind='bar', legend=False)
                plt.title('Figura 3. Índices de Ajuste del Modelo SEM')
                plt.ylabel('Valor')
                self._guardar_figura_segura('indices_ajuste')
            except: pass

        # Confiabilidad interna (Alfa y Omega)
        if 'confiabilidad' in self.resultados:
            try:
                conf = self.resultados['confiabilidad']
                df = pd.DataFrame(list(conf.items()), columns=['Coeficiente','Valor'])
                plt.figure(figsize=(6,4))
                sns.barplot(x='Coeficiente', y='Valor', data=df)
                plt.title('Figura 4. Confiabilidad Interna (α y ω)')
                plt.ylim(0,1)
                self._guardar_figura_segura('confiabilidad')
            except: pass

        # Invarianza de medida
        if 'invarianza' in self.resultados:
            try:
                df = pd.DataFrame(self.resultados['invarianza'])
                plt.figure(figsize=(8,5))
                sns.heatmap(df, annot=True, cmap='viridis')
                plt.title('Figura 5. Cargas factoriales por grupo (Invarianza)')
                self._guardar_figura_segura('invarianza')
            except: pass

        # Validación cruzada (Bootstrap)
        if 'validacion_cruzada' in self.resultados:
            try:
                media = self.resultados['validacion_cruzada']['media']
                se = self.resultados['validacion_cruzada']['se']
                items = self.items_cols
                df = pd.DataFrame({'Item': items, 'Media': media, 'SE': se})
                plt.figure(figsize=(8,5))
                sns.barplot(x='Item', y='Media', data=df, yerr=se)
                plt.title('Figura 6. Resultados Bootstrap (Media y Error Estándar)')
                self._guardar_figura_segura('bootstrap')
            except: pass

    def generar_reporte_texto(self):
        contenido = "REPORTE PSICOMÉTRICO\n===================\n\n"
        for k, v in self.resultados.items():
            contenido += f"BLOQUE: {k}\n"
            if isinstance(v, dict):
                for subk, subv in v.items():
                    contenido += f"{subk}:\n"
                    if isinstance(subv, pd.DataFrame):
                        contenido += subv.to_string(index=False) + "\n"
                    else:
                        contenido += str(subv) + "\n"
            elif isinstance(v, pd.DataFrame):
                contenido += v.to_string(index=False) + "\n"
            else:
                contenido += str(v) + "\n"
            contenido += "-"*50 + "\n"
        self._guardar_texto_seguro('reporte_completo', contenido)


    def exportar_resultados_csv(self):
        if 'ajuste' in self.resultados:
            df_res = pd.DataFrame.from_dict(self.resultados['ajuste'], orient='index', columns=['Valor'])
            self._guardar_csv_seguro('indices_ajuste', df_res)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    ARCHIVO_DATOS = 'datos_invar.csv'
    ESTRUCTURA_FACTORIAL = {'Bienestar': ['i1', 'i2', 'i3', 'i4', 'i5']}
    LISTA_ITEMS = ['i1', 'i2', 'i3', 'i4', 'i5']
    
    sistema = SistemaPsicometrico(ARCHIVO_DATOS)
    
    print("\n🚀 INICIANDO SISTEMA")
    
    if sistema.paso_1_diagnostico(items_cols=LISTA_ITEMS, estructura_factores=ESTRUCTURA_FACTORIAL):
        sistema.paso_2_cfa_sem(estructura_factores=ESTRUCTURA_FACTORIAL)
        sistema.paso_3_invarianza(variable_grupo='sexo')
        sistema.paso_4_validacion_cruzada()
        sistema.paso_5_reportes()
    
    print("\n✅ PROCESO FINALIZADO.")
    print("Presione Enter para cerrar...")
    input()


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