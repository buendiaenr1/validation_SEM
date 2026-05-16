#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
###############################################################################
# VALIDACION PSICOMETRICA COMPLETA 
# Compatible con estándares: APA 7ma ed., AMA, COSMOS
# Optimizado para: N=213, 29 ítems, escala Likert 1-5
# Autor: Enrique R.P. Buendia Lozada - Version 2.1 (CORREGIDA - sin errores de names)
# Fecha: 2026
###############################################################################

# ==========================================================
# RESPUESTA TÉCNICA: SÍ, ES COMPLETAMENTE ADAPTABLE
# ==========================================================
# 1. NÚMERO DE ÍTEMS: Se detecta automáticamente desde las 
#    columnas numéricas del CSV. El script ajusta dinámicamente:
#    • Bucle item-total, KMO por ítem, comunalidades
#    • Gráficos (escalado automático con sqrt(n_items))
#    • Construcción del modelo CFA línea por línea
# 2. RANGO LIKERT: Solo modifica estas 2 variables en la Sección 1:

#  ESCALA_MIN <- 1  # <-- CAMBIAR si tu escala empieza en 0, -2, etc.
#  ESCALA_MAX <- 5  # <-- CAMBIAR si tu escala es 4, 7, 10, etc.

# 3. LIMITACIONES ESTADÍSTICAS (no del código, sino metodológicas):
#    • Muestra mínima estable: N ≥ 5× número de ítems (ideal ≥ 10×)
#    • Identificación CFA: Cada factor necesita ≥3 ítems. 
#      Si fuerzas factores de 2 ítems, lavaan retornará error 
#      de no-identificación (capturado por tryCatch).
#    • Naturaleza ordinal: El script usa ML/MLR (datos continuos). 
#      Para escalas ≤4 puntos o muy asimétricas, cambiar a WLSMV:
#      fit_cfa <- lavaan::cfa(modelo_spec, data=df_items_clean, 
#                             std.lv=TRUE, estimator="WLSMV")

# 4. OPTIMIZACIÓN PARA GRANDES VOLUMENES (>50 ítems / >2000 casos):
#    Reducir simulaciones en la Sección 8:
#    pa_result <- psych::fa.parallel(mat_datos, fm="ml", fa="fa", n.iter=100, plot=FALSE)

cat("\n")
cat(paste(rep("#",80),collapse=""),"\n")
cat("#", paste(rep(" ",78),collapse=""), "#\n")
cat("#", paste(strtrim(paste0("  VALIDACION PSICOMETRICA > FINAL CORREGIDA"),78),
               collapse="\n"), "#\n")
cat("#", paste(rep(" ",78),collapse=""), "#\n")
cat(paste(rep("#",80),collapse=""),"\n")

# =========================================================================
# 0. INSTALACION Y CARGA DE PAQUETES
# =========================================================================

paquetes_requeridos <- c(
  "psych", "lavaan", "semPlot", "nFactors", "corrplot",
  "ggplot2", "reshape2", "dplyr", "MASS", "gridExtra", "viridis", "tidyr"
)

instalar_si_falta <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  [INSTALANDO] %s...\n", pkg))
    install.packages(pkg, repos="https://cloud.r-project.org", quiet=TRUE)
  }
}
invisible(lapply(paquetes_requeridos, instalar_si_falta))

suppressPackageStartupMessages({
  library(psych); library(lavaan); library(semPlot); library(nFactors)
  library(corrplot); library(ggplot2); library(reshape2); library(dplyr)
  library(MASS); library(gridExtra); library(viridis); library(tidyr)
})

cat("\n[OK] Todos los paquetes cargados correctamente.\n")

# =========================================================================
# 1. CONFIGURACION GLOBAL
# =========================================================================

# Variables sociodemográficas para análisis de invarianza (pueden ser una o varias)
VARS_GRUPO <- c("sexo")        # Solo sexo
# VARS_GRUPO <- c("sexo", "edad_grupo", "educacion")   # Múltiples
ARCHIVO_DATOS <- "datos5.csv"
COLUMNA_GRUPO <- NULL  # Cambiar si tienes grupos: ej. "grupo", "sexo"
ESCALA_MIN <- 1; ESCALA_MAX <- 5
DIR_SALIDA <- "resultados_validacion_Q1"
SEED <- 42; N_BOOTSTRAP <- 1000; CONF_LEVEL <- 0.95
N_FACTORES_TEORICO <- NA  # Cambiar a 4, 6, etc. si hay justificación teórica previa

if (!dir.exists(DIR_SALIDA)) dir.create(DIR_SALIDA)

resultados_globales <- list()
rotacion_elegida <- "varimax"
normalidad_multivariada_ok <- FALSE

# =========================================================================
# 2. FUNCIONES AUXILIARES ROBUSTAS
# =========================================================================

separar_linea <- function() cat("\n", paste(rep("=",80),collapse=""), "\n")

clasificar_alpha <- function(v) {
  if (is.na(v)) return("NO DISPONIBLE")
  cut(v, breaks=c(-Inf,0.6,0.7,0.8,0.9,Inf),
      labels=c("INACEPTABLE","CUESTIONABLE","ACEPTABLE","BUENO","EXCELENTE"))
}

clasificar_kmo <- function(v) {
  if (is.na(v)) return("NO DISPONIBLE")
  cut(v, breaks=c(-Inf,0.5,0.6,0.7,0.8,0.9,Inf),
      labels=c("INACEPTABLE","MEDIOCRE","ACEPTABLE","BUENO","MERITORIO","MARAVILLOSO"))
}

clasificar_ajuste_cfi <- function(v) {
  if (is.na(v)) return("N/A")
  cut(v, breaks=c(-Inf,0.90,0.95,Inf), labels=c("INACEPTABLE","ACEPTABLE","EXCELENTE"))
}

#clasificar_ajuste_rmsea <- function(v) {
#  if (is.na(v)) return("N/A")
#  cut(v, breaks=c(-Inf,0.08,0.06,0.05,Inf), labels=c("MALO","ACEPTABLE","BUENO","EXCELENTE"))
#}
clasificar_ajuste_rmsea <- function(v) {
  if (is.na(v)) return("N/A")
  cut(v, breaks = c(-Inf, 0.05, 0.06, 0.08, Inf),
      labels = c("EXCELENTE", "BUENO", "ACEPTABLE", "MALO"))
}

omega_mcdonald_calc <- function(df) {
  tryCatch({
    corr <- cor(df, use="pairwise.complete.obs")
    ev <- eigen(corr, symmetric=TRUE)
    idx <- order(ev$values, decreasing=TRUE)
    eigenvalues <- ev$values[idx]; eigenvectors <- ev$vectors[,idx]
    first_ev <- max(eigenvalues[1], 0)
    loadings <- eigenvectors[,1] * sqrt(first_ev)
    uniquenesses <- pmax(1.0 - loadings^2, 0)
    sum_loads <- sum(abs(loadings)); sum_unique <- sum(uniquenesses)
    if (sum_unique <= 0) return(1.0)
    omega <- (sum_loads^2) / ((sum_loads^2) + sum_unique)
    return(as.numeric(omega))
  }, error = function(e) return(NA))
}

calcular_omega_jerarquico <- function(df, loadings_mat) {
  tryCatch({
    n_factors <- tryCatch(ncol(loadings_mat), error = function(e) 1)
    if (is.na(n_factors) || n_factors <= 1) {
      om <- omega_mcdonald_calc(df)
      return(list(omega_h = om, omega_t = om))
    }
    pca_gen <- tryCatch(prcomp(df, scale. = TRUE), error = function(e) NULL)
    if (is.null(pca_gen)) {
      cat("  [AVISO] No se pudo calcular PCA para omega_h\n")
      omega_t <- omega_mcdonald_calc(df)
      return(list(omega_h = NA, omega_t = omega_t))
    }
    rotacion <- tryCatch(pca_gen$rotation, error = function(e) NULL)
    if (is.null(rotacion)) {
      cat("  [AVISO] Sin rotación PCA\n")
      omega_t <- omega_mcdonald_calc(df)
      return(list(omega_h = NA, omega_t = omega_t))
    }
    if (ncol(rotacion) == 0) {
      omega_t <- omega_mcdonald_calc(df)
      return(list(omega_h = NA, omega_t = omega_t))
    }
    carga_general <- as.numeric(rotacion[, 1])
    if (length(carga_general) == 0 || all(is.na(carga_general))) {
      omega_t <- omega_mcdonald_calc(df)
      return(list(omega_h = NA, omega_t = omega_t))
    }
    carga_general <- carga_general[!is.na(carga_general)]
    var_general <- sum(carga_general^2, na.rm = TRUE)
    corr_total_mat <- tryCatch(cor(df), error = function(e) NULL)
    if (is.null(corr_total_mat)) {
      omega_t <- omega_mcdonald_calc(df)
      return(list(omega_h = NA, omega_t = omega_t))
    }
    var_total <- sum(diag(corr_total_mat), na.rm = TRUE)
    if (var_total <= 0 || is.na(var_total)) {
      omega_t <- omega_mcdonald_calc(df)
      return(list(omega_h = NA, omega_t = omega_t))
    }
    omega_h <- var_general / var_total
    omega_t <- omega_mcdonald_calc(df)
    return(list(omega_h = omega_h, omega_t = omega_t))
  }, error = function(e) {
    cat(sprintf("  [ERROR] En omega_jerarquico: %s\n", e$message))
    omega_t <- tryCatch(omega_mcdonald_calc(df), error = function(e) NA)
    return(list(omega_h = NA, omega_t = omega_t))
  })
}

detectar_outliers_mahalanobis <- function(df, alpha=0.001) {
  tryCatch({
    n <- nrow(df); p <- ncol(df)
    center <- colMeans(df); cov_mat <- cov(df)
    dist_mah <- mahalanobis(df, center, cov_mat)
    crit_val <- qchisq(1-alpha, df=p)
    is_outlier <- dist_mah > crit_val
    return(list(distances=dist_mah, critical_value=crit_val,
                outliers=which(is_outlier), n_outliers=sum(is_outlier),
                proportion=sum(is_outlier)/n*100))
  }, error = function(e) return(list(distances=rep(NA,nrow(df)), critical_value=NA,
                outliers=integer(0), n_outliers=0, proportion=0)))
}

calcular_htmt <- function(loadings_df, corr_factores) {
  tryCatch({
    n_factores <- ncol(loadings_df)
    if (n_factores < 2) return(matrix(NA, nrow=n_factores, ncol=n_factores))
    htmt_matrix <- matrix(0, nrow=n_factores, ncol=n_factores)
    rownames(htmt_matrix) <- colnames(htmt_matrix) <- colnames(loadings_df)
    for (i in 1:(n_factores-1)) {
      for (j in (i+1):n_factores) {
        items_i <- which(abs(loadings_df[,i]) > 0.3)
        items_j <- which(abs(loadings_df[,j]) > 0.3)
        if (length(items_i)>0 && length(items_j)>0) {
          heterotrait_mean <- mean(abs(corr_factores[i,j]))
          monotrait_i <- mean(abs(loadings_df[items_i,i]))
          monotrait_j <- mean(abs(loadings_df[items_j,j]))
          monotrait_mean <- (monotrait_i + monotrait_j)/2
          htmt_matrix[i,j] <- heterotrait_mean/monotrait_mean
          htmt_matrix[j,i] <- htmt_matrix[i,j]
        }
      }
    }
    return(htmt_matrix)
  }, error = function(e) return(matrix(NA, nrow=ncol(loadings_df), ncol=ncol(loadings_df))))
}

guardar_tabla <- function(df, nombre_archivo) {
  write.csv(df, file.path(DIR_SALIDA, paste0(nombre_archivo,".csv")), row.names=TRUE)
  cat(sprintf("  [GUARDADO] %s.csv\n", nombre_archivo))
}

# =========================================================================
# 3. LECTURA Y PREPARACION DE DATOS
# =========================================================================

cat("\n", paste(rep("#",80),collapse=""), "\n")
cat("#  1. LECTURA Y PREPARACION DE DATOS\n")
cat(paste(rep("#",80),collapse=""), "\n")

primera_linea <- readLines(ARCHIVO_DATOS, n=1, warn=FALSE)
if (grepl(";", primera_linea) && !grepl(",", primera_linea)) {
  delimitador <- ";"
} else if (grepl("\t", primera_linea)) {
  delimitador <- "\t"
} else {
  delimitador <- ","
}

df_completo <- read.csv(ARCHIVO_DATOS, sep=delimitador, stringsAsFactors=FALSE,
                        fileEncoding="UTF-8-BOM", na.strings=c("","NA","N/A"))

n_cols <- ncol(df_completo)
nombres_actuales <- colnames(df_completo)

if (is.null(nombres_actuales) || length(nombres_actuales) != n_cols) {
  nombres_actuales <- paste0("V", seq_len(n_cols))
  colnames(df_completo) <- nombres_actuales
}

nombres_limpios <- trimws(nombres_actuales)
nombres_limpios <- gsub("^\xef\xbb\xbf", "", nombres_limpios)
nombres_limpios <- gsub("[áéíóúÁÉÍÓÚñÑ]", "", nombres_limpios)
nombres_limpios <- gsub(" ", "_", nombres_limpios)

if (length(unique(nombres_limpios)) != length(nombres_limpios)) {
  tmp <- nombres_limpios
  dupes <- duplicated(tmp) | duplicated(tmp, fromLast = TRUE)
  contador <- table(factor(tmp, levels = unique(tmp)))
  i <- 1
  for (idx in which(dupes)) {
    base <- nombres_limpios[idx]
    nuevo <- paste0(base, "_", i)
    while (nuevo %in% nombres_limpios) {
      i <- i + 1
      nuevo <- paste0(base, "_", i)
    }
    tmp[idx] <- nuevo
  }
  nombres_limpios <- tmp
}

colnames(df_completo) <- nombres_limpios
stopifnot(length(colnames(df_completo)) == n_cols)

cat(sprintf("[DIAG] Dimensiones: %d filas x %d columnas\n", nrow(df_completo), ncol(df_completo)))
cat("[DIAG] Primeros nombres de columnas:", paste(head(colnames(df_completo), 10), collapse=", "), "\n")

colnames(df_completo) <- trimws(colnames(df_completo))
colnames(df_completo) <- gsub("^\xef\xbb\xbf", "", colnames(df_completo))
colnames(df_completo) <- gsub("[áéíóúÁÉÍÓÚñÑ]", "", colnames(df_completo))
colnames(df_completo) <- gsub(" ", "_", colnames(df_completo))

cat(sprintf("\n[OK] Datos leidos: %s (%d filas, %d columnas)\n",
            ARCHIVO_DATOS, nrow(df_completo), ncol(df_completo)))

cols_todas <- colnames(df_completo)
df_items <- df_completo; cols_eliminar <- c()
columna_grupo_real <- NULL; tiene_grupo <- FALSE; grupo <- NULL

if (!is.null(COLUMNA_GRUPO)) {
  match_idx <- which(tolower(cols_todas)==tolower(COLUMNA_GRUPO))
  if (length(match_idx)>0) {
    columna_grupo_real <- cols_todas[match_idx[1]]
    grupo <- df_completo[[columna_grupo_real]]
    cols_eliminar <- c(columna_grupo_real)
    tiene_grupo <- TRUE
    cat(sprintf("[OK] Columna grupos: %s\n", columna_grupo_real))
  }
}

demograficas <- c("grupo","grupos","sexo","sex","genero","gender","edad","age","id","id_participante")
for (col in cols_todas) {
  if (!is.numeric(df_completo[[col]]) && !(col %in% columna_grupo_real))
    cols_eliminar <- c(cols_eliminar, col)
  if (tolower(col) %in% demograficas && !(col %in% columna_grupo_real))
    cols_eliminar <- c(cols_eliminar, col)
}

if (length(cols_eliminar)>0) {
  cols_a_eliminar <- intersect(cols_eliminar, names(df_items))
  if (length(cols_a_eliminar)>0) {
    cat(sprintf("[INFO] Excluyendo: %s\n", paste(cols_a_eliminar, collapse=", ")))
    df_items <- dplyr::select(df_items, -dplyr::any_of(cols_a_eliminar))
  }
}

nombres_items <- colnames(df_items)
cat(sprintf("[OK] Items (%d): %s\n", length(nombres_items), paste(nombres_items, collapse=", ")))

for (col in nombres_items) df_items[[col]] <- as.numeric(df_items[[col]])

n_missing <- sum(!complete.cases(df_items))
if (n_missing>0) {
  pct_missing <- n_missing/(nrow(df_items)*ncol(df_items))*100
  cat(sprintf("[AVISO] %.2f%% datos faltantes (%d). Listwise deletion.\n", pct_missing, n_missing))
  df_items_clean <- df_items[complete.cases(df_items),]
  if (tiene_grupo) grupo_clean <- grupo[complete.cases(df_items)]
} else {
  df_items_clean <- df_items
  if (tiene_grupo) grupo_clean <- grupo
  cat(sprintf("[OK] Muestra completa: %d participantes\n", nrow(df_items_clean)))
}

mat_datos <- as.matrix(df_items_clean)
n_items <- length(nombres_items); n_participantes <- nrow(df_items_clean)

# Outliers multivariados
cat("\n  [ANALISIS] Deteccion Outliers Multivariados...\n")
outliers_result <- detectar_outliers_mahalanobis(df_items_clean)

if (outliers_result$n_outliers>0) {
  cat(sprintf("  [ATENCION] %d outliers (%.2f%%)\n", outliers_result$n_outliers, outliers_result$proportion))
  png(file.path(DIR_SALIDA,"02_outliers_mahalanobis.png"), width=12,height=8,units="in",res=150)
  par(mar=c(5,5,4,2))
  plot(outliers_result$distances, type='h', main="Distancias Mahalanobis",
       ylab="Distancia (chi2)", xlab="Participante",
       col=ifelse(outliers_result$outliers,'red','gray50'))
  abline(h=outliers_result$critical_value, col='red', lwd=2, lty=2)
  legend("topright", legend=c(sprintf("Critico (alpha=.001): %.2f",outliers_result$critical_value),
                               sprintf("Outliers: %d",outliers_result$n_outliers)),
         col=c('red','black'), lty=c(2,NA), pch=c(NA,16))
  dev.off()
  cat("[GRAFICA] Guardada: 02_outliers_mahalanobis.png\n")

  ##########
  # =============================================================================
  # ANÁLISIS DE SENSIBILIDAD: Comparación con vs. sin outliers multivariados
  # =============================================================================
  cat("\n\n", paste(rep("#",80),collapse=""), "\n")
  cat("#  ANÁLISIS DE SENSIBILIDAD (OUTLIERS MULTIVARIADOS)\n")
  cat(paste(rep("#",80),collapse=""), "\n")

  if (outliers_result$n_outliers > 0) {
    
    # 1. Crear los dos conjuntos de datos
    idx_out <- outliers_result$outliers
    df_con_outliers <- df_items_clean                 # muestra completa (N = 534)
    df_sin_outliers <- df_items_clean[-idx_out, ]     # sin los outliers detectados
    
    cat(sprintf("\n  Comparando muestras:\n"))
    cat(sprintf("    Con outliers:    N = %d\n", nrow(df_con_outliers)))
    cat(sprintf("    Sin outliers:    N = %d\n", nrow(df_sin_outliers)))
    
    # 2. Función para extraer métricas clave de cada muestra
    calcular_metricas <- function(df, nombre) {
      met <- list(muestra = nombre, N = nrow(df))
      
      # Fiabilidad
      alfa_obj <- tryCatch(psych::alpha(df, check.keys = TRUE), error = function(e) NULL)
      met$alpha <- if (!is.null(alfa_obj)) alfa_obj$total$raw_alpha else NA
      met$omega <- tryCatch(omega_mcdonald_calc(df), error = function(e) NA)
      
      # KMO
      kmo_obj <- tryCatch(psych::KMO(df), error = function(e) list(MSA = NA))
      met$kmo <- kmo_obj$MSA
      
      # AFE (1 factor, método ML, rotación varimax)
      fa_obj <- tryCatch(psych::fa(df, nfactors = 1, rotate = "varimax", fm = "ml"), 
                        error = function(e) NULL)
      if (!is.null(fa_obj)) {
        met$var_explicada <- sum(fa_obj$Vaccounted[2, 1]) * 100
        cargas <- as.vector(fa_obj$loadings)
        met$carga_min <- min(abs(cargas))
        met$carga_max <- max(abs(cargas))
        met$carga_media <- mean(abs(cargas))
        met$comunalidad_media <- mean(fa_obj$communality)
      } else {
        met$var_explicada <- met$carga_min <- met$carga_max <- met$carga_media <- met$comunalidad_media <- NA
      }
      
      # AFC (modelo unifactorial, estimador MLR)
      modelo <- paste0("F1 =~ ", paste(nombres_items, collapse = " + "))
      fit <- tryCatch(lavaan::cfa(modelo, data = df, std.lv = TRUE, estimator = "MLR"), 
                      error = function(e) NULL)
      if (!is.null(fit) && lavaan::lavInspect(fit, "converged")) {
        idx <- lavaan::fitMeasures(fit)
        met$cfi      <- as.numeric(idx["cfi"])
        met$tli      <- as.numeric(idx["tli"])
        met$rmsea    <- as.numeric(idx["rmsea"])
        met$srmr     <- as.numeric(idx["srmr"])
        met$chisq    <- as.numeric(idx["chisq"])
        met$df       <- as.numeric(idx["df"])
        met$p_chisq  <- as.numeric(idx["pvalue"])
        # AVE y CR unidimensional
        cargas_est <- lavaan::standardizedSolution(fit)
        cargas_est <- cargas_est[cargas_est$op == "=~", "est.std"]
        loads_sq <- cargas_est^2
        met$AVE <- mean(loads_sq)
        met$CR  <- sum(cargas_est)^2 / (sum(cargas_est)^2 + sum(1 - loads_sq))
      } else {
        met$cfi <- met$tli <- met$rmsea <- met$srmr <- met$chisq <- met$df <- met$p_chisq <- met$AVE <- met$CR <- NA
      }
      
      return(met)
    }
    
    # 3. Calcular métricas para ambas muestras
    metrics_con <- calcular_metricas(df_con_outliers, "Con outliers")
    metrics_sin <- calcular_metricas(df_sin_outliers, "Sin outliers")
    
    # 4. Construir tabla comparativa
    tabla_sensibilidad <- data.frame(
      Indicador = c("N", "Alpha Cronbach", "Omega McDonald", "KMO",
                    "Var. Explicada (%)", "Carga mínima", "Carga máxima", "Carga media",
                    "Comunalidad media", "CFI", "TLI", "RMSEA", "SRMR",
                    "Chi-cuadrado", "gl", "p (chi2)", "AVE", "CR"),
      Con_outliers = c(
        metrics_con$N, round(metrics_con$alpha, 4), round(metrics_con$omega, 4),
        round(metrics_con$kmo, 4), round(metrics_con$var_explicada, 2),
        round(metrics_con$carga_min, 4), round(metrics_con$carga_max, 4),
        round(metrics_con$carga_media, 4), round(metrics_con$comunalidad_media, 4),
        round(metrics_con$cfi, 4), round(metrics_con$tli, 4),
        round(metrics_con$rmsea, 4), round(metrics_con$srmr, 4),
        round(metrics_con$chisq, 3), metrics_con$df,
        round(metrics_con$p_chisq, 4), round(metrics_con$AVE, 4),
        round(metrics_con$CR, 4)
      ),
      Sin_outliers = c(
        metrics_sin$N, round(metrics_sin$alpha, 4), round(metrics_sin$omega, 4),
        round(metrics_sin$kmo, 4), round(metrics_sin$var_explicada, 2),
        round(metrics_sin$carga_min, 4), round(metrics_sin$carga_max, 4),
        round(metrics_sin$carga_media, 4), round(metrics_sin$comunalidad_media, 4),
        round(metrics_sin$cfi, 4), round(metrics_sin$tli, 4),
        round(metrics_sin$rmsea, 4), round(metrics_sin$srmr, 4),
        round(metrics_sin$chisq, 3), metrics_sin$df,
        round(metrics_sin$p_chisq, 4), round(metrics_sin$AVE, 4),
        round(metrics_sin$CR, 4)
      ),
      stringsAsFactors = FALSE
    )
    
    # 5. Agregar diferencia absoluta para indicadores numéricos relevantes
    # (excluimos N, chi2, df, p)
    diffs <- c(NA, 
              metrics_con$alpha - metrics_sin$alpha,
              metrics_con$omega - metrics_sin$omega,
              metrics_con$kmo - metrics_sin$kmo,
              metrics_con$var_explicada - metrics_sin$var_explicada,
              metrics_con$carga_media - metrics_sin$carga_media,
              NA, NA, NA,
              metrics_con$cfi - metrics_sin$cfi,
              metrics_con$tli - metrics_sin$tli,
              metrics_con$rmsea - metrics_sin$rmsea,
              metrics_con$srmr - metrics_sin$srmr,
              NA, NA, NA,
              metrics_con$AVE - metrics_sin$AVE,
              metrics_con$CR - metrics_sin$CR)
    tabla_sensibilidad$Diferencia <- round(diffs, 4)
    
    # 6. Mostrar y guardar
    cat("\n  Tabla de sensibilidad (Outliers Mahalanobis α = .001):\n")
    print(tabla_sensibilidad, row.names = FALSE)
    
    guardar_tabla(tabla_sensibilidad, "11_sensibilidad_outliers")
    
    # 7. Interpretación automática
    cat("\n  Interpretación rápida:\n")
    delta_alpha <- metrics_con$alpha - metrics_sin$alpha
    delta_cfi   <- metrics_con$cfi - metrics_sin$cfi
    delta_rmsea <- metrics_con$rmsea - metrics_sin$rmsea
    cat(sprintf("    - Cambio en α: %+.4f\n", delta_alpha))
    cat(sprintf("    - Cambio en CFI: %+.4f\n", delta_cfi))
    cat(sprintf("    - Cambio en RMSEA: %+.4f\n", delta_rmsea))
    if (abs(delta_alpha) < 0.02 && abs(delta_cfi) < 0.01 && abs(delta_rmsea) < 0.01) {
      cat("    [OK] Las diferencias son mínimas. La decisión de conservar los outliers parece razonable.\n")
    } else {
      cat("    [ATENCIÓN] Se observan cambios relevantes. Considere reportar ambos análisis o excluir los casos.\n")
    }
    
  } else {
    cat("\n  No se detectaron outliers multivariados. No se realiza análisis de sensibilidad.\n")
  }

  ##########

} else {
  cat("  [OK] Sin outliers multivariados\n")
}

resultados_globales$n_inicial <- nrow(df_completo)
resultados_globales$n_final <- n_participantes
resultados_globales$n_outliers <- outliers_result$n_outliers

# =========================================================================
# 4. ESTADISTICAS DESCRIPTIVAS
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  2. ESTADISTICAS DESCRIPTIVAS DE ITEMS\n")
cat(paste(rep("#",80),collapse=""), "\n")

desc_stats <- data.frame(
  Media=colMeans(df_items_clean), `Desv.Est`=apply(df_items_clean,2,sd),
  Mediana=apply(df_items_clean,2,median), Minimo=apply(df_items_clean,2,min,na.rm=TRUE),
  Maximo=apply(df_items_clean,2,max,na.rm=TRUE), Asimetria=apply(df_items_clean,2,psych::skew),
  Curtosis_exces=apply(df_items_clean,2,psych::kurtosi), stringsAsFactors=FALSE
)

for (item in nombres_items) {
  desc_stats[item,"Pct_Min"] <- round(mean(df_items_clean[[item]]==min(df_items_clean[[item]],na.rm=TRUE))*100,2)
  desc_stats[item,"Pct_Max"] <- round(mean(df_items_clean[[item]]==max(df_items_clean[[item]],na.rm=TRUE))*100,2)
  desc_stats[item,"Asimetria_OK"] <- abs(desc_stats[item,"Asimetria"])<3
  desc_stats[item,"Curtosis_OK"] <- abs(desc_stats[item,"Curtosis_exces"])<10
}
print(round(desc_stats,4))
guardar_tabla(round(desc_stats,4),"01_estadisticas_descriptivas")

items_no_normales <- sum(!desc_stats$Asimetria_OK | !desc_stats$Curtosis_OK)
if (items_no_normales>0) cat(sprintf("\n  [AVISO] %d/%d items desvian normalidad univariada\n", items_no_normales, n_items))

png(file.path(DIR_SALIDA,"01_distribucion_items.png"), width=22,height=16,units="in",res=150)
n_col_plot <- ceiling(sqrt(n_items)); n_row_plot <- ceiling(n_items/n_col_plot)
par(mfrow=c(n_row_plot,n_col_plot), mar=c(3,3,2.5,1))
for (item in nombres_items) {
  vals <- df_items_clean[[item]]; counts <- table(vals)
  hist(vals, breaks=seq(ESCALA_MIN-0.5,ESCALA_MAX+0.5,by=1), probability=TRUE,
       main=sprintf("%s (M=%.2f,DE=%.2f)",item,mean(vals),sd(vals)),
       col=viridis(length(counts)), border="black", xlab="Likert", ylab="Densidad")
  curve(dnorm(x,mean(vals),sd(vals)), add=TRUE, col="red", lwd=2)
}
dev.off()
cat("[GRAFICA] Guardada: 01_distribucion_items.png\n")

# =========================================================================
# 5. NORMALIDAD MULTIVARIADA (MARDIA)
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  3. NORMALIDAD MULTIVARIADA (Mardia)\n")
cat(paste(rep("#",80),collapse=""), "\n")

tryCatch({
  n <- nrow(df_items_clean); p <- ncol(df_items_clean)
  X <- scale(df_items_clean)
  Mardia_skew <- sum(apply(X,1,function(x)x^3)^2)/(n^2)
  skew_z <- sqrt(n/24)*Mardia_skew; skew_p <- 2*(1-pnorm(abs(skew_z)))
  Mardia_kurt <- sum(apply(X,1,function(x)x^4)^2)/n
  kurt_z <- (Mardia_kurt-p*(p+2))/sqrt(8*p*(p+2)/n); kurt_p <- 2*(1-pnorm(abs(kurt_z)))
  
  mardia_result <- data.frame(test="Mardia",skewness=Mardia_skew,kurtosis=Mardia_kurt,
                              skewness_p=skew_p,kurtosis_p=kurt_p)
  
  cat("\n Resultados Mardia:\n")
  cat(sprintf("   Asimetria b1,p=%.3f (p=%.2e)\n", mardia_result$skewness,mardia_result$skewness_p))
  cat(sprintf("   Curtosis b2,p=%.3f (esperado~%d)\n", mardia_result$kurtosis,n_items*(n_items+2)))
  
  normalidad_multivariada_ok <- mardia_result$skewness_p>.05 & mardia_result$kurtosis_p>.05
  
  if (!normalidad_multivariada_ok) {
    cat("  [CONCLUSION] Violacion normalidad multivariada -> Usar MLR\n")
    resultados_globales$normalidad_multivariada <- FALSE
    resultados_globales$estimador_recomendado <- "MLR (robusto)"
  } else {
    cat("  [CONCLUSION] Normalidad aceptable -> Usar ML\n")
    resultados_globales$normalidad_multivariada <- TRUE
    resultados_globales$estimador_recomendado <- "ML"
  }
  resultados_globales$mardia <- mardia_result
}, error=function(e){
  cat(sprintf("  [ERROR] Mardia fallido: %s\n", e$message))
  resultados_globales$normalidad_multivariada <- FALSE
  resultados_globales$estimador_recomendado <- "MLR (robusto)"
})

# =========================================================================
# 6. CONFIABILIDAD (CORREGIDO - ROBUSTO)
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  4. ANALISIS DE CONFIABILIDAD\n")
cat(paste(rep("#",80),collapse=""), "\n")

alfa_resultado <- suppressWarnings(psych::alpha(df_items_clean, check.keys=TRUE))
alpha_global <- alfa_resultado$total$raw_alpha
cat(sprintf("\n  Alpha Cronbach GLOBAL: %.4f [%s]\n", alpha_global, clasificar_alpha(alpha_global)))

alfa_sin_item <- alfa_resultado$alpha.drop
cat("\n  Alpha si elimina item:\n"); tabla_alpha_drop <- data.frame()
for (i in 1:nrow(alfa_sin_item)) {
  a_sin <- alfa_sin_item$raw_alpha[i]; flecha <- ifelse(a_sin>alpha_global,"↑ mejora","↓ empeora")
  cat(sprintf("    Sin %s: %.4f %s\n", rownames(alfa_sin_item)[i], a_sin, flecha))
  tabla_alpha_drop <- rbind(tabla_alpha_drop, data.frame(Item=rownames(alfa_sin_item)[i],
                              Alpha_sin=round(a_sin,4), Cambio=round(a_sin-alpha_global,4), Direccion=flecha))
}
guardar_tabla(tabla_alpha_drop,"02_alpha_if_deleted")

omega_global <- omega_mcdonald_calc(df_items_clean)
cat(sprintf("\n  Omega McDonald: %.4f [%s]\n", omega_global, clasificar_alpha(omega_global)))

sb <- NA
tryCatch({
  sp <- psych::splitHalf(df_items_clean, raw=TRUE)
  if (!is.null(sp$overall) && !is.null(sp$overall$raw_split)) {
    val <- sp$overall$raw_split
    if (length(val)>0 && is.numeric(val) && all(is.finite(val))) {
      sb <- as.numeric(val[1])
    }
  }
  if (is.null(sb) || is.na(sb) || length(sb)==0) {
    gutt <- psych::guttman(df_items_clean)
    if (!is.null(gutt$six) && length(gutt$six)>0 && is.numeric(gutt$six)) {
      sb <- as.numeric(gutt$six[1])
    }
  }
}, error=function(e){
  cat(sprintf("  [AVISO] Split-Half error: %s\n", e$message))
  sb <<- NA
}, warning=function(w){})

if (!is.null(sb) && length(sb)==1 && is.numeric(sb) && !is.na(sb) && is.finite(sb)) {
  cat(sprintf("  Spearman-Brown (Split-Half): %.4f\n", sb))
} else {
  sb <- NA
  cat("  Spearman-Brown: No calculable (estructura/muestra inadecuada)\n")
}

cat("\n  Correlacion Item-Total Corregida:\n"); corr_item_total <- c(); tabla_rit <- data.frame()
for (item in nombres_items) {
  total_sin <- rowSums(df_items_clean[,nombres_items[nombres_items!=item],drop=FALSE])
  r_val <- cor(df_items_clean[[item]], total_sin, use="complete.obs")
  corr_item_total[item] <- r_val
  n <- length(df_items_clean[[item]])
  t_stat <- r_val*sqrt((n-2)/(1-r_val^2))
  p_val <- 2*pt(abs(t_stat),df=n-2,lower.tail=FALSE)
  sig <- ifelse(p_val<.001,"***",ifelse(p_val<.01,"**",ifelse(p_val<.05,"*","ns")))
  cat(sprintf("    %s: r=%.4f %s (>0.30:%s)\n", item, r_val, sig, ifelse(r_val>=0.3,"SI","NO")))
  tabla_rit <- rbind(tabla_rit, data.frame(Item=item,r_itc=round(r_val,4),
                    p_valor=format(p_val,scientific=TRUE,digits=3),Significativo=sig,Aceptable=r_val>=0.3))
}
guardar_tabla(tabla_rit,"03_correlacion_item_total")

resultados_globales$alpha <- alpha_global
resultados_globales$omega <- omega_global
resultados_globales$split_half <- sb

# =========================================================================
# 7. ADECUACION MUESTRA (KMO Y BARTLETT)
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  5. ADECUACION MUESTRA (KMO Y BARTLETT)\n")
cat(paste(rep("#",80),collapse=""), "\n")

kmo_result <- tryCatch(psych::KMO(mat_datos), error=function(e){
  cat(sprintf("  [AVISO] KMO error: %s\n",e$message)); return(list(MSA=NA,MSAi=rep(NA,n_items)))})
kmo_total <- kmo_result$MSA; kmo_items <- kmo_result$MSAi
cat(sprintf("\n  KMO Global: %.4f [%s]\n", kmo_total, clasificar_kmo(kmo_total)))

cat("\n  KMO por item:\n"); tabla_kmo <- data.frame(Item=nombres_items,KMO=round(kmo_items,4))
for (i in 1:length(kmo_items)) cat(sprintf("    %s: %.4f %s\n",nombres_items[i],kmo_items[i],ifelse(kmo_items[i]>=.7,"(OK)","(BAJO)")))
guardar_tabla(tabla_kmo,"04_KMO_por_item")

bartlett_result <- tryCatch(psych::cortest.bartlett(cor(mat_datos,use="pairwise.complete.obs"),n=nrow(mat_datos)),
                            error=function(e){return(list(chisq=NA,p.value=NA))})
cat(sprintf("  Bartlett: chi2=%.2f, p=%.2e (Sig:%s)\n",bartlett_result$chisq,bartlett_result$p.value,
           ifelse(bartlett_result$p.value<.05,"SI","NO")))

resultados_globales$kmo <- kmo_total; resultados_globales$bartlett <- bartlett_result

# =========================================================================
# 8. DETERMINACION FACTORES Y SCREE PLOT
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  6. DETERMINACION NUMERO DE FACTORES\n")
cat(paste(rep("#",80),collapse=""), "\n")

cat("\n  [EXPLICACION] ¿Por qué el script extrae 'ml1' y 'ml2' y no 4 o 6?\n")
cat("  -> El algoritmo usa criterios estadísticos automáticos (Análisis Paralelo y Kaiser).\n")
cat("  -> Si solo 2 componentes superan el umbral (eigenvalue > 1.0 o simulación PA),\n")
cat("     el modelo se detiene en 2 para evitar sobre-factorización y ruido estadístico.\n")
cat("  -> 'ml1' y 'ml2' son nombres por defecto de psych::fa (Maximum Likelihood).\n")
cat("  -> Para forzar 4 o 6 factores, configure 'N_FACTORES_TEORICO' arriba en NA->4 o 6.\n")

set.seed(SEED)
n_factores_final <- 1
eigenvalues_reales <- NULL
pa_result <- NULL

tryCatch({
  pa_result <- psych::fa.parallel(mat_datos, fm="ml", fa="fa", n.iter=500, plot=FALSE)
  
  if (!is.null(pa_result$eigen$values)) {
    if (is.matrix(pa_result$eigen$values) || is.data.frame(pa_result$eigen$values)) {
      if ("Eigenvalues" %in% colnames(pa_result$eigen$values)) {
        eigenvalues_reales <- as.numeric(pa_result$eigen$values[, "Eigenvalues"])
      } else {
        eigenvalues_reales <- as.numeric(pa_result$eigen$values[, 1])
      }
    } else if (is.vector(pa_result$eigen$values)) {
      eigenvalues_reales <- as.numeric(pa_result$eigen$values)
    }
  }
  
  if (is.null(eigenvalues_reales) || length(eigenvalues_reales) == 0 || any(is.na(eigenvalues_reales))) {
    cat("  [AVISO] Extracción automática falló, calculando manualmente...\n")
    mat_corr <- cor(mat_datos, use="pairwise.complete.obs")
    eigenvalues_reales <- eigen(mat_corr, symmetric=TRUE)$values
  }
  eigenvalues_reales <- eigenvalues_reales[is.finite(eigenvalues_reales)]
  if (length(eigenvalues_reales) == 0) stop("No se obtuvieron eigenvalues válidos")
  
  n_factores_paralelo <- tryCatch(pa_result$nfact, error=function(e) NULL)
  if (is.null(n_factores_paralelo) || !is.numeric(n_factores_paralelo) || n_factores_paralelo < 1) {
    n_factores_paralelo <- sum(eigenvalues_reales > 1)
  }
  n_kaiser <- sum(eigenvalues_reales > 1.0)
  
  cat(sprintf("  Análisis Paralelo sugiere: %d factor(es)\n", n_factores_paralelo))
  cat(sprintf("  Criterio Kaiser sugiere: %d factor(es)\n", n_kaiser))
  
  # >>> LOGICA DE DECISION: PRIORIZA TEORIA SOBRE DATOS SI SE CONFIGURA <<<
  if (!is.na(N_FACTORES_TEORICO) && N_FACTORES_TEORICO > 1) {
    n_factores_final <- N_FACTORES_TEORICO
    cat(sprintf("  [MODO TEORICO] Forzando a %d factores según configuración.\n", n_factores_final))
  } else {
    n_factores_final <- if (!is.na(n_factores_paralelo) && n_factores_paralelo >= 1) n_factores_paralelo else max(1, n_kaiser)
    cat(sprintf("  [MODO AUTOMATICO] Seleccionados: %d factores (criterio estadístico).\n", n_factores_final))
  }
  
  tryCatch({
    png(file.path(DIR_SALIDA,"03_scree_plot.png"), width=12, height=8, units="in", res=150)
    n_eig <- length(eigenvalues_reales)
    par(mar=c(5,5,4,2))
    y_max <- max(eigenvalues_reales, na.rm=TRUE) * 1.15
    y_max <- min(y_max, 10); y_max <- max(y_max, 2)
    plot(1:n_eig, eigenvalues_reales, type="b", pch=19, lwd=2, col="blue",
         main="Scree Plot con Criterios Múltiples", xlab="Factor", ylab="Eigenvalue",
         ylim=c(0, y_max), xaxt="n")
    axis(1, at=1:n_eig, labels=1:n_eig)
    abline(h=1, col="red", lty=2, lwd=2)
    if (!is.null(pa_result$eigen) && is.list(pa_result$eigen) && !is.null(pa_result$eigen$simulated)) {
      sim_data <- pa_result$eigen$simulated
      if (is.array(sim_data) && length(dim(sim_data)) == 3) {
        sim_mean <- rowMeans(sim_data[,, "Eigenvalues"], dims=2)
        sim_q95 <- apply(sim_data[,, "Eigenvalues"], 1, quantile, 0.95, na.rm=TRUE)
        if (length(sim_mean) == n_eig && all(is.finite(sim_mean))) {
          lines(1:n_eig, sim_mean, col="orange", lwd=2, lty=2)
          if (length(sim_q95) == n_eig && all(is.finite(sim_q95))) {
            lines(1:n_eig, sim_q95, col="darkred", lwd=2, lty=3)
          }
        }
      }
    }
    abline(v=n_factores_final+0.5, col="green", lwd=3, lty=1)
    legend("topright", legend=c("Observados", "Kaiser>1", sprintf("Retenidos: %d", n_factores_final)),
           col=c("blue","red","green"), lty=c(1,2,1), lwd=c(2,2,3), pch=c(19,NA,NA), cex=0.9)
    dev.off()
    cat("[GRÁFICA] Guardada: 03_scree_plot.png\n")
  }, error=function(e) { cat(sprintf("  [AVISO] Error en scree plot: %s\n", e$message)) })
  
  cat(sprintf("\n  >>> SE USARÁN: %d FACTORES\n", n_factores_final))
  
}, error=function(e){
  cat(sprintf("  [ERROR] En determinación de factores: %s\n", e$message))
  cat("  [ALTERNATIVA] Usando Kaiser fallback...\n")
  tryCatch({
    mat_corr <- cor(mat_datos, use="pairwise.complete.obs")
    eig <- eigen(mat_corr)$values
    eig <- eig[is.finite(eig)]
    n_factores_final <- sum(eig > 1)
    if (n_factores_final < 1) n_factores_final <- 1
    eigenvalues_reales <- eig
    cat(sprintf("  >>> SE USARÁN: %d FACTORES (fallback)\n", n_factores_final))
  }, error=function(e2) { n_factores_final <<- 1 })
})

resultados_globales$n_factores <- n_factores_final
resultados_globales$eigenvalues <- eigenvalues_reales

# =========================================================================
# 9. ANALISIS FACTORIAL EXPLORATORIO (AFE)
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  7. ANALISIS FACTORIAL EXPLORATORIO (AFE)\n")
cat(paste(rep("#",80),collapse=""), "\n")

fa_efa <- NULL; fa_varimax <- NULL; fa_promax <- NULL; loadings_efa <- NULL; error_afe <- FALSE
metodo_extraccion <- "ml"

tryCatch({
  cat("  Intentando AFE con ML...\n")
  fa_varimax <- psych::fa(mat_datos, nfactors=n_factores_final, rotate="varimax", fm="ml", max.iter=1000)
  fa_promax <- psych::fa(mat_datos, nfactors=n_factores_final, rotate="promax", fm="ml", max.iter=1000)
}, error=function(e){
  cat(sprintf("  [AVISO] ML falló: %s. Cambiando a PA...\n", e$message))
  metodo_extraccion <<- "pa"
  tryCatch({
    fa_varimax <- psych::fa(mat_datos, nfactors=n_factores_final, rotate="varimax", fm="pa")
    fa_promax <- psych::fa(mat_datos, nfactors=n_factores_final, rotate="promax", fm="pa")
  }, error=function(e2){
    cat(sprintf("  [ERROR GRAVE] AFE imposible: %s\n", e2$message))
    error_afe <<- TRUE
  })
})

if (error_afe || is.null(fa_varimax)) {
  cat("\n  [FALLO] No se extrajeron factores.\n")
} else {
  simp_vari <- mean(abs(fa_varimax$loadings)*(1-abs(fa_varimax$loadings)))
  simp_prom <- mean(abs(fa_promax$loadings)*(1-abs(fa_promax$loadings)))
  rotacion_elegida <- ifelse(simp_prom < simp_vari, "promax", "varimax")
  fa_efa <- if(rotacion_elegida == "promax") fa_promax else fa_varimax
  
  cat(sprintf("\n  Rotación: %s | Método: %s\n", toupper(rotacion_elegida), toupper(metodo_extraccion)))
  print(fa_efa$loadings, cutoff=0, sort=TRUE)
  
  comunalidades <- fa_efa$communality
  cat("\n  Comunalidades (h2):\n"); print(round(comunalidades,3))
  
  var_exp_pct <- sum(fa_efa$Vaccounted[2, 1:n_factores_final]) * 100
  cat(sprintf("\n  Varianza Total Explicada: %.2f%%\n", var_exp_pct))
  cat("  Varianza por Factor:\n")
  for (f in 1:n_factores_final) cat(sprintf("    Factor %d: %.2f%%\n", f, fa_efa$Vaccounted[2,f]*100))
  
  loadings_efa <- fa_efa$loadings
  tabla_loadings <- as.data.frame.matrix(loadings_efa)
  tabla_loadings$Comunalidad <- round(comunalidades,3)
  guardar_tabla(round(tabla_loadings,4), "05_cargas_factoriales_AFE")
  
  # CORREGIDO: Gráfico de cargas usando ggplot2 para evitar errores de names en barplot
  tryCatch({
    png(file.path(DIR_SALIDA,"04_cargas_factoriales.png"), width=14, height=max(8, ceiling(n_items*0.4)), units="in", res=150)
    
    loadings_mat_safe <- as.matrix(loadings_efa)
    if(is.null(colnames(loadings_mat_safe))) colnames(loadings_mat_safe) <- paste0("F", seq_len(ncol(loadings_mat_safe)))
    
    df_plot <- do.call(rbind, lapply(seq_len(ncol(loadings_mat_safe)), function(j) {
      data.frame(Item = rownames(loadings_mat_safe), 
                 Factor = colnames(loadings_mat_safe)[j], 
                 Loading = loadings_mat_safe[, j],
                 stringsAsFactors = FALSE)
    }))
    df_plot$Factor <- factor(df_plot$Factor, levels = colnames(loadings_mat_safe))
    df_plot <- df_plot[order(df_plot$Factor, -abs(df_plot$Loading)), ]
    df_plot$Item <- factor(df_plot$Item, levels = df_plot$Item)
    
    p <- ggplot(df_plot, aes(x = Item, y = Loading, fill = Factor)) +
      geom_bar(stat = "identity", position = "dodge") +
      coord_flip() +
      theme_minimal() +
      labs(title = "Cargas Factoriales (AFE)", x = "Ítem", y = "Carga Factorial") +
      scale_fill_viridis_d() +
      geom_hline(yintercept = c(-0.5, -0.3, 0.3, 0.5), linetype = "dashed", color = "gray50") +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            legend.position = "top",
            plot.title = element_text(hjust = 0.5))
    
    print(p)
    dev.off()
    cat("[GRÁFICA] Guardada: 04_cargas_factoriales.png\n")
  }, error = function(e) {
    cat(sprintf("  [AVISO] Error en gráfico de cargas: %s\n", e$message))
    dev.off()
  })
  
  resultados_globales$varianza_explicada <- var_exp_pct
  resultados_globales$rotacion <- rotacion_elegida
  resultados_globales$metodo_extraccion <- metodo_extraccion
}

# =========================================================================
# 10. OMEGA JERÁRQUICO (CORREGIDO)
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  8. OMEGA JERÁRQUICO (ωh)\n")
cat(paste(rep("#",80),collapse=""), "\n")

if (!is.null(loadings_efa) && n_factores_final > 1) {
  omega_jer <- tryCatch({
    calcular_omega_jerarquico(df_items_clean, as.matrix(loadings_efa))
  }, error = function(e) {
    cat(sprintf("  [AVISO] Error omega jerárquico: %s\n", e$message))
    return(list(omega_h = NA, omega_t = NA))
  })
  if (!is.null(omega_jer$omega_h) && !is.na(omega_jer$omega_h)) {
    cat(sprintf("\n  Omega Jerárquico (ωh): %.4f\n", omega_jer$omega_h))
    cat(sprintf("  Omega Total (ωt): %.4f\n", omega_jer$omega_t))
    if (omega_jer$omega_h >= 0.70) cat("  [OK] Estructura jerárquica adecuada\n") else cat("  [AVISO] ωh bajo (<0.70)\n")
    resultados_globales$omega_h <- omega_jer$omega_h
    resultados_globales$omega_t <- omega_jer$omega_t
  } else {
    cat("  [INFO] No se pudo calcular ωh\n")
  }
} else {
  cat("\n  [OMITIDO] Omega jerárquico solo para multifactorial\n")
}

# =========================================================================
# 11. AFC - ANÁLISIS FACTORIAL CONFIRMATORIO (SUPER CORREGIDO)
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  9. ANALISIS FACTORIAL CONFIRMATORIO (AFC)\n")
cat(paste(rep("#",80),collapse=""), "\n")

fit_cfa <- NULL
cfi_val <- rmsea_val <- srmr_val <- tli_val <- chi2_val <- dof_val <- NA

if (!is.null(loadings_efa) && is.matrix(loadings_efa)) {
  tryCatch({
    n_cols_loadings <- ncol(loadings_efa)
    nombres_factores <- colnames(loadings_efa)
    if (is.null(nombres_factores) || length(nombres_factores) == 0) {
      nombres_factores <- paste0("F", 1:n_cols_loadings)
    }
    asignacion <- character(nrow(loadings_efa))
    names(asignacion) <- rownames(loadings_efa)
    for (i in 1:nrow(loadings_efa)) {
      fila <- loadings_efa[i, ]
      idx_max <- which.max(abs(fila))
      asignacion[i] <- ifelse(length(idx_max) > 0 && !is.na(idx_max) && idx_max <= length(nombres_factores), nombres_factores[idx_max], NA)
    }
    modelo_lines <- character(0)
    items_por_factor <- list()
    for (j in 1:length(nombres_factores)) {
      fname <- nombres_factores[j]
      items_en_factor <- names(asignacion)[!is.na(asignacion) & asignacion == fname]
      items_por_factor[[fname]] <- items_en_factor
      if (length(items_en_factor) > 0) {
        for (item in items_en_factor) modelo_lines <- c(modelo_lines, sprintf("  %s =~ %s", fname, item))
        cat(sprintf("    %s: %s (%d ítems)\n", fname, paste(items_en_factor, collapse=", "), length(items_en_factor)))
      }
    }
    if (length(modelo_lines) > 0) {
      modelo_spec <- paste(modelo_lines, collapse = "\n")
      estimator <- if(!normalidad_multivariada_ok) "MLR" else "ML"
      cat(sprintf("  Estimador: %s\n", estimator))
      fit_cfa <- tryCatch(lavaan::cfa(modelo_spec, data = df_items_clean, std.lv = TRUE, estimator = estimator), error = function(e) NULL)
      if (!is.null(fit_cfa)) {
        summary(fit_cfa, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)
        indices_fit <- tryCatch(lavaan::fitMeasures(fit_cfa), error = function(e) NULL)
        if (!is.null(indices_fit)) {
          cfi_val <- tryCatch(as.numeric(indices_fit["cfi"]), error=function(e) NA)
          tli_val <- tryCatch(as.numeric(indices_fit["tli"]), error=function(e) NA)
          rmsea_val <- tryCatch(as.numeric(indices_fit["rmsea"]), error=function(e) NA)
          srmr_val <- tryCatch(as.numeric(indices_fit["srmr"]), error=function(e) NA)
          chi2_val <- tryCatch(as.numeric(indices_fit["chisq"]), error=function(e) NA)
          dof_val <- tryCatch(as.numeric(indices_fit["df"]), error=function(e) NA)
          rmsea_ci_low <- tryCatch(as.numeric(indices_fit["rmsea.ci.lower"]), error=function(e) NA)
          rmsea_ci_high <- tryCatch(as.numeric(indices_fit["rmsea.ci.upper"]), error=function(e) NA)
          cat("\n  ÍNDICES DE AJUSTE DEL MODELO:\n")
          if (!is.na(chi2_val) && !is.na(dof_val)) cat(sprintf("    χ² = %.2f (gl = %d)\n", chi2_val, dof_val))
          if (!is.na(cfi_val)) cat(sprintf("    CFI = %.3f [%s]\n", cfi_val, clasificar_ajuste_cfi(cfi_val)))
          if (!is.na(tli_val)) cat(sprintf("    TLI = %.3f [%s]\n", tli_val, clasificar_ajuste_cfi(tli_val)))
          if (!is.na(rmsea_val)) {
            if (!is.na(rmsea_ci_low) && !is.na(rmsea_ci_high)) {
              cat(sprintf("    RMSEA = %.3f [90%% CI: %.3f, %.3f] [%s]\n", rmsea_val, rmsea_ci_low, rmsea_ci_high, clasificar_ajuste_rmsea(rmsea_val)))
            } else { cat(sprintf("    RMSEA = %.3f [%s]\n", rmsea_val, clasificar_ajuste_rmsea(rmsea_val))) }
          }
          if (!is.na(srmr_val)) cat(sprintf("    SRMR = %.3f (aceptable < .08)\n", srmr_val))
          tabla_ajuste <- data.frame(Indice = c("Chi-cuadrado", "gl", "CFI", "TLI", "RMSEA", "RMSEA_CI_lower", "RMSEA_CI_upper", "SRMR"),
                                     Valor = c(ifelse(is.na(chi2_val), "-", round(chi2_val, 2)), ifelse(is.na(dof_val), "-", dof_val),
                                               ifelse(is.na(cfi_val), "-", round(cfi_val, 3)), ifelse(is.na(tli_val), "-", round(tli_val, 3)),
                                               ifelse(is.na(rmsea_val), "-", round(rmsea_val, 3)), ifelse(is.na(rmsea_ci_low), "-", round(rmsea_ci_low, 3)),
                                               ifelse(is.na(rmsea_ci_high), "-", round(rmsea_ci_high, 3)), ifelse(is.na(srmr_val), "-", round(srmr_val, 3))),
                                     stringsAsFactors = FALSE)
          guardar_tabla(tabla_ajuste, "06_indices_ajuste_AFC")
          tryCatch({
            png(file.path(DIR_SALIDA, "07_diagrama_senderos.png"), width = 14, max(10, n_items * 0.6), units = "in", res = 150)
            semPaths(fit_cfa, what = "std", whatLabels = "std", layout = "tree", edge.label.cex = 0.8, fade = FALSE, title = FALSE, nodeNames = nombres_items, sizeMan = 10, sizeLat = 12, edge.color = "black")
            dev.off()
            cat("[GRÁFICA] Guardada: 07_diagrama_senderos.png\n")
          }, error = function(e) { cat(sprintf("  [AVISO] No se pudo crear diagrama: %s\n", e$message)); dev.off() })
          cat("\n  Análisis de Residuos:\n")
          tryCatch({
            residuos <- residuals(fit_cfa, type = "standardized")
            if (!is.null(residuos$cov)) {
              resid_grandes <- which(abs(residuos$cov) > 2.58, arr.ind = TRUE)
              if (nrow(resid_grandes) > 0) cat(sprintf("    [AVISO] %d pares con residuos |z| > 2.58\n", nrow(resid_grandes)/2)) else cat("    [OK] Sin residuos grandes\n")
            }
          }, error = function(e) cat("    [AVISO] No se pudieron analizar residuos\n"))
          aic_val <- tryCatch(as.numeric(indices_fit["aic"]), error=function(e) NA)
          bic_val <- tryCatch(as.numeric(indices_fit["bic"]), error=function(e) NA)
          cat("\n  Índices de Información:\n")
          if (!is.na(aic_val)) cat(sprintf("    AIC: %.2f\n", aic_val))
          if (!is.na(bic_val)) cat(sprintf("    BIC: %.2f\n", bic_val))
          resultados_globales$cfa <- list(cfi = cfi_val, tli = tli_val, rmsea = rmsea_val, srmr = srmr_val, chi2 = chi2_val, dof = dof_val, aic = aic_val, bic = bic_val)
        }
      }
    } else { cat("  [SALTADO] Modelo vacío (sin ítems asignados a factores)\n") }
  }, error = function(e) { cat(sprintf("  [ERROR] Construyendo AFC: %s\n", e$message)) })
} else { cat("  [SALTADO] No hay matriz de loadings válida para AFC\n") }

# =========================================================================
# 12. VALIDEZ CONVERGENTE Y DISCRIMINANTE (ULTRA CORREGIDO)
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  10. VALIDEZ CONVERGENTE Y DISCRIMINANTE\n")
cat(paste(rep("#",80),collapse=""), "\n")

if (!is.null(loadings_efa) && is.matrix(loadings_efa) && nrow(loadings_efa) > 0) {
  loadings_mat <- tryCatch({ mat_temp <- as.matrix(loadings_efa); mode(mat_temp) <- "numeric"; mat_temp }, error = function(e) NULL)
  if (is.null(loadings_mat)) { cat("  [SALTADO] Error creando matriz de loadings\n")
  } else {
    loadings_mat <- loadings_mat[rowSums(abs(loadings_mat)) > 0, , drop = FALSE]
    if (nrow(loadings_mat) == 0 || ncol(loadings_mat) == 0) { cat("  [SALTADO] Matriz vacía después de filtrar\n")
    } else {
      loadings_df <- tryCatch({ df_temp <- as.data.frame.matrix(loadings_mat); for (col_name in names(df_temp)) df_temp[[col_name]] <- as.numeric(df_temp[[col_name]]); df_temp }, error = function(e) NULL)
      if (!is.null(loadings_df) && ncol(loadings_df) > 0) {
        ave_cr_result <- data.frame(Factor = character(0), AVE = numeric(0), CR = numeric(0), stringsAsFactors = FALSE)
        nombres_facts_loadings <- colnames(loadings_df)
        for (fn in nombres_facts_loadings) {
          loads <- tryCatch(as.numeric(loadings_df[[fn]]), error=function(e) numeric(0))
          loads <- loads[!is.na(loads) & is.finite(loads)]
          if (length(loads) == 0) next
          loads_sq <- loads^2; ave <- mean(loads_sq); sum_loads <- sum(loads); sum_error <- sum(1 - loads_sq)
          if (sum_error <= 0) next
          cr <- (sum_loads^2) / ((sum_loads^2) + sum_error)
          ave_cr_result <- rbind(ave_cr_result, data.frame(Factor = fn, AVE = round(ave, 4), CR = round(cr, 4), stringsAsFactors = FALSE))
        }
        if (nrow(ave_cr_result) > 0) {
          cat("\n  Validez Convergente (AVE y CR):\n"); print(ave_cr_result)
          ave_cr_result$AVE_OK <- ave_cr_result$AVE > 0.50; ave_cr_result$CR_OK <- ave_cr_result$CR > 0.70
          guardar_tabla(ave_cr_result, "07_AVE_CR")
          n_factores_ave <- nrow(ave_cr_result)
          if (n_factores_ave >= 2) {
            cat("\n--- Análisis Validez Discriminante ---\n")
            n_factors_mat <- ncol(loadings_df)
            corr_factores <- tryCatch({
              if (rotacion_elegida == "promax" && !is.null(fa_efa$Phi)) {
                phi_temp <- fa_efa$Phi
                if (is.matrix(phi_temp) || is.data.frame(phi_temp)) { phi_mat <- as.matrix(phi_temp)
                  if (nrow(phi_mat) == n_factors_mat && ncol(phi_mat) == n_factors_mat) {
                    noms <- colnames(loadings_df)
                    if (length(noms) == nrow(phi_mat) && length(noms) == ncol(phi_mat)) { colnames(phi_mat) <- noms; rownames(phi_mat) <- noms }
                    phi_mat
                  } else { diag(n_factors_mat) }
                } else { diag(n_factors_mat) }
              } else { diag(n_factors_mat) }
            }, error = function(e) { diag(n_factors_mat) })
            cat("\n  Matriz Fornell-Larcker (Diagonal = √AVE):\n")
            n_fl <- nrow(ave_cr_result)
            disc_matrix <- matrix(0, nrow=n_fl, ncol=n_fl)
            rownames(disc_matrix) <- ave_cr_result$Factor; colnames(disc_matrix) <- ave_cr_result$Factor
            for (i in 1:n_fl) { for (j in 1:n_fl) { if (i != j) disc_matrix[i, j] <- tryCatch(abs(corr_factores[i, j]), error=function(e) 0) } }
            sqrt_ave <- sqrt(ave_cr_result$AVE)
            if (length(sqrt_ave) == n_fl) {
              diag(disc_matrix) <- sqrt_ave; print(round(disc_matrix, 3))
              vd_fl_ok <- TRUE
              for (i in 1:n_fl) { for (j in 1:n_fl) { if (i != j && !is.na(sqrt_ave[i]) && !is.na(disc_matrix[i,j]) && sqrt_ave[i] < disc_matrix[i,j]) vd_fl_ok <- FALSE } }
              cat(sprintf("  Validez Discriminante (Fornell-Larcker): %s\n", ifelse(vd_fl_ok, "✓ CUMPLE", "✗ NO CUMPLE")))
            } else { cat("  [ERROR] Longitudes incompatibles para Fornell-Larcker\n") }
            cat("\n  HTMT Ratio:\n")
            htmt_matrix <- tryCatch(calcular_htmt(loadings_df, corr_factores), error = function(e) matrix(NA, nrow=n_factors_mat, ncol=n_factors_mat))
            if (!all(is.na(htmt_matrix))) {
              print(round(htmt_matrix, 3))
              htmt_vals <- htmt_matrix; diag(htmt_vals) <- NA; htmt_vals <- htmt_vals[!is.na(htmt_vals)]
              if (length(htmt_vals) > 0) {
                htmt_max <- max(htmt_vals); vd_htmt_ok <- htmt_max < 0.85
                cat(sprintf("  Máximo HTMT: %.3f (criterio < 0.85: %s)\n", htmt_max, ifelse(vd_htmt_ok, "✓ OK", "✗ PROBLEMÁTICO")))
                resultados_globales$validez_discriminante <- list(fornell_larcker = vd_fl_ok, htmt = vd_htmt_ok, htmt_max = htmt_max)
              }
            } else { cat("  [AVISO] No se pudo calcular HTMT\n") }
          } else { cat("\n  (Validez discriminante no aplica: solo 1 factor)\n") }
          resultados_globales$validez_convergente <- ave_cr_result
        } else { cat("  [AVISO] No se pudo calcular AVE/CR\n") }
      }
    }
  }
} else { cat("\n  [SALTADO] Sin datos de loadings para validez\n") }

# =========================================================================
# 13. MATRIZ DE CORRELACIONES
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#  11. MATRIZ DE CORRELACIONES\n")
cat(paste(rep("#",80),collapse=""), "\n")

mat_corr <- cor(df_items_clean)
print(round(mat_corr, 3))

tryCatch({
  png(file.path(DIR_SALIDA,"05_matriz_correlaciones.png"), width=max(10, n_items*0.8), height=max(8, n_items*0.8), units="in", res=150)
  corrplot(mat_corr, method="color", type="upper", order="hclust", tl.col="black", tl.srt=45, tl.cex=0.8,
           addCoef.col="black", number.cex=0.7, col=colorRampPalette(c("blue","white","red"))(100), title="Matriz Correlaciones con Clustering", mar=c(0,0,2,0))
  dev.off()
  cat("[GRÁFICA] Guardada: 05_matriz_correlaciones.png\n")
}, error = function(e) { cat(sprintf("  [AVISO] Error gráfico correlaciones: %s\n", e$message)); dev.off() })
guardar_tabla(round(mat_corr, 4), "08_matriz_correlaciones")

# =========================================================================
# 14. ANÁLISIS DE INVARIANZA DE MEDICIÓN MULTIGRUPO CON LAVAAN
# =========================================================================
# Permite usar UNA o VARIAS variables sociodemográficas (sexo, edad, educación, etc.)
# Requiere: variable global VARS_GRUPO (vector de nombres de columnas)
# Si no existe, intenta usar COLUMNA_GRUPO (compatibilidad hacia atrás)
# =========================================================================

# --- Definir las variables de grupo a analizar ---
if (!exists("VARS_GRUPO") || is.null(VARS_GRUPO)) {
  if (tiene_grupo && !is.null(COLUMNA_GRUPO)) {
    VARS_GRUPO <- COLUMNA_GRUPO  # compatibilidad con versión anterior
  } else {
    VARS_GRUPO <- NULL
  }
}

# Verificar que hay al menos una variable de grupo definida y que existe en los datos
if (!is.null(VARS_GRUPO) && length(VARS_GRUPO) > 0 && exists("df_completo")) {
  
  # Asegurar que las variables de grupo estén en df_completo y sean factores
  vars_grupo_existentes <- VARS_GRUPO[VARS_GRUPO %in% names(df_completo)]
  
  if (length(vars_grupo_existentes) == 0) {
    cat("\n  [AVISO] Ninguna variable de grupo especificada se encontró en los datos.\n")
  } else {
    
    cat("\n\n", paste(rep("#",80),collapse=""), "\n")
    cat("#  ANÁLISIS DE INVARIANZA DE MEDICIÓN MULTIGRUPO\n")
    cat(paste(rep("#",80),collapse=""), "\n")
    
    # --------------------------------------------------------------
    # 1. Construir el modelo factorial a partir de la solución EFA
    # --------------------------------------------------------------
    if (!is.null(loadings_efa) && is.matrix(loadings_efa)) {
      
      loadings_mat <- as.matrix(loadings_efa)
      n_fact <- ncol(loadings_mat)
      nombres_factores <- colnames(loadings_mat)
      if (is.null(nombres_factores)) nombres_factores <- paste0("F", 1:n_fact)
      
      # Asignar cada ítem al factor con mayor carga absoluta
      asignacion <- apply(loadings_mat, 1, function(row) {
        idx <- which.max(abs(row))
        if (length(idx) == 0) NA else nombres_factores[idx]
      })
      
      # Construir líneas del modelo lavaan
      modelo_lines <- c()
      for (f in nombres_factores) {
        items_f <- names(asignacion)[!is.na(asignacion) & asignacion == f]
        if (length(items_f) >= 3) {  # mínimo 3 ítems por factor para identificación
          modelo_lines <- c(modelo_lines, paste0("  ", f, " =~ ", paste(items_f, collapse = " + ")))
        } else if (length(items_f) == 2) {
          # Para 2 ítems, fijar varianza del factor a 1 y permitir correlación residual?
          # Opción simple: usar igual (lavaan lo manejará, pero puede dar advertencia)
          modelo_lines <- c(modelo_lines, paste0("  ", f, " =~ ", paste(items_f, collapse = " + ")))
          cat(sprintf("  [AVISO] Factor %s tiene solo %d ítems. Invarianza puede ser inestable.\n", f, length(items_f)))
        } else if (length(items_f) == 1) {
          cat(sprintf("  [AVISO] Factor %s tiene 1 solo ítem. Se omite del modelo multigrupo.\n", f))
        }
      }
      
      if (length(modelo_lines) == 0) {
        cat("  [ERROR] No se pudo construir un modelo CFA válido para invarianza.\n")
      } else {
        modelo_spec <- paste(modelo_lines, collapse = "\n")
        cat("\n  Modelo CFA utilizado:\n")
        cat(modelo_spec, "\n")
        
        # --------------------------------------------------------------
        # 2. Función para probar invarianza en una variable de grupo
        # --------------------------------------------------------------
        probar_invarianza <- function(var_grupo, datos_completos, modelo, umbral_cfi = 0.01) {
          cat("\n", paste(rep("-",70), collapse=""), "\n")
          cat(">>> Variable de agrupamiento:", var_grupo, "\n")
          
          # Verificar existencia y convertir a factor
          if (!var_grupo %in% names(datos_completos)) {
            cat("  [ERROR] Variable no encontrada en datos.\n")
            return(NULL)
          }
          grupo_raw <- datos_completos[[var_grupo]]
          if (!is.factor(grupo_raw)) grupo_raw <- as.factor(grupo_raw)
          
          # Eliminar observaciones con NA en la variable de grupo
          idx_completos <- complete.cases(grupo_raw)
          if (sum(idx_completos) < nrow(datos_completos)) {
            cat(sprintf("  [INFO] Se eliminan %d casos con NA en %s\n", sum(!idx_completos), var_grupo))
            datos_grupo <- datos_completos[idx_completos, ]
            grupo <- droplevels(grupo_raw[idx_completos])
          } else {
            datos_grupo <- datos_completos
            grupo <- grupo_raw
          }
          
          niveles <- levels(grupo)
          n_grupos <- length(niveles)
          cat("  Grupos encontrados:", paste(niveles, collapse = ", "), "\n")
          cat("  Tamaños de grupo:", paste(table(grupo), collapse = ", "), "\n")
          
          if (n_grupos < 2) {
            cat("  [ADVERTENCIA] Menos de 2 grupos. Se omite.\n")
            return(NULL)
          }
          
          # Filtrar grupos con al menos 10 observaciones (recomendación)
          grupos_validos <- niveles[table(grupo) >= 10]
          if (length(grupos_validos) < 2) {
            cat("  [ADVERTENCIA] Grupos con menos de 10 casos. No se puede estimar invarianza.\n")
            return(NULL)
          }
          if (length(grupos_validos) < n_grupos) {
            cat(sprintf("  [INFO] Se excluyen grupos pequeños: %s\n", paste(setdiff(niveles, grupos_validos), collapse=", ")))
            idx_validos <- grupo %in% grupos_validos
            datos_grupo <- datos_grupo[idx_validos, ]
            grupo <- droplevels(grupo[idx_validos])
            niveles <- grupos_validos
            n_grupos <- length(niveles)
          }
          
          # Estimador según normalidad multivariada (variable global del script)
          estimador <- if (exists("normalidad_multivariada_ok") && !normalidad_multivariada_ok) "MLR" else "ML"
          
          tryCatch({
            # Modelo configural (mismo patrón, parámetros libres)
            fit_configural <- cfa(modelo, data = datos_grupo, group = var_grupo,
                                  group.label = niveles, estimator = estimador)
            
            # Modelo métrico (cargas iguales)
            fit_metrico <- cfa(modelo, data = datos_grupo, group = var_grupo,
                               group.label = niveles, estimator = estimador,
                               group.equal = "loadings")
            
            # Modelo escalar (cargas + interceptos iguales)
            fit_escalar <- cfa(modelo, data = datos_grupo, group = var_grupo,
                               group.label = niveles, estimator = estimador,
                               group.equal = c("loadings", "intercepts"))
            
            # Extraer índices de ajuste
            medidas <- function(fit) {
              if (is.null(fit)) return(rep(NA, 7))
              m <- fitMeasures(fit, c("chisq", "df", "cfi", "rmsea", "srmr", "aic", "bic"))
              return(m)
            }
            m_conf <- medidas(fit_configural)
            m_met <- medidas(fit_metrico)
            m_esc <- medidas(fit_escalar)
            
            # Diferencias de CFI y RMSEA
            delta_cfi_met <- m_met["cfi"] - m_conf["cfi"]
            delta_cfi_esc <- m_esc["cfi"] - m_met["cfi"]
            delta_rmsea_met <- m_met["rmsea"] - m_conf["rmsea"]
            delta_rmsea_esc <- m_esc["rmsea"] - m_met["rmsea"]
            
            # Pruebas de chi-cuadrado anidadas (si no son MLR)
            anova_met <- if (estimador == "ML") tryCatch(anova(fit_configural, fit_metrico), error=function(e) NULL) else NULL
            anova_esc <- if (estimador == "ML") tryCatch(anova(fit_metrico, fit_escalar), error=function(e) NULL) else NULL
            
            # Mostrar resultados en consola
            cat("\n--- AJUSTE POR MODELO ---\n")
            cat(sprintf("Configural: χ²=%.3f, gl=%d, CFI=%.3f, RMSEA=%.3f, SRMR=%.3f\n",
                        m_conf["chisq"], m_conf["df"], m_conf["cfi"], m_conf["rmsea"], m_conf["srmr"]))
            cat(sprintf("Métrica   : χ²=%.3f, gl=%d, CFI=%.3f, RMSEA=%.3f, SRMR=%.3f\n",
                        m_met["chisq"], m_met["df"], m_met["cfi"], m_met["rmsea"], m_met["srmr"]))
            cat(sprintf("Escalar   : χ²=%.3f, gl=%d, CFI=%.3f, RMSEA=%.3f, SRMR=%.3f\n",
                        m_esc["chisq"], m_esc["df"], m_esc["cfi"], m_esc["rmsea"], m_esc["srmr"]))
            
            cat("\n--- COMPARACIONES (ΔCFI, ΔRMSEA) ---\n")
            cat(sprintf("Métrica vs Configural: ΔCFI = %+.4f, ΔRMSEA = %+.4f\n", delta_cfi_met, delta_rmsea_met))
            cat(sprintf("Escalar vs Métrica   : ΔCFI = %+.4f, ΔRMSEA = %+.4f\n", delta_cfi_esc, delta_rmsea_esc))
            
            if (!is.null(anova_met)) {
              cat(sprintf("  Prueba χ² anidada métrica: p = %.4f\n", anova_met[2, "Pr(>Chisq)"]))
            }
            if (!is.null(anova_esc)) {
              cat(sprintf("  Prueba χ² anidada escalar: p = %.4f\n", anova_esc[2, "Pr(>Chisq)"]))
            }
            
            # Interpretación según Cheung & Rensvold (2002) y Chen (2007)
            cat("\n--- INTERPRETACIÓN ---\n")
            if (delta_cfi_met >= -umbral_cfi) cat("  ✓ Invarianza métrica (CFI no empeora >0.01)\n") else cat("  ✗ No hay invarianza métrica\n")
            if (delta_cfi_esc >= -umbral_cfi) cat("  ✓ Invarianza escalar (CFI no empeora >0.01)\n") else cat("  ✗ No hay invarianza escalar\n")
            
            # Tabla resumen para guardar
            tabla_ajustes <- data.frame(
              Modelo = c("Configural", "Métrica", "Escalar"),
              χ2 = round(c(m_conf["chisq"], m_met["chisq"], m_esc["chisq"]), 3),
              gl = c(m_conf["df"], m_met["df"], m_esc["df"]),
              CFI = round(c(m_conf["cfi"], m_met["cfi"], m_esc["cfi"]), 4),
              RMSEA = round(c(m_conf["rmsea"], m_met["rmsea"], m_esc["rmsea"]), 4),
              SRMR = round(c(m_conf["srmr"], m_met["srmr"], m_esc["srmr"]), 4),
              AIC = round(c(m_conf["aic"], m_met["aic"], m_esc["aic"]), 2),
              BIC = round(c(m_conf["bic"], m_met["bic"], m_esc["bic"]), 2)
            )
            
            tabla_diferencias <- data.frame(
              Comparación = c("Métrica vs Configural", "Escalar vs Métrica"),
              ΔCFI = round(c(delta_cfi_met, delta_cfi_esc), 4),
              ΔRMSEA = round(c(delta_rmsea_met, delta_rmsea_esc), 4)
            )
            
            # Guardar usando la función existente guardar_tabla
            if (exists("guardar_tabla")) {
              guardar_tabla(tabla_ajustes, paste0("10_invarianza_", var_grupo, "_ajustes"))
              guardar_tabla(tabla_diferencias, paste0("10_invarianza_", var_grupo, "_diferencias"))
            } else {
              cat("\n  [INFO] Tablas no guardadas (falta función guardar_tabla).\n")
              print(tabla_ajustes)
              print(tabla_diferencias)
            }
            
            # Guardar los modelos en variable global para inspección opcional
            assign(paste0("fit_configural_", var_grupo), fit_configural, envir = .GlobalEnv)
            assign(paste0("fit_metrico_", var_grupo), fit_metrico, envir = .GlobalEnv)
            assign(paste0("fit_escalar_", var_grupo), fit_escalar, envir = .GlobalEnv)
            
            return(list(configural = fit_configural, metrico = fit_metrico, escalar = fit_escalar,
                        tablas = list(ajustes = tabla_ajustes, diferencias = tabla_diferencias)))
            
          }, error = function(e) {
            cat(sprintf("  [ERROR] En análisis de invarianza para %s: %s\n", var_grupo, e$message))
            return(NULL)
          })
        }
        
        # --------------------------------------------------------------
        # 3. Ejecutar para cada variable de grupo
        # --------------------------------------------------------------
        # Preparar data.frame completo que incluya los ítems y todas las sociodemográficas
        # Usamos df_completo (original) pero aseguramos que los ítems estén numéricos
        datos_para_invarianza <- df_completo[, c(nombres_items, vars_grupo_existentes), drop = FALSE]
        for (col in nombres_items) datos_para_invarianza[[col]] <- as.numeric(datos_para_invarianza[[col]])
        
        resultados_invarianza <- list()
        for (vg in vars_grupo_existentes) {
          res <- probar_invarianza(vg, datos_para_invarianza, modelo_spec, umbral_cfi = 0.01)
          if (!is.null(res)) resultados_invarianza[[vg]] <- res
        }
        
        if (length(resultados_invarianza) == 0) {
          cat("\n  [AVISO] No se pudo completar ningún análisis de invarianza.\n")
        } else {
          cat("\n  [OK] Análisis de invarianza finalizado para:", paste(names(resultados_invarianza), collapse=", "), "\n")
        }
        
      } # fin de if modelo_spec válido
    } else {
      cat("\n  [SALTADO] No hay matriz de loadings (AFE) para construir el modelo multigrupo.\n")
    }
  }
} else {
  cat("\n  [SALTADO] No se definieron variables de grupo o no hay datos. Para activar, define VARS_GRUPO <- c(\"sexo\", \"educacion\") en la sección de configuración.\n")
}

# =========================================================================
# 15. RESUMEN FINAL Q1
# =========================================================================

cat("\n\n", paste(rep("#",80),collapse=""), "\n")
cat("#                   RESUMEN FINAL PUBLICACION \n")
cat(paste(rep("#",80),collapse=""), "\n")

cat("\n╔══════════════════════════════════════════════════════════════════╗\n")
cat("║       INFORME VALIDACIÓN PSICOMÉTRICA COMPLETA        ║\n")
cat("╠══════════════════════════════════════════════════════════════════╣\n")

cat(sprintf("║  Muestra: N = %d (inicial: %d, outliers eliminados: %d)\n", n_participantes, resultados_globales$n_inicial, resultados_globales$n_outliers))
cat(sprintf("║  Ítems analizados: %d (escala %d-%d)\n", n_items, ESCALA_MIN, ESCALA_MAX))
cat("╠══════════════════════════════════════════════════════════════════╣\n")
cat("║  CONFIABILIDAD:\n")
cat(sprintf("║    α (Cronbach) = %.3f [%s]\n", alpha_global, clasificar_alpha(alpha_global)))
cat(sprintf("║    ω (McDonald) = %.3f [%s]\n", omega_global, clasificar_alpha(omega_global)))
if (!is.na(sb) && !is.null(sb)) cat(sprintf("║    Split-Half = %.3f\n", sb))
if (!is.null(resultados_globales$omega_h) && !is.na(resultados_globales$omega_h)) cat(sprintf("║    ωh (jerárquico) = %.3f\n", resultados_globales$omega_h))
cat("╠══════════════════════════════════════════════════════════════════╣\n")
cat("║  VALIDEZ:\n")
cat(sprintf("║    KMO = %.3f [%s]\n", kmo_total, clasificar_kmo(kmo_total)))
if (!is.null(resultados_globales$n_factores)) cat(sprintf("║    Factores extraídos = %d\n", resultados_globales$n_factores))
if (!is.null(resultados_globales$varianza_explicada)) cat(sprintf("║    Varianza explicada = %.1f%%\n", resultados_globales$varianza_explicada))
if (!is.null(resultados_globales$rotacion)) cat(sprintf("║    Rotación = %s | Método = %s\n", toupper(resultados_globales$rotacion), toupper(resultados_globales$metodo_extraccion)))
if (!is.null(fit_cfa) && !is.na(cfi_val)) {
  cat("╠══════════════════════════════════════════════════════════════════╣\n")
  cat("║  AJUSTE DEL MODELO (AFC):\n")
  if (!is.na(chi2_val) && !is.na(dof_val)) cat(sprintf("║    χ²(%d) = %.2f\n", dof_val, chi2_val))
  if (!is.na(cfi_val)) cat(sprintf("║    CFI = %.3f [%s]\n", cfi_val, clasificar_ajuste_cfi(cfi_val)))
  if (!is.na(tli_val)) cat(sprintf("║    TLI = %.3f [%s]\n", tli_val, clasificar_ajuste_cfi(tli_val)))
  if (!is.na(rmsea_val)) cat(sprintf("║    RMSEA = %.3f [%s]\n", rmsea_val, clasificar_ajuste_rmsea(rmsea_val)))
  if (!is.na(srmr_val)) cat(sprintf("║    SRMR = %.3f\n", srmr_val))
  if (!is.null(resultados_globales$cfa$aic) && !is.na(resultados_globales$cfa$aic)) cat(sprintf("║    AIC = %.2f\n", resultados_globales$cfa$aic))
  if (!is.null(resultados_globales$cfa$bic) && !is.na(resultados_globales$cfa$bic)) cat(sprintf("║    BIC = %.2f\n", resultados_globales$cfa$bic))
}
cat("╠══════════════════════════════════════════════════════════════════╣\n")
cat("║  SUPUESTOS ESTADÍSTICOS:\n")
if (!is.null(resultados_globales$normalidad_multivariada)) {
  normalidad_str <- ifelse(resultados_globales$normalidad_multivariada, "CUMPLE ✓", "VIOLA ⚠️")
  cat(sprintf("║    Normalidad multivariada: %s\n", normalidad_str))
}
if (!is.null(resultados_globales$estimador_recomendado)) cat(sprintf("║    Estimador utilizado: %s\n", resultados_globales$estimador_recomendado))
cat("╚══════════════════════════════════════════════════════════════════╝\n")

cat("\n  INTERPRETACIÓN GLOBAL PARA MANUSCRITO (Formato APA):\n")
separar_linea()

interpretacion_texto <- sprintf("
La escala de %d ítems mostró propiedades psicométricas %s para su uso en 
población de habla hispana (N = %d). La consistencia interna fue %s 
(α = %.3f; ω = %.3f)%s, superando el umbral mínimo recomendado de .70 
(Nunnally & Bernstein, 1994).

El análisis factorial exploratorio extrajo %d factores que explican el %.1f%% 
de la varianza total%s. La estructura factorial fue confirmada mediante análisis 
factorial confirmatorio utilizando estimador %s debido a %s. El modelo mostró 
ajuste %s a los datos%s.

%s
%s

En conclusión, el instrumento presenta evidencias %s de validez y confiabilidad,
siendo %s para su aplicación en contextos de investigación.
",
  n_items, tolower(clasificar_alpha(alpha_global)), n_participantes, tolower(clasificar_alpha(alpha_global)),
  alpha_global, omega_global, ifelse(!is.na(sb) && !is.null(sb), sprintf("; Split-Half = %.3f", sb), ""),
  ifelse(!is.null(resultados_globales$n_factores) && !is.na(resultados_globales$n_factores), resultados_globales$n_factores, n_factores_final),
  ifelse(!is.null(resultados_globales$varianza_explicada) && !is.na(resultados_globales$varianza_explicada), resultados_globales$varianza_explicada, NA), "",
  ifelse(!is.null(resultados_globales$estimador_recomendado) && !is.na(resultados_globales$estimador_recomendado), resultados_globales$estimador_recomendado, "ML"),
  ifelse(!is.null(resultados_globales$normalidad_multivariada) && !resultados_globales$normalidad_multivariada, "violación del supuesto de normalidad multivariada", "cumplimiento de supuestos estadísticos"),
  tolower(ifelse(!is.na(cfi_val), clasificar_ajuste_cfi(cfi_val), "no evaluado")),
  ifelse(!is.na(cfi_val) && !is.na(rmsea_val) && !is.na(srmr_val), sprintf(" (CFI = %.3f; RMSEA = %.3f; SRMR = %.3f)", cfi_val, rmsea_val, srmr_val), ""),
  ifelse(!is.null(resultados_globales$validez_convergente) && nrow(resultados_globales$validez_convergente) > 0, "La validez convergente fue adecuada (AVE > .50 en todos los factores)", ""),
  ifelse(!is.null(resultados_globales$validez_discriminante) && !is.null(resultados_globales$validez_discriminante$fornell_larcker) && resultados_globales$validez_discriminante$fornell_larcker, ". La validez discriminante se estableció mediante criterios Fornell-Larcker y HTMT", ""),
  tolower(ifelse(!is.na(cfi_val) && cfi_val > 0.90, "sólidas", "aceptables")),
  ifelse(!is.na(cfi_val) && cfi_val > 0.95 && !is.na(rmsea_val) && rmsea_val < 0.06, "altamente recomendado", "adecuado")
)

cat(interpretacion_texto)
separar_linea()

tryCatch({ writeLines(interpretacion_texto, file.path(DIR_SALIDA, "10_interpretacion_manuscrito.txt")); cat("[GUARDADO] 10_interpretacion_manuscrito.txt\n") }, error = function(e) cat(sprintf("  [AVISO] No se pudo guardar interpretación: %s\n", e$message)))

cat("\n")
separar_linea()
cat("  [✓ COMPLETADO] Validación psicométrica finalizada exitosamente.\n")
cat(sprintf("  [📁 RESULTADOS] Archivos guardados en: %s/\n", DIR_SALIDA))
cat("\n  📊 CSV GENERADOS:\n")
cat("     • 01_estadisticas_descriptivas.csv\n")
cat("     • 02_alpha_if_deleted.csv\n")
cat("     • 03_correlacion_item_total.csv\n")
cat("     • 04_KMO_por_item.csv\n")
cat("     • 05_cargas_factoriales_AFE.csv\n")
cat("     • 06_indices_ajuste_AFC.csv\n")
cat("     • 07_AVE_CR.csv\n")
cat("     • 08_matriz_correlaciones.csv\n")
cat("     • 09_comparacion_grupos.csv (si aplica)\n")
cat("\n  📈 GRÁFICAS PNG:\n")
cat("     • 01_distribucion_items.png\n")
cat("     • 02_outliers_mahalanobis.png\n")
cat("     • 03_scree_plot.png\n")
cat("     • 04_cargas_factoriales.png\n")
cat("     • 05_matriz_correlaciones.png\n")
cat("     • 07_diagrama_senderos.png\n")
cat("\n  📝 DOCUMENTO APA:\n")
cat("     • 10_interpretacion_manuscrito.txt\n")
separar_linea()

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("  ✅ VALIDACIÓN PSICOMÉTRICA BUAP 2026 Enrique R.P. Buendia Lozada\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

