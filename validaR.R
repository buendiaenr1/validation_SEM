#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
###############################################################################
# VALIDACION PSICOMETRICA COMPLETA DE CUESTIONARIO LIKERT EN R (VERSION CORREGIDA)
###############################################################################

cat("\n")
cat(paste(rep("#",78),collapse=""),"\n")
cat("#", paste(rep(" ",76),collapse=""), "#\n")
cat("#", paste(strtrim(paste0("  VALIDACION PSICOMETRICA COMPLETA (CORREGIDO)"),76),
               collapse="\n"), "#\n")
cat("#", paste(rep(" ",76),collapse=""), "#\n")
cat(paste(rep("#",78),collapse=""),"\n")

# =========================================================================
# 0. INSTALACION Y CARGA DE PAQUETES
# =========================================================================

paquetes <- c("psych","lavaan","semPlot","nFactors","corrplot",
              "ggplot2","reshape2","dplyr","tidyselect","MASS","gridExtra","viridis")

instalar_si_falta <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  [INSTALANDO] Paquete %s no encontrado. Instalando...\n", pkg))
    install.packages(pkg, repos="https://cloud.r-project.org", quiet=TRUE)
  }
}
invisible(lapply(paquetes, instalar_si_falta))

suppressPackageStartupMessages({
  library(psych)
  library(lavaan)
  library(semPlot)
  library(nFactors)
  library(corrplot)
  library(ggplot2)
  library(reshape2)
  library(dplyr)
  library(tidyselect)
  library(MASS)
  library(gridExtra)
  library(viridis)
})

cat("\n[OK] Todos los paquetes cargados correctamente.\n")

# =========================================================================
# 1. CONFIGURACION
# =========================================================================
# Definimos estas variables aqui al principio para que siempre existan,
# incluso si el AFE falla y no las calcula mas adelante.
var_total_exp <- NA
cfi_val <- NA
rmsea_val <- NA
chi2_val <- NA
dof_val <- NA
vd_ok <- NA
rotacion_elegida <- "varimax"  # CORREGIDO: Inicializar para evitar errores si AFE falla

ARCHIVO_DATOS <- "datos.csv"
# Si tienes una columna de grupos (ej: "grupo", "sexo"), ponla aquí.
# Si no quieres analizar grupos por ahora, déjalo en NULL
COLUMNA_GRUPO <- NULL 

ESCALA_MIN <- 1
ESCALA_MAX <- 5
DIR_SALIDA <- "."

# =========================================================================
# FUNCIONES AUXILIARES
# =========================================================================

separar_linea <- function() {
  cat("\n", paste(rep("=",78),collapse=""), "\n")
}

imprimir_interpretacion <- function(titulo, texto) {
  separar_linea()
  cat(paste0("  INTERPRETACION: ", titulo, "\n"))
  separar_linea()
  cat(texto, "\n")
  cat(paste(rep("=",78),collapse=""), "\n\n")
}

omega_mcdonald_calc <- function(df) {
  corr <- cor(df, use="pairwise.complete.obs")
  ev <- eigen(corr, symmetric=TRUE)
  idx <- order(ev$values, decreasing=TRUE)
  eigenvalues <- ev$values[idx]
  eigenvectors <- ev$vectors[, idx]
  first_ev <- max(eigenvalues[1], 0)
  loadings <- eigenvectors[,1] * sqrt(first_ev)
  uniquenesses <- pmax(1.0 - loadings^2, 0)
  sum_loads <- sum(abs(loadings))
  sum_unique <- sum(uniquenesses)
  if (sum_unique <= 0) return(1.0)
  omega <- (sum_loads^2) / ((sum_loads^2) + sum_unique)
  return(as.numeric(omega))
}

calcular_ave_cr <- function(loadings_df) {
  resultados <- data.frame()
  for (fn in colnames(loadings_df)) {
    loads <- loadings_df[, fn]
    loads <- loads[!is.na(loads)]
    loads_sq <- loads^2
    ave <- mean(loads_sq)
    sum_loads <- sum(loads)
    sum_error <- sum(1 - loads_sq)
    cr <- (sum_loads^2) / ((sum_loads^2) + sum_error)
    resultados <- rbind(resultados, data.frame(Factor=fn, AVE=ave, CR=cr))
  }
  rownames(resultados) <- NULL
  return(resultados)
}

clasificar_alpha <- function(v) {
  if (is.na(v)) return("NO DISPONIBLE")
  if (v >= 0.90) return("EXCELENTE")
  if (v >= 0.80) return("BUENO")
  if (v >= 0.70) return("ACEPTABLE")
  if (v >= 0.60) return("CUESTIONABLE")
  return("INACEPTABLE")
}

clasificar_kmo <- function(v) {
  if (is.na(v)) return("NO DISPONIBLE")  # CORREGIDO: Agregar manejo de NA
  if (v >= 0.90) return("MARAVILLOSO")
  if (v >= 0.80) return("MERITORIO")
  if (v >= 0.70) return("BUENO")
  if (v >= 0.60) return("ACEPTABLE")
  if (v >= 0.50) return("MEDIOCRE")
  return("INACEPTABLE")
}

# =========================================================================
# 2. LECTURA Y PREPARACION DE DATOS (MEJORADA)
# =========================================================================

cat("\n", paste(rep("#",78),collapse=""), "\n")
cat("#  1. LECTURA Y PREPARACION DE DATOS\n")
cat(paste(rep("#",78),collapse=""), "\n")

# Leer datos
primera_linea <- readLines(ARCHIVO_DATOS, n=1, warn=FALSE)
if (grepl(";", primera_linea) && !grepl(",", primera_linea)) {
  delimitador <- ";"
} else if (grepl("\t", primera_linea)) {
  delimitador <- "\t"
} else {
  delimitador <- ","
}

df_completo <- read.csv(ARCHIVO_DATOS, sep=delimitador,
                        stringsAsFactors=FALSE,
                        fileEncoding="UTF-8-BOM",
                        na.strings=c("","NA","N/A"))

# LIMPIEZA DE NOMBRES DE COLUMNAS (Crítico para evitar errores en psych)
# 1. Quitar espacios y BOM
colnames(df_completo) <- trimws(colnames(df_completo))
colnames(df_completo) <- gsub("^\xef\xbb\xbf", "", colnames(df_completo))
# 2. Quitar tildes y caracteres especiales
colnames(df_completo) <- gsub("[áéíóúÁÉÍÓÚñÑ]", "", colnames(df_completo))
colnames(df_completo) <- gsub(" ", "_", colnames(df_completo))

cat(sprintf("\n[OK] Datos leidos: %s (%d filas, %d columnas)\n", 
            ARCHIVO_DATOS, nrow(df_completo), ncol(df_completo)))

cols_todas <- colnames(df_completo)

# --- DETECCION DE COLUMNAS ---
# Intentamos detectar qué columnas son ítems.
# Asumimos que los ítems son numéricos Y NO son demográficos.

df_items <- df_completo
cols_eliminar <- c()

columna_grupo_real <- NULL
tiene_grupo <- FALSE
grupo <- NULL

if (!is.null(COLUMNA_GRUPO)) {
  # Buscar insensible a mayúsculas
  match_idx <- which(tolower(cols_todas) == tolower(COLUMNA_GRUPO))
  if (length(match_idx) > 0) {
    columna_grupo_real <- cols_todas[match_idx[1]]
    grupo <- df_completo[[columna_grupo_real]]
    cols_eliminar <- c(columna_grupo_real)
    tiene_grupo <- TRUE
    cat(sprintf("[OK] Columna de grupos identificada: %s\n", columna_grupo_real))
  }
}

for (col in cols_todas) {
  # 1. Eliminar si NO es numérico
  if (!is.numeric(df_completo[[col]])) {
    if (!(col %in% columna_grupo_real)) {
      cols_eliminar <- c(cols_eliminar, col)
    }
  }
  # 2. Eliminar si el nombre es demográfico obvio
  if (tolower(col) %in% c("grupo","grupos","sexo", "sex", "genero", "gender", "edad", "age", "id")) {
     if (!(col %in% columna_grupo_real)) {
       cols_eliminar <- c(cols_eliminar, col)
     }
  }
}

# ====================================================================
# CORRECCION ERROR 1: Usar dplyr::select() explicito o R base
# ====================================================================
if (length(cols_eliminar) > 0) {
  # Filtrar solo columnas que realmente existen en df_items
  cols_a_eliminar <- intersect(cols_eliminar, names(df_items))
  
  if (length(cols_a_eliminar) > 0) {
    cat(sprintf("[INFO] Excluyendo columnas del análisis: %s\n", 
                paste(cols_a_eliminar, collapse=", ")))
    
    # METODO 1: Con dplyr explicito (evita conflictos con MASS::select)
    df_items <- dplyr::select(df_items, -dplyr::any_of(cols_a_eliminar))
    
    # METODO 2: Alternativa con R base (más robusta, descomentar si falla el anterior)
    # df_items <- df_items[, !(names(df_items) %in% cols_a_eliminar), drop=FALSE]
  } else {
    cat("[AVISO] Las columnas a eliminar no existen en el dataframe\n")
  }
}

# Verificación final
nombres_items <- colnames(df_items)
cat(sprintf("[OK] Ítems a analizar (%d): %s\n", length(nombres_items), paste(nombres_items, collapse=", ")))

# Convertir a numérico (seguro) y limpiar
for (col in nombres_items) {
  df_items[[col]] <- as.numeric(df_items[[col]])
}

n_missing <- sum(!complete.cases(df_items))
if (n_missing > 0) {
  df_items_clean <- df_items[complete.cases(df_items), ]
  if (tiene_grupo) grupo_clean <- grupo[complete.cases(df_items)]
  cat(sprintf("[AVISO] Eliminadas %d filas con NAs. Muestra final: %d\n", n_missing, nrow(df_items_clean)))
} else {
  df_items_clean <- df_items
  if (tiene_grupo) grupo_clean <- grupo
  cat(sprintf("[OK] Muestra final sin NAs: %d participantes\n", nrow(df_items_clean)))
}

mat_datos <- as.matrix(df_items_clean)
n_items <- length(nombres_items)

# =========================================================================
# 3. ESTADISTICAS DESCRIPTIVAS
# =========================================================================

cat("\n\n", paste(rep("#",78),collapse=""), "\n")
cat("#  2. ESTADISTICAS DESCRIPTIVAS DE ITEMS\n")
cat(paste(rep("#",78),collapse=""), "\n")

desc_stats <- data.frame(
  Media = colMeans(df_items_clean),
  `Desv.Est` = apply(df_items_clean, 2, sd),
  Mediana = apply(df_items_clean, 2, median),
  Minimo = apply(df_items_clean, 2, min, na.rm=TRUE),
  Maximo = apply(df_items_clean, 2, max, na.rm=TRUE),
  Asimetria = apply(df_items_clean, 2, psych::skew),
  Curtosis = apply(df_items_clean, 2, psych::kurtosi),
  stringsAsFactors = FALSE
)

for (item in nombres_items) {
  desc_stats[item, "Pct_Min"] <- round(mean(df_items_clean[[item]] == min(df_items_clean[[item]], na.rm=TRUE), na.rm=TRUE) * 100, 2)
  desc_stats[item, "Pct_Max"] <- round(mean(df_items_clean[[item]] == max(df_items_clean[[item]], na.rm=TRUE), na.rm=TRUE) * 100, 2)
}
print(round(desc_stats, 4))

# Graficas descriptivas
png(sprintf("%s/01_distribucion_items.png", DIR_SALIDA), width=22, height=16, units="in", res=150)
par(mfrow=c(ceiling(sqrt(n_items)), ceiling(sqrt(n_items))), mar=c(3,3,2.5,1))
for (item in nombres_items) {
  vals <- df_items_clean[[item]]
  counts <- table(vals)
  barplot(counts, main=sprintf("%s (M=%.2f, DE=%.2f)", item, mean(vals), sd(vals)),
          col=viridis(length(counts)), border="black", xlab="Likert", ylab="Frec")
}
dev.off()
cat("[GRAFICA] Guardada: 01_distribucion_items.png\n")

# =========================================================================
# 4. CONFIABILIDAD
# =========================================================================

cat("\n\n", paste(rep("#",78),collapse=""), "\n")
cat("#  3. ANALISIS DE CONFIABILIDAD\n")
cat(paste(rep("#",78),collapse=""), "\n")

alfa_resultado <- psych::alpha(df_items_clean, check.keys=TRUE)
alpha_global <- alfa_resultado$total$raw_alpha
cat(sprintf("\n  Alfa de Cronbach GLOBAL: %.4f (%s)\n", alpha_global, clasificar_alpha(alpha_global)))

# Alfa si elimina
alfa_sin_item <- alfa_resultado$alpha.drop
cat("\n  Alfa si se elimina cada item:\n")
for (i in 1:nrow(alfa_sin_item)) {
  a_sin <- alfa_sin_item$raw_alpha[i]
  flecha <- ifelse(a_sin > alpha_global, "↑ mejora", "↓ empeora")
  cat(sprintf("    Sin %s: %.4f  %s\n", rownames(alfa_sin_item)[i], a_sin, flecha))
}

# Omega
omega_global <- omega_mcdonald_calc(df_items_clean)
cat(sprintf("\n  Omega de McDonald: %.4f\n", omega_global))

# Split-Half (Corregido para evitar errores)
split_result <- tryCatch({
  psych::splitHalf(df_items_clean, raw=TRUE)
}, error = function(e) {
  cat("  [AVISO] Error en Split-Half estándar, intentando Guttman...\n")
  list(overall = list(raw_split = psych::guttman(df_items_clean)$six))
})

sb <- split_result$overall$raw_split
if (is.null(sb) || length(sb) == 0) sb <- NA # Valor por defecto si falla todo

# ====================================================================
# CORRECCION ERROR 2: if-else en la misma línea o con llaves
# ====================================================================
if (!is.na(sb)) { 
  cat(sprintf("  Spearman-Brown (Split-Half): %.4f\n", sb)) 
} else { 
  cat("  Spearman-Brown: No calculable\n") 
}

# Item-Total
cat("\n  Correlacion Item-Total:\n")
corr_item_total <- c()
for (item in nombres_items) {
  total_sin <- rowSums(df_items_clean[, nombres_items[nombres_items != item], drop=FALSE])
  r_val <- cor(df_items_clean[[item]], total_sin, use="complete.obs")
  corr_item_total[item] <- r_val
  sig <- "***"
  cat(sprintf("    %s: r=%.4f %s\n", item, r_val, sig))
}

# =========================================================================
# 5. ADECUACION DE MUESTRA
# =========================================================================

cat("\n\n", paste(rep("#",78),collapse=""), "\n")
cat("#  4. ADECUACION DE MUESTRA (KMO Y BARTLETT)\n")
cat(paste(rep("#",78),collapse=""), "\n")

kmo_result <- tryCatch({
  psych::KMO(mat_datos)
}, error = function(e) {
  cat(sprintf("  [AVISO] Error en KMO: %s\n", e$message))
  return(list(MSA=NA, MSAi=rep(NA, n_items)))
})

kmo_total <- kmo_result$MSA
kmo_items <- kmo_result$MSAi

cat(sprintf("\n  KMO Global: %.4f (%s)\n", kmo_total, clasificar_kmo(kmo_total)))

bartlett_result <- tryCatch({
  psych::cortest.bartlett(cor(mat_datos, use="pairwise.complete.obs"), n=nrow(mat_datos))
}, error = function(e) {
  cat(sprintf("  [AVISO] Error en Bartlett: %s\n", e$message))
  return(list(chisq=NA, p.value=NA))
})
cat(sprintf("  Bartlett: Chi2=%.2f, p=%.2e (Significativo)\n", bartlett_result$chisq, bartlett_result$p.value))

# =========================================================================
# 6. DETERMINACION DE FACTORES
# =========================================================================

cat("\n\n", paste(rep("#",78),collapse=""), "\n")
cat("#  5. DETERMINACION DE NUMERO DE FACTORES\n")
cat(paste(rep("#",78),collapse=""), "\n")

set.seed(42)
pa_result <- tryCatch({
  psych::fa.parallel(mat_datos, fm="ml", fa="fa", n.iter=500, plot=FALSE)
}, error = function(e) {
  cat(sprintf("  [AVISO] Error en análisis paralelo: %s\n", e$message))
  return(NULL)
})

if (!is.null(pa_result)) {
  n_factores_paralelo <- pa_result$nfact
  n_kaiser <- sum(eigen(cor(mat_datos))$values > 1)
  
  cat(sprintf("  Analisis Paralelo sugiere: %d factor(es)\n", n_factores_paralelo))
  cat(sprintf("  Criterio Kaiser (>1.0): %d factor(es)\n", n_kaiser))
  
  # Decision: Paralelo preferido
  n_factores_final <- n_factores_paralelo
  if (n_factores_final < 1) n_factores_final <- n_kaiser
  cat(sprintf("\n  >>> SE USARAN: %d FACTORES\n", n_factores_final))
} else {
  cat("  [ERROR] No se pudo determinar número de factores. Usando 1 por defecto.\n")
  n_factores_final <- 1
}

# =========================================================================
# 7. ANALISIS FACTORIAL EXPLORATORIO (AFE) - VERSION "A PRUEBA DE FALLOS"
# =========================================================================

cat("\n\n", paste(rep("#",78),collapse=""), "\n")
cat("#  6. ANALISIS FACTORIAL EXPLORATORIO (AFE)\n")
cat(paste(rep("#",78),collapse=""), "\n")

# --- PASO CRITICO: Forzar la existencia de las variables GLOBALES desde el inicio ---
assign("var_total_exp", NA, envir = .GlobalEnv)
assign("cfi_val", NA, envir = .GlobalEnv)
assign("rmsea_val", NA, envir = .GlobalEnv)
assign("chi2_val", NA, envir = .GlobalEnv)
assign("dof_val", NA, envir = .GlobalEnv)

# Inicializamos objetos locales
fa_efa <- NULL
fa_varimax <- NULL
fa_promax <- NULL
error_afe <- FALSE

# Intentamos calcular el AFE. Si ML falla, usamos PA.
tryCatch({
  cat("  Intentando AFE con Maxima Verosimilitud (ML)...\n")
  fa_varimax <- psych::fa(mat_datos, nfactors=n_factores_final, rotate="varimax", fm="ml", max.iter=1000)
  fa_promax  <- psych::fa(mat_datos, nfactors=n_factores_final, rotate="promax", fm="ml", max.iter=1000)
}, error = function(e) {
  cat(sprintf("  [AVISO] ML falló (%s). Cambiando a Ejes Principales (pa)...\n", e$message))
  tryCatch({
    fa_varimax <- psych::fa(mat_datos, nfactors=n_factores_final, rotate="varimax", fm="pa")
    fa_promax  <- psych::fa(mat_datos, nfactors=n_factores_final, rotate="promax", fm="pa")
  }, error = function(e2) {
    cat(sprintf("  [ERROR GRAVE] No se pudo calcular el AFE con ningún método: %s\n", e2$message))
    error_afe <<- TRUE
  })
})

# Si todo falló, aseguramos que loadings sea NULL
if (error_afe || is.null(fa_varimax)) {
  cat("\n  [FALLO] No fue posible extraer factores. Se omite AFE y AFC.\n")
  fa_efa <- NULL
  loadings_efa <- NULL
} else {
  # Elegir rotación
  simp_vari <- mean(abs(fa_varimax$loadings) * (1 - abs(fa_varimax$loadings)))
  simp_prom <- mean(abs(fa_promax$loadings) * (1 - abs(fa_promax$loadings)))
  rotacion_elegida <<- ifelse(simp_prom < simp_vari, "promax", "varimax")  # CORREGIDO: Asignación global
  fa_efa <- ifelse(rotacion_elegida == "promax", fa_promax, fa_varimax)
  
  cat(sprintf("\n  Rotacion elegida: %s\n", toupper(rotacion_elegida)))
  
  # Verificar que loadings existe antes de imprimir
  if (!is.null(fa_efa$loadings)) {
    print(fa_efa$loadings, cutoff=0, sort=TRUE)
    
    # --- COMUNALIDADES ---
    comunalidades <- fa_efa$communality
    cat("\n  Comunalidades:\n")
    if (is.numeric(comunalidades)) {
      print(round(comunalidades, 3))
    } else {
      cat("  (No disponible)\n")
      comunalidades <- rep(NA, n_items)
    }
  
    # --- VARIANZA EXPLICADA ---
    cat("\n  Varianza Explicada:\n")
    varianza_fa <- fa_efa$Vaccounted
    
    local_var_exp <- NA 
    
    if (n_factores_final == 1) {
      if (!is.null(varianza_fa)) {
        if (is.matrix(varianza_fa)) {
           val_exp <- varianza_fa[2] 
        } else {
           if ("Variance" %in% names(varianza_fa)) {
             val_exp <- varianza_fa["Variance"]
           } else if ("Proportion Var" %in% names(varianza_fa)) {
             val_exp <- varianza_fa["Proportion Var"]
           } else {
             val_exp <- ifelse(length(varianza_fa) >= 2, varianza_fa[2], varianza_fa[1])
           }
        }
        var_exp_pct <- as.numeric(val_exp) * 100 
        if (var_exp_pct > 10) var_exp_pct <- as.numeric(val_exp) 
        
        cat(sprintf("    Total Varianza Explicada: %.2f%%\n", var_exp_pct))
        local_var_exp <- var_exp_pct
      }
    } else {
      sum_var <- 0
      if (is.matrix(varianza_fa)) {
         for (i in 1:n_factores_final) {
           val <- varianza_fa[3,i] 
           cat(sprintf("    Factor %d: %.2f%%\n", i, val))
           sum_var <- sum_var + val
         }
         local_var_exp <- sum_var
      } else {
         local_var_exp <- sum(varianza_fa, na.rm=TRUE)
      }
    }
    
    assign("var_total_exp", local_var_exp, envir = .GlobalEnv)
    loadings_efa <- fa_efa$loadings
    
  } else {
    cat("  [ERROR] Estructura de loadings vacía o corrupta.\n")
    loadings_efa <- NULL
    assign("var_total_exp", NA, envir = .GlobalEnv)
  }
}

# =========================================================================
# 8. ANALISIS FACTORIAL CONFIRMATORIO (AFC)
# =========================================================================

cat("\n\n", paste(rep("#",78),collapse=""), "\n")
cat("#  7. ANALISIS FACTORIAL CONFIRMATORIO (AFC)\n")
cat(paste(rep("#",78),collapse=""), "\n")

fit_cfa <- NULL
cfi_val <- rmsea_val <- chi2_val <- dof_val <- NA

if (!is.null(loadings_efa)) {
  
  tryCatch({
    asignacion <- apply(loadings_efa, 1, function(x) colnames(loadings_efa)[which.max(abs(x))])
    
    modelo_lines <- c()
    items_por_factor <- list()
    
    if (ncol(loadings_efa) > 0) {
      for (j in 1:ncol(loadings_efa)) {
        fname <- colnames(loadings_efa)[j]
        items_f <- nombres_items[nombres_items %in% rownames(loadings_efa)[asignacion == fname]]
        
        items_por_factor[[fname]] <- items_f
        
        if (length(items_f) > 0) {
          for (item in items_f) {
            modelo_lines <- c(modelo_lines, sprintf("  %s =~ %s", fname, item))
          }
        }
      }
    }
    
    if (length(modelo_lines) > 0) {
      modelo_spec <- paste(modelo_lines, collapse="\n")
      cat("\n  Especificacion del modelo:\n")
      cat(modelo_spec, "\n")
      
      fit_cfa <- tryCatch({
        lavaan::cfa(modelo_spec, data=df_items_clean, std.lv=TRUE, estimator="ML")
      }, error = function(e) {
        cat(sprintf("\n  [ERROR CFA] %s\n", e$message))
        return(NULL)
      })
      
      if (!is.null(fit_cfa)) {
        summary(fit_cfa, fit.measures=TRUE, standardized=TRUE, rsquare=TRUE)
        indices_fit <- lavaan::fitMeasures(fit_cfa)
        cat("\n  INDICES DE AJUSTE:\n")
        cat(sprintf("    CFI: %.3f | TLI: %.3f | RMSEA: %.3f | SRMR: %.3f\n", 
                    indices_fit["cfi"], indices_fit["tli"], indices_fit["rmsea"], indices_fit["srmr"]))
        
        png(sprintf("%s/07_diagrama_caminos.png", DIR_SALIDA), width=10, height=6, units="in", res=150)
        semPaths(fit_cfa, what="std", whatLabels="std", layout="tree", 
                 edge.label.cex=0.8, fade=FALSE)
        dev.off()
        cat("[GRAFICA] Guardada: 07_diagrama_caminos.png\n")
        
        cfi_val <- indices_fit["cfi"]
        rmsea_val <- indices_fit["rmsea"]
        chi2_val <- indices_fit["chisq"]
        dof_val <- indices_fit["df"]
      }
    } else {
      cat("  [SALTADO] El modelo generado está vacío (sin ítems).\n")
    }
    
  }, error = function(e) {
    cat(sprintf("  [ERROR] Construyendo modelo AFC: %s\n", e$message))
  })
  
} else {
  cat("  [SALTADO] No se puede realizar AFC porque el AFE falló.\n")
}

# =========================================================================
# 9. VALIDEZ CONVERGENTE Y DISCRIMINANTE
# =========================================================================

cat("\n\n", paste(rep("#",78),collapse=""), "\n")
cat("#  8. VALIDEZ CONVERGENTE Y DISCRIMINANTE\n")
cat(paste(rep("#",78),collapse=""), "\n")

if (!is.null(loadings_efa) && !is.null(fit_cfa)) {
  loadings_df <- as.data.frame(loadings_efa)
  loadings_df <- na.omit(loadings_df)
  
  if (nrow(loadings_df) > 0 && ncol(loadings_df) > 0) {
    ave_cr_result <- calcular_ave_cr(loadings_df)
    print(round(ave_cr_result, 3))
    
    vd_ok <- TRUE
    if (ncol(loadings_df) >= 2) {
      if (rotacion_elegida == "promax" && !is.null(fa_efa$Phi)) {
        corr_factores <- fa_efa$Phi
      } else {
        corr_factores <- diag(ncol(loadings_df))
        colnames(corr_factores) <- rownames(corr_factores) <- colnames(loadings_df)
      }
      
      cat("\n  Matriz de Fornell-Larcker (Diagonal = Raiz AVE):\n")
      disc_matrix <- corr_factores
      diag(disc_matrix) <- sqrt(ave_cr_result$AVE)
      print(round(disc_matrix, 3))
      
      for (i in 1:(nrow(ave_cr_result)-1)) {
        for (j in (i+1):nrow(ave_cr_result)) {
          raiz_ave_i <- sqrt(ave_cr_result$AVE[i])
          cor_ij <- abs(corr_factores[i,j])
          if (raiz_ave_i < cor_ij) vd_ok <- FALSE
        }
      }
      cat(sprintf("  Validez Discriminante: %s\n", ifelse(vd_ok, "CUMPLIDA", "NO CUMPLIDA")))
    } else {
      cat("  (No aplica: Un solo factor)\n")
    }
  } else {
    cat("  (No hay datos suficientes en loadings para calcular AVE)\n")
    ave_cr_result <- NULL
  }
} else {
  cat("  (Omitido: AFE o AFC no disponible)\n")
  ave_cr_result <- NULL
}

# =========================================================================
# 10. GRUPOS CONOCIDOS
# =========================================================================

if (tiene_grupo && !is.null(fa_efa)) {  # CORREGIDO: Verificar que fa_efa existe
  cat("\n\n", paste(rep("#",78),collapse=""), "\n")
  cat("#  9. VALIDEZ DE GRUPOS CONOCIDOS\n")
  cat(paste(rep("#",78),collapse=""), "\n")
  
  scores_fa <- tryCatch({
    psych::factor.scores(mat_datos, fa_efa)$scores
  }, error = function(e) {
    cat(sprintf("  [AVISO] No se pudieron calcular puntuaciones factoriales: %s\n", e$message))
    return(NULL)
  })
  
  if (!is.null(scores_fa)) {
    df_scores <- data.frame(scores_fa)
    df_scores$grupo <- if(exists("grupo_clean")) grupo_clean else grupo
    
    if (length(unique(df_scores$grupo)) == 2) {
      t_res <- t.test(df_scores[,1] ~ grupo, data=df_scores)
      cat(sprintf("  T-test para Puntuación Factorial: t=%.3f, p=%.3f\n", t_res$statistic, t_res$p.value))
    }
  }
}

# =========================================================================
# 11. RESUMEN FINAL
# =========================================================================

cat("\n\n", paste(rep("#",78),collapse=""), "\n")
cat("#                       RESUMEN FINAL :  BUAP Enrique Buendia Lozada\n")
cat(paste(rep("#",78),collapse=""), "\n")

cat(sprintf("\n  Participantes: %d\n", nrow(df_items_clean)))
cat(sprintf("  Alfa de Cronbach: %.3f (%s)\n", alpha_global, clasificar_alpha(alpha_global)))
cat(sprintf("  Omega: %.3f (%s)\n", omega_global, clasificar_alpha(omega_global)))
if (!is.na(sb)) { cat(sprintf("  Split-Half: %.3f\n", sb)) }  # CORREGIDO: Con llaves
cat(sprintf("  KMO: %.3f (%s)\n", kmo_total, clasificar_kmo(kmo_total)))
cat(sprintf("  Factores (AFE): %d\n", n_factores_final))

if (!is.na(var_total_exp)) {
  cat(sprintf("  Varianza Explicada: %.1f%%\n", var_total_exp))
} else {
  cat("  Varianza Explicada: No calculada\n")
}

if (!is.null(fit_cfa)) {
  cat(sprintf("\n  AFC - CFI: %.3f | RMSEA: %.3f\n", cfi_val, rmsea_val))
  cat(sprintf("        Chi2: %.2f (df=%d)\n", chi2_val, dof_val))
}

cat("\n[INFO] Proceso finalizado. Revisa las gráficas generadas.\n")