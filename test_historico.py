import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from config import ANALYSIS_DIR
from data_utils import generar_datos

if __name__ == '__main__':
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    print(f"Generando datos para {years}...")
    df_hist = generar_datos(years, 2000)
    df_hist = df_hist.dropna(subset=['spread_pct_real'])
    print(f"Generados {len(df_hist)} registros válidos.")
    
    # Preparar datos para el modelo (igual que en random_forest_spread.py)
    df_hist["euribor_ref_pct"] = df_hist["euribor_12m_pct"].fillna(df_hist["euribor_3m_pct"]).fillna(0.0)
    df_hist["tiene_euribor_12m"] = df_hist["euribor_12m_pct"].notna().astype(int)
    df_hist["tiene_euribor_3m"]  = df_hist["euribor_3m_pct"].notna().astype(int)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # ─────────────────────────────────────────────────────────────────────────
    print("\nCargando el pipeline optimizado (Random Forest + Preprocesamiento)...")
    ruta_modelo = os.path.join(ANALYSIS_DIR, "mejor_pipeline_rf.pkl")
    
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en {ruta_modelo}. Ejecuta primero random_forest_spread.py")
        
    mejor_modelo_rf = joblib.load(ruta_modelo)
    
    # Definir las features EXACTAMENTE como entraron al modelo original
    FEATURES = [
        "contract_month", "client_segment_id", "customer_type_id", 
        "entity_id", "family_id", "counterpart_id", "contract_collateral_type",
        "local_curr_orig_risk_amount", "euribor_ref_pct", 
        "tiene_euribor_12m", "tiene_euribor_3m"
    ]
    
    X_test_hist = df_hist[FEATURES].copy()
    
    # Predecir directamente: el pipeline aplica automáticamente el OneHotEncoder, 
    # la imputación de nulos y el modelo Random Forest con sus mejores hiperparámetros.
    print("Generando predicciones de spread...")
    df_hist['spread_pred'] = mejor_modelo_rf.predict(X_test_hist)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Calcular TEDR predicho (Esto continúa igual que antes)
    df_hist['tin_pred_dec'] = (df_hist['spread_pred'] / 100.0) + (df_hist['ref_pct'] / 100.0)
    df_hist['tedr_pred_pct'] = ((1 + df_hist['tin_pred_dec'] / df_hist['m_freq'])**df_hist['m_freq'] - 1) * 100
    
    # Comparar TEDR Real vs Predicho por año
    print("\nResultados de predicción TEDR por año:")
    for yr in sorted(df_hist['year'].unique()):
        df_yr = df_hist[df_hist['year'] == yr]
        if len(df_yr) > 0:
            r2 = r2_score(df_yr['tedr_pct'], df_yr['tedr_pred_pct'])
            mae = mean_absolute_error(df_yr['tedr_pct'], df_yr['tedr_pred_pct'])
            print(f"  Año {yr} -> R2: {r2:.4f} | MAE: {mae:.4f} %")
            
    print("\nResultados de predicción TEDR por Tipo de Contrato (Familia):")
    for fam in sorted(df_hist['family_id'].dropna().unique()):
        df_fam = df_hist[df_hist['family_id'] == fam]
        if len(df_fam) > 0:
            mae = mean_absolute_error(df_fam['tedr_pct'], df_fam['tedr_pred_pct'])
            print(f"  {fam:<15} -> Error Medio (MAE): {mae:.4f} %")
    
    # Global
    r2_global = r2_score(df_hist['tedr_pct'], df_hist['tedr_pred_pct'])
    mae_global = mean_absolute_error(df_hist['tedr_pct'], df_hist['tedr_pred_pct'])
    print(f"\nResultados Globales (2010-2017):")
    print(f"R2 : {r2_global:.4f}")
    print(f"MAE: {mae_global:.4f} %")
    
    # Guardar resultados
    ruta_salida = os.path.join(ANALYSIS_DIR, "resultados_historicos_2010_2017.csv")
    df_hist.to_csv(ruta_salida, index=False)
    print(f"\nResultados exportados a: {ruta_salida}")
    
    # Mostrar muestra
    print("\nMuestra de predicciones:")
    print(df_hist[['year', 'family_id', 'tedr_pct', 'tedr_pred_pct', 'spread_pct_real', 'spread_pred']].head(10).to_string())

    # Generar y guardar gráfica de MAE por tipo de contrato
    print("\nGenerando gráfica de errores por tipo de contrato...")
    resultados_grafica = []
    for fam in sorted(df_hist['family_id'].dropna().unique()):
        df_fam = df_hist[df_hist['family_id'] == fam]
        if len(df_fam) > 0:
            mae = mean_absolute_error(df_fam['tedr_pct'], df_fam['tedr_pred_pct'])
            resultados_grafica.append({'family_id': fam, 'MAE': mae})
            
    df_resultados = pd.DataFrame(resultados_grafica).sort_values('MAE')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_resultados['family_id'].astype(str), df_resultados['MAE'], color='skyblue', edgecolor='black')
    plt.xlabel('Tipo de Contrato (family_id)', fontsize=12)
    plt.ylabel('Error Medio Absoluto (MAE) en %', fontsize=12)
    plt.title('Error de Predicción (MAE) por Tipo de Contrato', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    
    # Crear carpeta 'random forest' si no existe
    rf_dir = os.path.join(ANALYSIS_DIR, "random forest")
    os.makedirs(rf_dir, exist_ok=True)
    
    # Guardar gráfica
    out_path = os.path.join(rf_dir, "error_mae_por_familia.png")
    plt.savefig(out_path, dpi=300)
    print(f"Gráfica guardada exitosamente en: {out_path}")
############----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

    
    # 2. Preparación de variables base
    df_hist["euribor_ref_pct"] = df_hist["euribor_12m_pct"].fillna(df_hist["euribor_3m_pct"]).fillna(0.0)
    df_hist["log_importe"] = np.log1p(df_hist["local_curr_orig_risk_amount"])
    df_hist["euribor_sq"] = df_hist["euribor_ref_pct"] ** 2

    # 3. Carga del Modelo Ridge y Preprocesadores
    print("\nCargando el pipeline optimizado de Regresión Ridge...")
    ruta_modelo = os.path.join(ANALYSIS_DIR, "mejor_pipeline_tie_seleccion.pkl")
    
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en {ruta_modelo}. Ejecuta ridge_lasso_tie.py primero.")
        
    artefacto_ridge = joblib.load(ruta_modelo)
    mejor_modelo_ridge = artefacto_ridge['pipeline']
    num_imputer = artefacto_ridge['imputer_num']
    cat_pipeline = artefacto_ridge['encoder_cat']
    features_ganadoras = artefacto_ridge['features_seleccionadas']

    # 4. Reconstrucción de la Matriz de Interacciones (Aplicando solo .transform())
    NUMERICAS = ["log_importe", "euribor_sq", "inflation_rate", "gdp_growth", "unemployment_rate", "contract_month"]
    CATEGORICAS = ["client_segment_id", "customer_type_id", "entity_id", "family_id", "counterpart_id", "contract_collateral_type"]

    # Transformación coherente con los datos de entrenamiento
    num_raw = num_imputer.transform(df_hist[NUMERICAS])
    cat_raw = cat_pipeline.transform(df_hist[CATEGORICAS])

    POOL_HIST = {}
    for i, col in enumerate(NUMERICAS): POOL_HIST[col] = num_raw[:, i]
    for j, col in enumerate(CATEGORICAS): POOL_HIST[col] = cat_raw[:, j]
    for i, col_num in enumerate(NUMERICAS):
        for j, col_cat in enumerate(CATEGORICAS):
            POOL_HIST[f"{col_num} _X_ {col_cat}"] = num_raw[:, i] * cat_raw[:, j]
            
    # Ensamblar columnas seleccionadas
    X_test_hist = pd.DataFrame(np.column_stack([POOL_HIST[f] for f in features_ganadoras]), columns=features_ganadoras)
    
    # 5. Predicción
    print("Generando predicciones directas de TEDR...")
    df_hist['tedr_pred_pct'] = mejor_modelo_ridge.predict(X_test_hist)
    
    # Neutralizamos columnas de spread para mantener compatibilidad con la estructura
    df_hist['spread_pred'] = 0.0
    df_hist['spread_pct_real'] = 0.0 
    
    # 6. Evaluación de Resultados
    print("\nResultados de predicción TEDR por año:")
    for yr in sorted(df_hist['year'].unique()):
        df_yr = df_hist[df_hist['year'] == yr]
        if len(df_yr) > 0:
            r2 = r2_score(df_yr['tedr_pct'], df_yr['tedr_pred_pct'])
            mae = mean_absolute_error(df_yr['tedr_pct'], df_yr['tedr_pred_pct'])
            print(f"  Año {yr} -> R2: {r2:.4f} | MAE: {mae:.4f} %")
            
    print("\nResultados de predicción TEDR por Tipo de Contrato (Familia):")
    for fam in sorted(df_hist['family_id'].dropna().unique()):
        df_fam = df_hist[df_hist['family_id'] == fam]
        if len(df_fam) > 0:
            mae = mean_absolute_error(df_fam['tedr_pct'], df_fam['tedr_pred_pct'])
            print(f"  {fam:<15} -> Error Medio (MAE): {mae:.4f} %")
    
    # Métricas Globales
    r2_global = r2_score(df_hist['tedr_pct'], df_hist['tedr_pred_pct'])
    mae_global = mean_absolute_error(df_hist['tedr_pct'], df_hist['tedr_pred_pct'])
    print(f"\nResultados Globales (2010-2017):")
    print(f"R2 : {r2_global:.4f}")
    print(f"MAE: {mae_global:.4f} %")
    
    # 7. Exportación y Gráficas
    ruta_salida = os.path.join(ANALYSIS_DIR, "resultados_historicos_ridge_2010_2017.csv")
    df_hist.to_csv(ruta_salida, index=False)
    print(f"\nResultados exportados a: {ruta_salida}")
    
    print("\nMuestra de predicciones:")
    print(df_hist[['year', 'family_id', 'tedr_pct', 'tedr_pred_pct']].head(10).to_string())

    print("\nGenerando gráfica de errores por tipo de contrato...")
    resultados_grafica = []
    for fam in sorted(df_hist['family_id'].dropna().unique()):
        df_fam = df_hist[df_hist['family_id'] == fam]
        if len(df_fam) > 0:
            mae = mean_absolute_error(df_fam['tedr_pct'], df_fam['tedr_pred_pct'])
            resultados_grafica.append({'family_id': fam, 'MAE': mae})
            
    df_resultados = pd.DataFrame(resultados_grafica).sort_values('MAE')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_resultados['family_id'].astype(str), df_resultados['MAE'], color='coral', edgecolor='black')
    plt.xlabel('Tipo de Contrato (family_id)', fontsize=12)
    plt.ylabel('Error Medio Absoluto (MAE) en %', fontsize=12)
    plt.title('Error de Predicción Ridge (MAE) por Tipo de Contrato', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    
    # Crear carpeta 'ridge_models' si no existe
    ridge_dir = os.path.join(ANALYSIS_DIR, "ridge_models")
    os.makedirs(ridge_dir, exist_ok=True)
    
    out_path = os.path.join(ridge_dir, "error_mae_por_familia_historico.png")
    plt.savefig(out_path, dpi=300)
    print(f"Gráfica guardada exitosamente en: {out_path}")