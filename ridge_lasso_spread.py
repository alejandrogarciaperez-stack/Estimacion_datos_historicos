import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import DB_DIR, ANALYSIS_DIR

if __name__ == '__main__':
    print("Iniciando modelado con Regresión Ridge y Lasso (2018-2025)...")
    
    # 1. Cargar Datos
    ruta_datos = os.path.join(DB_DIR, 'contratos_regresion_2018_2025.csv')
    if not os.path.exists(ruta_datos):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_datos}. Ejecuta generador_datos_regresion.py primero.")
        
    df = pd.read_csv(ruta_datos)
    
    # Limpieza básica
    df = df.dropna(subset=['spread_pct_real'])
    
    # Preparar datos
    df["euribor_ref_pct"] = df["euribor_12m_pct"].fillna(df["euribor_3m_pct"]).fillna(0.0)
    df["tiene_euribor_12m"] = df["euribor_12m_pct"].notna().astype(int)
    df["tiene_euribor_3m"]  = df["euribor_3m_pct"].notna().astype(int)
    
    # TRANSFORMACIONES por no-linealidad detectada en el diagnostico
    # local_curr_orig_risk_amount -> log(importe): relacion logaritmica con la tasa
    df["log_importe"] = np.log1p(df["local_curr_orig_risk_amount"])  # log(1+x) evita log(0)
    # euribor_ref_pct -> termino cuadratico: comportamiento no lineal a tasas altas
    df["euribor_sq"] = df["euribor_ref_pct"] ** 2
    
    # 2. Seleccionar Features (con variables macroeconomicas y transformadas)
    FEATURES = [
        "contract_month", "client_segment_id", "customer_type_id", 
        "entity_id", "family_id", "counterpart_id", "contract_collateral_type",
        "log_importe", "euribor_ref_pct", "euribor_sq",
        "tiene_euribor_12m", "tiene_euribor_3m",
        "inflation_rate", "gdp_growth", "unemployment_rate"
    ]
    TARGET = "spread_pct_real"
    
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    
    # Identificar tipos de columnas
    categorical_features = [
        "client_segment_id", "customer_type_id", "entity_id", 
        "family_id", "counterpart_id", "contract_collateral_type"
    ]
    numeric_features = [
        "contract_month", "log_importe", "euribor_ref_pct", "euribor_sq",
        "inflation_rate", "gdp_growth", "unemployment_rate"
    ]
    
    # 3. Pipelines de Preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('', (handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    print("\n--- ANÁLISIS DE CORRELACIÓN ---")
    corr_matrix = df[numeric_features + [TARGET]].corr()
    print("Correlación de variables numéricas con la variable objetivo (spread):")
    print(corr_matrix[TARGET].sort_values(ascending=False))
    
    print("\n--- ANÁLISIS DE MULTICOLINEALIDAD (VIF) ---")
    from sklearn.linear_model import LinearRegression
    # Calculamos VIF usando las numéricas (sin nulos)
    X_num = df[numeric_features].dropna()
    for i, col in enumerate(numeric_features):
        X_other = X_num.drop(columns=[col])
        y_vif = X_num[col]
        # Regresión simple para calcular R2
        r2 = LinearRegression().fit(X_other, y_vif).score(X_other, y_vif)
        vif = 1 / (1 - r2) if r2 < 1.0 else float('inf')
        print(f"VIF para {col}: {vif:.2f}")
    print("Nota: Un VIF > 10 indica alta multicolinealidad. Penalizar estos efectos es justo")
    print("el propósito de las regresiones Ridge y Lasso.\n")

    # 4. Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dimensiones de entrenamiento: {X_train.shape}")
    print(f"Dimensiones de prueba: {X_test.shape}")
    
    # 5. Definir Pipelines completos con Interacciones y Modelos
    # PolynomialFeatures creará interacciones (grado 2, solo interacciones X1*X2)
    # Al aplicarlo después del preprocesador, afectará a las numéricas y variables OneHot
    
    print("\nEntrenando Regresión Ridge (con validación cruzada)...")
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('model', RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5))
    ])
    
    ridge_pipeline.fit(X_train, y_train)
    y_pred_ridge = ridge_pipeline.predict(X_test)
    
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f"Resultados Ridge -> R2: {r2_ridge:.4f} | MAE: {mae_ridge:.4f} | Mejor Alpha: {ridge_pipeline.named_steps['model'].alpha_:.4f}")
    
    print("\nEntrenando Regresión Lasso (con validación cruzada)...")
    lasso_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('model', LassoCV(alphas=np.logspace(-3, 1, 10), cv=5, max_iter=2000, n_jobs=-1))
    ])
    
    lasso_pipeline.fit(X_train, y_train)
    y_pred_lasso = lasso_pipeline.predict(X_test)
    
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    print(f"Resultados Lasso -> R2: {r2_lasso:.4f} | MAE: {mae_lasso:.4f} | Mejor Alpha: {lasso_pipeline.named_steps['model'].alpha_:.4f}")
    
    print("\n--- EXPLICACIÓN: PROCESO DE RIDGE Y LASSO ---")
    print("Ridge (Penalización L2): Reduce el valor absoluto de los coeficientes para evitar que unas pocas")
    print("variables dominen el modelo (sobreajuste), lo cual es muy útil cuando hay alta multicolinealidad.")
    print("Lasso (Penalización L1): Penaliza los coeficientes de tal forma que puede reducirlos exactamente")
    print("a CERO. Esto hace que Lasso actúe como un selector automático de variables.")
    
    print("\n--- SELECCIÓN DE VARIABLES EXPLICATIVAS (LASSO) ---")
    lasso_model = lasso_pipeline.named_steps['model']
    coefs = lasso_model.coef_
    ceros = np.sum(coefs == 0)
    totales = len(coefs)
    print(f"De un total de {totales} características (variables originales + sus interacciones),")
    print(f"Lasso ha descartado {ceros} reduciendo su coeficiente a 0.")
    print(f"Han quedado {totales - ceros} variables/interacciones relevantes para explicar el TIE.\n")

    # 6. Seleccionar y Guardar el Mejor Modelo
    print("\n---------------------------------------------------------")
    if mae_ridge < mae_lasso:
        print("El modelo Ridge obtuvo mejor rendimiento en base al MAE.")
        mejor_modelo = ridge_pipeline
        nombre_modelo = "Ridge"
    else:
        print("El modelo Lasso obtuvo mejor rendimiento en base al MAE.")
        mejor_modelo = lasso_pipeline
        nombre_modelo = "Lasso"
        
    ruta_modelo = os.path.join(ANALYSIS_DIR, "mejor_pipeline_ridge_lasso.pkl")
    joblib.dump(mejor_modelo, ruta_modelo)
    print(f"Mejor modelo ({nombre_modelo}) guardado en: {ruta_modelo}")
