import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from config import DB_DIR, ANALYSIS_DIR

if __name__ == '__main__':
    print("=" * 90)
    print("  ENGINE FORWARD STEPWISE REAL (RMSE) - VIF, TRAIN/TEST SPLIT Y MODO TURBO")
    print("=" * 90)

    # ────────────────────────────────────────────────────────────────
    #  1. CARGA, PREPARACIÓN Y DIVISIÓN DE DATOS (TRAIN/TEST SPLIT)
    # ────────────────────────────────────────────────────────────────
    ruta_datos = os.path.join(DB_DIR, 'contratos_regresion_2018_2025.csv')
    if not os.path.exists(ruta_datos):
        raise FileNotFoundError(f"No se encontró el archivo histórico: {ruta_datos}")

    df = pd.read_csv(ruta_datos)
    df = df.dropna(subset=['tedr_pct'])
    df["euribor_ref_pct"] = df["euribor_12m_pct"].fillna(df["euribor_3m_pct"]).fillna(0.0)

    df["log_importe"] = np.log1p(df["local_curr_orig_risk_amount"])
    df["euribor_sq"] = df["euribor_ref_pct"] ** 2

    TARGET = "tedr_pct"

    TODAS_NUMERICAS = [
        "log_importe", "euribor_sq", "inflation_rate", 
        "gdp_growth", "unemployment_rate", "contract_month"
    ]
    TODAS_CATEGORICAS = [
        "client_segment_id", "customer_type_id", "entity_id",
        "family_id", "counterpart_id", "contract_collateral_type"
    ]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_train = df_train[TARGET].values
    y_test = df_test[TARGET].values
    
    print(f"\n[INFO] División completada: {len(df_train)} contratos para Train y {len(df_test)} para Test.")

    # ────────────────────────────────────────────────────────────────
    #  2. DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF SOBRE TRAIN)
    # ────────────────────────────────────────────────────────────────
    print("\n[FASE 1] DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF SOBRE TRAIN)")
    print("  Nota: Ridge mitigará esta colinealidad matemáticamente.")
    print("-" * 90)
    X_num_vif = df_train[TODAS_NUMERICAS].dropna()
    for col in TODAS_NUMERICAS:
        X_other = X_num_vif.drop(columns=[col])
        r2 = LinearRegression().fit(X_other, X_num_vif[col]).score(X_other, X_num_vif[col])
        vif = 1 / (1 - r2) if r2 < 1.0 else float('inf')
        print(f"  VIF({col:<25}): {vif:6.2f}  {'⚠  ALTA' if vif>10 else 'OK'}")

    # ────────────────────────────────────────────────────────────────
    #  3. PREPROCESAMIENTO BASE (Imputación y Codificación Raw)
    # ────────────────────────────────────────────────────────────────
    print("\n[FASE 2] PREPROCESAMIENTO Y CREACIÓN DE INTERACCIONES")
    print("-" * 90)
    
    num_imputer = SimpleImputer(strategy='median')
    num_raw_train = num_imputer.fit_transform(df_train[TODAS_NUMERICAS])
    num_raw_test = num_imputer.transform(df_test[TODAS_NUMERICAS])
    
    cat_pipeline = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    cat_raw_train = cat_pipeline.fit_transform(df_train[TODAS_CATEGORICAS])
    cat_raw_test = cat_pipeline.transform(df_test[TODAS_CATEGORICAS])

    def construir_pool(num_matrix, cat_matrix):
        pool = {}
        for i, col in enumerate(TODAS_NUMERICAS): pool[col] = num_matrix[:, i]
        for j, col in enumerate(TODAS_CATEGORICAS): pool[col] = cat_matrix[:, j]
        for i, col_num in enumerate(TODAS_NUMERICAS):
            for j, col_cat in enumerate(TODAS_CATEGORICAS):
                pool[f"{col_num} _X_ {col_cat}"] = num_matrix[:, i] * cat_matrix[:, j]
        return pool

    POOL_TRAIN = construir_pool(num_raw_train, cat_raw_train)
    POOL_TEST = construir_pool(num_raw_test, cat_raw_test)

    print(f"  > Universo total a escanear: {len(POOL_TRAIN)} variables e interacciones individuales.")

    # ────────────────────────────────────────────────────────────────
    #  4. BÚSQUEDA AVANZADA FORWARD STEPWISE REAL (SOBRE TRAIN)
    # ────────────────────────────────────────────────────────────────
    print("\n[FASE 3] ADUANA ALGORÍTMICA ITERATIVA (MODO TURBO EN TRAIN)")
    print("-" * 90)

    UMBRAL_MEJORA = 0.09
    TOLERANCIA = 1e-5

    features_disponibles = list(POOL_TRAIN.keys())
    features_activas = []
    
    rmse_actual = np.sqrt(mean_squared_error(y_train, np.full_like(y_train, y_train.mean())))
    print(f"  > RMSE Inicial del modelo (sin variables): {rmse_actual:.4f} %")
    print(f"  Criterio de parada: Detener si la ganancia marginal es < {UMBRAL_MEJORA}%\n")
    
    print(f"  {'Ronda':<6} | {'Mejor Variable Candidata Encontrada':<45} | {'RMSE (%)':>8} | {'Δ RMSE':>8} | {'R²':>6}")
    print(f"  {'-'*6}+{'-'*47}+{'-'*10}+{'-'*10}+{'-'*8}")

    resultados_grafica = []
    ronda = 1

    while len(features_disponibles) > 0:
        mejor_feature_ronda = None
        mejor_rmse_ronda = float('inf')
        mejor_r2_ronda = -float('inf')

        if features_activas:
            X_base_np = np.column_stack([POOL_TRAIN[f] for f in features_activas])
        else:
            X_base_np = np.empty((len(y_train), 0))

        for feature in features_disponibles:
            candidata_np = POOL_TRAIN[feature].reshape(-1, 1)
            X_test_loop = np.hstack([X_base_np, candidata_np])

            pipeline_evaluador = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RidgeCV(alphas=np.logspace(-2, 3, 10), cv=None))
            ])

            cv_scores = cross_val_score(pipeline_evaluador, X_test_loop, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=1)
            rmse_score = -cv_scores.mean()

            if rmse_score < mejor_rmse_ronda:
                mejor_rmse_ronda = rmse_score
                mejor_feature_ronda = feature

        delta_rmse = rmse_actual - mejor_rmse_ronda

        if (delta_rmse + TOLERANCIA) >= UMBRAL_MEJORA:
            features_activas.append(mejor_feature_ronda)
            features_disponibles.remove(mejor_feature_ronda)
            rmse_actual = mejor_rmse_ronda

            X_ganadora = np.hstack([X_base_np, POOL_TRAIN[mejor_feature_ronda].reshape(-1, 1)])
            pipeline_evaluador.fit(X_ganadora, y_train)
            r2_scores = cross_val_score(pipeline_evaluador, X_ganadora, y_train, cv=5, scoring='r2', n_jobs=1)
            mejor_r2_ronda = r2_scores.mean()

            tipo_var = 'num' if mejor_feature_ronda in TODAS_NUMERICAS else ('cat' if mejor_feature_ronda in TODAS_CATEGORICAS else 'inter')
            resultados_grafica.append({
                'n_vars': ronda, 'variable': mejor_feature_ronda, 'tipo': tipo_var, 'rmse_cv': mejor_rmse_ronda, 'r2_cv': mejor_r2_ronda
            })

            print(f"  #{ronda:<5} | {mejor_feature_ronda:<45} | {mejor_rmse_ronda:>8.4f} | {delta_rmse:>+8.4f} | {mejor_r2_ronda:>6.3f}")
            ronda += 1
        else:
            print(f"\n  [STOP] Proceso detenido. La variable '{mejor_feature_ronda}' mejoraba el RMSE en {delta_rmse:.4f}%, por debajo del umbral del {UMBRAL_MEJORA}%.")
            break

    # ────────────────────────────────────────────────────────────────
    #  5. GRÁFICA DE EVOLUCIÓN EN CROSS-VALIDATION
    # ────────────────────────────────────────────────────────────────
    if resultados_grafica:
        df_res = pd.DataFrame(resultados_grafica)
        
        fig, ax1 = plt.subplots(figsize=(13, 6))
        ax2 = ax1.twinx()

        colores_dict = {'num': '#3498DB', 'cat': '#E67E22', 'inter': '#2ECC71'}
        colores_barras = [colores_dict[tipo] for tipo in df_res['tipo']]

        ax1.bar(df_res['n_vars'], df_res['rmse_cv'], color=colores_barras, alpha=0.7, width=0.4, label='RMSE (Train CV)')
        ax2.plot(df_res['n_vars'], df_res['r2_cv'], color='#E74C3C', marker='o', linewidth=2, label='R² (Train CV)')

        ax1.set_xlabel('Progresión de Variables Seleccionadas', fontsize=11)
        ax1.set_ylabel('RMSE – Raíz del Error Cuadrático Medio (%)', color='#2980B9')
        ax2.set_ylabel('R² – Coeficiente de Determinación', color='#C0392B')
        ax1.tick_params(axis='y', labelcolor='#2980B9')
        ax2.tick_params(axis='y', labelcolor='#C0392B')
        ax1.set_xticks(df_res['n_vars'])
        ax1.set_xticklabels([f"{r['n_vars']}. {r['variable'][:18]}" for _, r in df_res.iterrows()], rotation=45, ha='right', fontsize=8)

        patch_num = mpatches.Patch(color='#3498DB', alpha=0.7, label='Numérica')
        patch_cat = mpatches.Patch(color='#E67E22', alpha=0.7, label='Categórica')
        patch_int = mpatches.Patch(color='#2ECC71', alpha=0.7, label='Interacción')
        linea_r2  = plt.Line2D([0], [0], color='#E74C3C', marker='o', linewidth=2, label='R²')
        ax1.legend(handles=[patch_num, patch_cat, patch_int, linea_r2], loc='upper right', fontsize=9)

        plt.title(f'Evolución en Selección Forward (Exigencia ΔRMSE >= {UMBRAL_MEJORA}%)', fontweight='bold')
        plt.tight_layout()

        rf_dir = os.path.join(ANALYSIS_DIR, "random forest")
        os.makedirs(rf_dir, exist_ok=True)
        ruta_grafica = os.path.join(rf_dir, "seleccion_variables_tie_estricta.png")
        plt.savefig(ruta_grafica, dpi=200)

        # ────────────────────────────────────────────────────────────────
        #  6. VALIDACIÓN OUT-OF-SAMPLE (TEST)
        # ────────────────────────────────────────────────────────────────
        print("\n" + "=" * 90)
        print("  Fase 4: VALIDACIÓN EXTERNA (OUT-OF-SAMPLE)")
        print("=" * 90)

        X_train_final = pd.DataFrame(np.column_stack([POOL_TRAIN[f] for f in features_activas]), columns=features_activas)
        X_test_final = pd.DataFrame(np.column_stack([POOL_TEST[f] for f in features_activas]), columns=features_activas)

        pipeline_final = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5))
        ])
        pipeline_final.fit(X_train_final, y_train)

        y_pred_test = pipeline_final.predict(X_test_final)
        
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)

        print(f"  RENDIMIENTO EN CONJUNTO DE TEST (Muestra Oculta del 20%):")
        print(f"  > RMSE Final : {rmse_test:.4f} %")
        print(f"  > R² Final   : {r2_test:.4f}")

        # ────────────────────────────────────────────────────────────────
        #  7. ANÁLISIS DE IMPORTANCIA DE VARIABLES (COEFICIENTES RIDGE)
        # ────────────────────────────────────────────────────────────────
        print("\n" + "=" * 90)
        print("  Fase 5: IMPORTANCIA DE VARIABLES (BASADO EN COEFICIENTES ESCALADOS)")
        print("=" * 90)

        # Extraemos el estimador Ridge entrenado de dentro del pipeline
        modelo_ridge = pipeline_final.named_steps['model']
        coeficientes = modelo_ridge.coef_

        # Creamos el dataframe de importancias y ordenamos por valor absoluto
        df_importancia = pd.DataFrame({
            'Variable': features_activas,
            'Coeficiente': coeficientes,
            'Impacto_Absoluto': np.abs(coeficientes)
        }).sort_values(by='Impacto_Absoluto', ascending=False)

        print(f"  {'Ranking':<8} | {'Variable':<45} | {'Coeficiente':>12} | {'Impacto Absoluto':>18}")
        print(f"  {'-'*8}+{'-'*47}+{'-'*14}+{'-'*20}")

        for i, (_, row) in enumerate(df_importancia.iterrows(), 1):
            # Formateamos con signo el coeficiente real, y el impacto como magnitud pura
            print(f"  #{i:<7} | {row['Variable']:<45} | {row['Coeficiente']:>12.4f} | {row['Impacto_Absoluto']:>18.4f}")

        # Exportación a disco
       # Exportación a disco (ACTUALIZADO para guardar preprocesadores)
        ruta_modelo = os.path.join(ANALYSIS_DIR, "mejor_pipeline_tie_seleccion.pkl")
        joblib.dump({
            'pipeline': pipeline_final,
            'imputer_num': num_imputer,          # <-- Añadido
            'encoder_cat': cat_pipeline,         # <-- Añadido
            'features_seleccionadas': features_activas,
            'metricas_test': {'rmse': rmse_test, 'r2': r2_test},
            'importancia_variables': df_importancia
        }, ruta_modelo)