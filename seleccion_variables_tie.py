import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from config import DB_DIR, ANALYSIS_DIR

# ────────────────────────────────────────────────────────────────
#  1. CARGA Y PREPARACIÓN
# ────────────────────────────────────────────────────────────────
ruta_datos = os.path.join(DB_DIR, 'contratos_regresion_2018_2025.csv')
df = pd.read_csv(ruta_datos)
df = df.dropna(subset=['tedr_pct'])
df["euribor_ref_pct"] = df["euribor_12m_pct"].fillna(df["euribor_3m_pct"]).fillna(0.0)

# TRANSFORMACIONES por no-linealidad detectada en el diagnostico
df["log_importe"] = np.log1p(df["local_curr_orig_risk_amount"])
df["euribor_sq"] = df["euribor_ref_pct"] ** 2

TARGET = "tedr_pct"

TODAS_NUMERICAS = [
    "log_importe", "euribor_sq",
    "inflation_rate", "gdp_growth", "unemployment_rate",
    "contract_month"
]
TODAS_CATEGORICAS = [
    "client_segment_id", "customer_type_id", "entity_id",
    "family_id", "counterpart_id", "contract_collateral_type"
]

# ────────────────────────────────────────────────────────────────
#  2. CÁLCULO DE VIF — Eliminar variables con alta multicolinealidad
# ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  PASO 1: ANÁLISIS DE MULTICOLINEALIDAD (VIF)")
print("=" * 65)
X_num = df[TODAS_NUMERICAS].dropna()
vif_dict = {}
for col in TODAS_NUMERICAS:
    X_other = X_num.drop(columns=[col])
    r2 = LinearRegression().fit(X_other, X_num[col]).score(X_other, X_num[col])
    vif = 1 / (1 - r2) if r2 < 1.0 else float('inf')
    vif_dict[col] = vif
    print(f"  VIF({col:<35}): {vif:6.2f}  {'⚠  ALTA' if vif>10 else 'OK'}")

NUMERICAS_VALIDAS = [col for col, v in vif_dict.items() if v <= 10]
print(f"\n  Variables numéricas válidas tras VIF: {NUMERICAS_VALIDAS}")

# ────────────────────────────────────────────────────────────────
#  3. ORDENAR CANDIDATAS POR CORRELACIÓN CON EL TIE
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PASO 2: ORDEN DE ENTRADA POR CORRELACIÓN CON EL TIE")
print("=" * 65)

corr = df[NUMERICAS_VALIDAS + [TARGET]].corr()[TARGET].drop(TARGET)
orden_numericas = corr.abs().sort_values(ascending=False).index.tolist()
for i, col in enumerate(orden_numericas):
    print(f"    {i+1}. {col:<35} r = {corr[col]:+.4f}")

ORDEN_CANDIDATAS = [(col, 'num') for col in orden_numericas] + \
                   [(cat, 'cat') for cat in TODAS_CATEGORICAS]

# ────────────────────────────────────────────────────────────────
#  4. ESTUDIO PROGRESIVO CON UMBRAL DE RECHAZO (FORWARD STEPWISE)
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("  PASO 3: SELECCION PROGRESIVA — ADUANA ESTRICTA DE RUIDO")
print("=" * 85)

UMBRAL_MEJORA = 0.1  # Si la mejora es < 0.1 o aumenta el error, se descarta

print(f"  Criterio de aceptación: ΔMAE >= {UMBRAL_MEJORA}%\n")
print(f"  {'N vars':<8} | {'Candidata evaluada':<30} | {'MAE (%)':>8} | {'Δ MAE':>8} | {'Estado':<15}")
print(f"  {'-'*8}+{'-'*32}+{'-'*10}+{'-'*10}+{'-'*15}")

y = df[TARGET]
resultados = []
num_activas = []
cat_activas = []
mae_prev = None

for nombre, tipo in ORDEN_CANDIDATAS:
    # Creamos un conjunto de prueba añadiendo la candidata a las que ya fueron aceptadas
    test_num = num_activas + ([nombre] if tipo == 'num' else [])
    test_cat = cat_activas + ([nombre] if tipo == 'cat' else [])

    # Construir transformador para el conjunto de prueba
    transformers = []
    if test_num:
        transformers.append(
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('sc', StandardScaler())
            ]), test_num)
        )
    if test_cat:
        transformers.append(
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), test_cat)
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RidgeCV(alphas=np.logspace(-2, 3, 15), cv=5))
    ])

    X_actual = df[test_num + test_cat].copy()
    
    # Evaluar rendimiento real con Cross-Validation 5-Fold
    cv_scores = cross_val_score(pipeline, X_actual, y, cv=5, scoring='neg_mean_absolute_error')
    mae_cv = -cv_scores.mean()
    r2_scores = cross_val_score(pipeline, X_actual, y, cv=5, scoring='r2')
    r2_cv = r2_scores.mean()

    # Calcular la diferencia con el modelo anterior
    delta = mae_prev - mae_cv if mae_prev is not None else 999

    # ─── LÓGICA DE RECHAZO / ACEPTACIÓN ───
    if mae_prev is None or delta >= UMBRAL_MEJORA:
        estado = "✅ ACEPTADA"
        # Oficializamos la variable en nuestras listas activas
        num_activas = test_num
        cat_activas = test_cat
        mae_prev = mae_cv
        n_vars_actual = len(num_activas) + len(cat_activas)
        
        # Solo guardamos resultados para la gráfica de las que han sido aceptadas
        resultados.append({
            'n_vars': n_vars_actual,
            'variable': nombre,
            'tipo': tipo,
            'mae_cv': mae_cv,
            'r2_cv': r2_cv,
            'delta_mae': delta if delta != 999 else 0.0
        })
        print(f"  {n_vars_actual:<8} | {nombre:<30} | {mae_cv:>8.4f} | {f'{delta:+.4f}' if delta != 999 else 'Inicio':>8} | {estado}")
    else:
        # Se detecta ruido: el MAE bajó muy poco o directamente subió (delta negativo)
        estado = "❌ RECHAZADA" 
        print(f"  {len(num_activas)+len(cat_activas):<8} | {nombre:<30} | {mae_cv:>8.4f} | {f'{delta:+.4f}':>8} | {estado}")