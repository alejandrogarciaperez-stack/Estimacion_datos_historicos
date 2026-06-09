"""
Random Forest Regressor - Predicción de Spread (spread_pct)
============================================================
Dataset: contratos_sinteticos.csv
Variable objetivo: spread_pct
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ─────────────────────────────────────────────
print("=" * 60)
print("  RANDOM FOREST REGRESSOR — PREDICCIÓN DE SPREAD (%)")
print("=" * 60)

RUTA_CSV = r"C:\Proyectos\Empresa\Caso_Pedro\Base_de_datos\contratos_sinteticos.xlsx"
df = pd.read_excel(RUTA_CSV)

print(f"\n▶ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
print("\n— Primeras filas —")
print(df.head(5).to_string())

print(f"\n— Información del dataset —")
print(df.info())

print(f"\n— Valores nulos por columna —")
print(df.isnull().sum())

print(f"\n— Estadísticas descriptivas —")
print(df.describe())

# ─────────────────────────────────────────────
# 2. PREPROCESAMIENTO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREPROCESAMIENTO")
print("=" * 60)

# 2.1 Filtrar filas con spread_pct nulo (no podemos entrenar sin target)

n_antes = len(df)
df_clean = df.dropna(subset=["spread_pct"]).copy() # dropna elimina todas las filas que contienen al menos un valor nulo
n_despues = len(df_clean)
print(f"\n▶ Filas con spread_pct no nulo: {n_despues} (eliminadas {n_antes - n_despues})")


# 2.2 Gestión del Euribor: ambas columnas son mutuamente excluyentes según el arquetipo.
#     Creamos una sola columna 'euribor_ref_pct' con el Euribor aplicable.
#     Cuando ambas son NaN (ej. tarjetas, consumo fijo), ponemos NaN.
df_clean["euribor_ref_pct"] = df_clean["euribor_12m_pct"].fillna(
    df_clean["euribor_3m_pct"]
).fillna(np.nan)

# Indicador del tipo de índice de referencia
df_clean["tiene_euribor_12m"] = df_clean["euribor_12m_pct"].notna().astype(int)
df_clean["tiene_euribor_3m"]  = df_clean["euribor_3m_pct"].notna().astype(int)


# 2.4 Definir features y target
FEATURES = [
    "contract_month",
    "client_segment_id",
    "customer_type_id",
    "entity_id",
    "family_id",
    "counterpart_id",
    "contract_collateral_type",
    "local_curr_orig_risk_amount",
    "euribor_ref_pct", 
    "tiene_euribor_12m", # Sin variables que indican si tienen o no este eruibor. 
    "tiene_euribor_3m",
]

TARGET = "spread_pct"

X = df_clean[FEATURES]
y = df_clean[TARGET]

print(f"\n▶ Features utilizadas ({len(FEATURES)}): {FEATURES}")
print(f"▶ Variable objetivo: '{TARGET}'")
print(f"▶ Distribución del target:")
print(f"  Media: {y.mean():.4f}% | Std: {y.std():.4f}% | Min: {y.min():.4f}% | Max: {y.max():.4f}%")

# ─────────────────────────────────────────────
# 3. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"\n▶ Split 70/30 → Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

# ─────────────────────────────────────────────
# 3.5 PREPROCESAMIENTO (Buenas Prácticas para evitar Data Leakage)
# ─────────────────────────────────────────────
# Identificamos automáticamente variables numéricas y categóricas
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO DEL MODELO 
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ENTRENAMIENTO — RANDOM FOREST REGRESSOR")
print("=" * 60)

# El Pipeline asegura que la codificación y la imputación ocurran dentro del CV
# Imputación de variables numéricas con la mediana
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Imputación de variables categóricas con la moda y ONEHOTENCODER para evitar el 'missing' 
# One Hot Enconder es un método de codificación que convierte las variables categóricas en variables binarias (0,1)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Notar el prefijo 'regressor__' requerido por el Pipeline para llegar al RandomForest
param_grid = {
    'regressor__bootstrap': [True],
    'regressor__n_estimators': [50,100, 200], # n_estimators es el número de árboles que se van a entrenar
    'regressor__max_depth': [2,4,6,8,10,12,14,20,30], # max_depth es la profundidad máxima de los árboles
    'regressor__max_features': [2,4,6], # max_features es el número de características que se van a utilizar para entrenar los árboles
    'regressor__min_samples_split': [10, 20, 50, 100], # min_samples_split es el número mínimo de observaciones que debe tener un nodo internoi para poder dividirse en dos.
    'regressor__min_samples_leaf': [5, 10, 25, 50,]# min_samples_leaf es el número mínimo de observaciones que debe contener un nodo hoja. Es el nodo final, que repsenta la decisión, predicción, o resultado final. 
}

# Grid search para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error', # scoring es la métrica que se va a utilizar para evaluar el modelo
    cv=5, # cv es el número de k-fold que se van a utilizar para evaluar el modelo
    n_jobs=-1,
    verbose=2
)

# Grid search para encontrar los mejores hiperparámetros
grid_search.fit(X_train, y_train)

# ¡CORRECCIÓN CRÍTICA! Extraemos el mejor modelo
best_model = grid_search.best_estimator_
best_rf = best_model.named_steps['regressor'] # El Random Forest puro

print("\n" + "=" * 60)
print("  Mejores Hiperparámetros encontrados:")
# Limpiamos el prefijo 'regressor__' para que se lea mejor en consola
clean_params = {k.replace('regressor__', ''): v for k, v in grid_search.best_params_.items()}
print(clean_params)
print("\n  Mejor MSE (Negativo) con Validación Cruzada:")
print(f"  {grid_search.best_score_:.6f}")

# ─────────────────────────────────────────────
# 5. EVALUACIÓN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUACIÓN DEL MODELO")
print("=" * 60)

# CORRECCIÓN: Usar best_model, NO rf
y_pred_train = best_model.predict(X_train)
y_pred_test  = best_model.predict(X_test)

def metricas(y_true, y_pred, nombre):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{nombre}]")
    print(f"    R²   : {r2:.6f}")
    print(f"    RMSE : {rmse:.6f} %")
    print(f"    MAE  : {mae:.6f} %")
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

m_train = metricas(y_train, y_pred_train, "TRAIN")
m_test  = metricas(y_test,  y_pred_test,  "TEST ")

# CORRECCIÓN: Validar el best_model, no el modelo vacío
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
print(f"\n  [Cross-Validation 5-fold sobre Train]")
print(f"    R² (media ± std): {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
print(f"    Scores por fold: {np.round(cv_scores, 6)}")

# ─────────────────────────────────────────────
# 6. IMPORTANCIA DE VARIABLES
# ─────────────────────────────────────────────
# Extraer nombres de las variables codificadas del preprocesador
feature_names_out = best_model.named_steps['preprocessor'].get_feature_names_out()

importances = pd.Series(best_rf.feature_importances_, index=feature_names_out).sort_values(ascending=False)
# Tomamos el top 15 para no saturar gráficos si hay mucho One-Hot Encoding
top_importances = importances.head(15) 

print("\n  [Top 15 Importancia de Variables — MDI]")
print(top_importances.round(6).to_string())

# Permutation importance (sobre el best_model completo para que haga el preprocesamiento)
perm_imp = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    "feature":        FEATURES,
    "importance_mean": perm_imp.importances_mean,
    "importance_std":  perm_imp.importances_std,
}).sort_values("importance_mean", ascending=False)
print("\n  [Permutation Importance sobre Test]")
print(perm_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. VISUALIZACIONES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  GENERANDO GRÁFICOS...")
print("=" * 60)

COLOR_PRIMARY = "#4F46E5"
COLOR_SECONDARY = "#06B6D4"
COLOR_ACCENT = "#F59E0B"
COLOR_BG = "#0F172A"
COLOR_SURFACE = "#1E293B"
COLOR_TEXT = "#E2E8F0"
COLOR_GRID = "#334155"

plt.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor":   COLOR_SURFACE,
    "axes.edgecolor":   COLOR_GRID,
    "axes.labelcolor":  COLOR_TEXT,
    "axes.titlecolor":  COLOR_TEXT,
    "xtick.color":      COLOR_TEXT,
    "ytick.color":      COLOR_TEXT,
    "text.color":       COLOR_TEXT,
    "grid.color":       COLOR_GRID,
    "grid.alpha":       0.4,
    "font.family":      "sans-serif",
    "font.size":        10,
})

fig = plt.figure(figsize=(20, 22), facecolor=COLOR_BG)
fig.suptitle(
    "Random Forest Regressor — Predicción de Spread (%)\nContratos Sintéticos",
    fontsize=18, fontweight="bold", color=COLOR_TEXT, y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Gráfico 1: Distribución del target
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(y, bins=40, color=COLOR_PRIMARY, edgecolor="#312E81", alpha=0.9)
ax1.axvline(y.mean(), color=COLOR_ACCENT, linestyle="--", linewidth=1.8, label=f"Media: {y.mean():.3f}%")
ax1.set_title("Distribución del Spread (%)", fontweight="bold", pad=10)
ax1.set_xlabel("spread_pct (%)")
ax1.set_ylabel("Frecuencia")
ax1.legend(fontsize=9)
ax1.grid(True, axis="y")

# ── Gráfico 2: Real vs Predicho
ax2 = fig.add_subplot(gs[0, 1:])
ax2.scatter(y_test, y_pred_test, alpha=0.55, s=22, color=COLOR_SECONDARY, edgecolors="none", label="Predicciones")
lims = [min(y_test.min(), y_pred_test.min()) - 0.1,
        max(y_test.max(), y_pred_test.max()) + 0.1]
ax2.plot(lims, lims, color=COLOR_ACCENT, linewidth=2, linestyle="--", label="Predicción perfecta")
ax2.set_title(f"Real vs Predicho (Test) — R²={m_test['R2']:.4f}", fontweight="bold", pad=10)
ax2.set_xlabel("spread_pct Real (%)")
ax2.set_ylabel("spread_pct Predicho (%)")
ax2.legend(fontsize=9)
ax2.grid(True)

# ── Gráfico 3: Importancia de Variables (MDI) - CORRECCIÓN: top_importances
ax3 = fig.add_subplot(gs[1, 0:2])
colors_imp = [COLOR_PRIMARY] * len(top_importances)
colors_imp[0] = COLOR_ACCENT
bars = ax3.barh(top_importances.index[::-1], top_importances.values[::-1], color=colors_imp[::-1], edgecolor="none", height=0.65)
ax3.set_title("Top 15 Importancia de Variables (MDI)", fontweight="bold", pad=10)
ax3.set_xlabel("Importancia relativa")
for bar, val in zip(bars, top_importances.values[::-1]):
    ax3.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=8.5, color=COLOR_TEXT)
ax3.grid(True, axis="x")
ax3.set_xlim(0, top_importances.max() * 1.25)

# ── Gráfico 4: Permutation Importance
ax4 = fig.add_subplot(gs[1, 2])
perm_sorted = perm_df.sort_values("importance_mean")
ax4.barh(perm_sorted["feature"], perm_sorted["importance_mean"],
         xerr=perm_sorted["importance_std"], color=COLOR_SECONDARY,
         edgecolor="none", capsize=4, height=0.6, ecolor=COLOR_ACCENT)
ax4.set_title("Permutation Importance\n(Test Set)", fontweight="bold", pad=10)
ax4.set_xlabel("Reducción en R²")
ax4.grid(True, axis="x")

# ── Gráfico 5: Residuos vs Predicho
ax5 = fig.add_subplot(gs[2, 0])
residuos = y_test.values - y_pred_test
ax5.scatter(y_pred_test, residuos, alpha=0.5, s=18, color=COLOR_PRIMARY, edgecolors="none")
ax5.axhline(0, color=COLOR_ACCENT, linewidth=1.5, linestyle="--")
ax5.set_title("Residuos vs Predicho", fontweight="bold", pad=10)
ax5.set_xlabel("spread_pct Predicho (%)")
ax5.set_ylabel("Residuo (%)")
ax5.grid(True)

# ── Gráfico 6: Distribución de Residuos
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(residuos, bins=35, color=COLOR_PRIMARY, edgecolor="#312E81", alpha=0.9)
ax6.axvline(0, color=COLOR_ACCENT, linewidth=1.5, linestyle="--")
ax6.set_title(f"Distribución de Residuos\nMedia={residuos.mean():.4f} | Std={residuos.std():.4f}", fontweight="bold", pad=10)
ax6.set_xlabel("Residuo (%)")
ax6.set_ylabel("Frecuencia")
ax6.grid(True, axis="y")

# ── Gráfico 7: Métricas resumen
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis("off")
metricas_tabla = [
    ["Métrica",   "Train",                 "Test"],
    ["R²",        f"{m_train['R2']:.6f}",  f"{m_test['R2']:.6f}"],
    ["RMSE (%)",  f"{m_train['RMSE']:.6f}", f"{m_test['RMSE']:.6f}"],
    ["MAE (%)",   f"{m_train['MAE']:.6f}",  f"{m_test['MAE']:.6f}"],
    ["CV R² μ",   f"{cv_scores.mean():.6f}", "—"],
    ["CV R² σ",   f"{cv_scores.std():.6f}",  "—"],
]
tabla = ax7.table(
    cellText=metricas_tabla[1:],
    colLabels=metricas_tabla[0],
    loc="center",
    cellLoc="center",
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.2, 1.8)
for (row, col), cell in tabla.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLOR_PRIMARY)
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#0F172A")
        cell.set_text_props(color=COLOR_TEXT)
    else:
        cell.set_facecolor(COLOR_SURFACE)
        cell.set_text_props(color=COLOR_TEXT)
    cell.set_edgecolor(COLOR_GRID)
ax7.set_title("Resumen de Métricas", fontweight="bold", pad=15)

plt.show()

# ─────────────────────────────────────────────
# 8. RESUMEN FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESUMEN FINAL DEL MODELO")
print("=" * 60)
# CORRECCIÓN CRÍTICA: Valores extraídos dinámicamente del mejor modelo
def get_r2_adjusted(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

r2_adj_train = get_r2_adjusted(m_train['R2'], len(y_train), X_train.shape[1])
r2_adj_test = get_r2_adjusted(m_test['R2'], len(y_test), X_test.shape[1])

print(f"""
  Algoritmo       : Random Forest Regressor
  Árboles         : {best_rf.n_estimators}
  Max Depth       : {best_rf.max_depth}
  Max Features    : {best_rf.max_features}
  Min Samples Leaf: {best_rf.min_samples_leaf}

  TRAIN
    R²            : {m_train['R2']:.6f}
    RMSE          : {m_train['RMSE']:.6f} %
    MAE           : {m_train['MAE']:.6f} %
    R² ajustada   : {r2_adj_train:.6f}

  TEST
    R²            : {m_test['R2']:.6f}
    RMSE          : {m_test['RMSE']:.6f} %
    MAE           : {m_test['MAE']:.6f} %
    R² ajustada   : {r2_adj_test:.6f}

  Cross-Val 5-fold (Train)
    R² media      : {cv_scores.mean():.6f}
    R² std        : {cv_scores.std():.6f}

  Feature MDI más importante : {top_importances.index[0]} ({top_importances.iloc[0]:.4f})
  Feature Perm más importante: {perm_df.iloc[0]['feature']} ({perm_df.iloc[0]['importance_mean']:.4f}""")


import joblib
import os

# Asegúrate de que la carpeta existe
ruta_directorio = r"C:\Proyectos\Empresa\Caso_Pedro\Analisis"
os.makedirs(ruta_directorio, exist_ok=True)

# Guardar el pipeline completo
ruta_modelo = os.path.join(ruta_directorio, "mejor_pipeline_rf.pkl")
joblib.dump(best_model, ruta_modelo)
print(f"\n▶ Modelo optimizado exportado exitosamente en: {ruta_modelo}")

