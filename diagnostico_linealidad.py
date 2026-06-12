import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
from config import DB_DIR, ANALYSIS_DIR

# ─── CARGA ────────────────────────────────────────────────────────
ruta = os.path.join(DB_DIR, 'contratos_regresion_2018_2025.csv')
df = pd.read_csv(ruta)
df = df.dropna(subset=['tedr_pct'])
df["euribor_ref_pct"] = df["euribor_12m_pct"].fillna(df["euribor_3m_pct"]).fillna(0.0)
df["tiene_euribor_12m"] = df["euribor_12m_pct"].notna().astype(int)
df["tiene_euribor_3m"]  = df["euribor_3m_pct"].notna().astype(int)

TARGET = "tedr_pct"
NUMERICAS = [
    "inflation_rate", "gdp_growth", "unemployment_rate",
    "euribor_ref_pct", "local_curr_orig_risk_amount"
]

y = df[TARGET]

# ─── TABLA PEARSON vs SPEARMAN vs INFORMACIÓN MUTUA ───────────────
print("=" * 80)
print("  ANÁLISIS DE CORRELACIÓN Y DEPENDENCIA: Pearson vs Spearman vs MI")
print("=" * 80)
print(f"  {'Variable':<30} | {'Pearson (r)':>12} | {'Spearman (ρ)':>12} | {'Mutual Info':>12}")
print(f"  {'-'*30}+{'-'*14}+{'-'*14}+{'-'*14}")

diagnosticos = {}
for col in NUMERICAS:
    # Alineación de índices sin nulos
    x = df[col].dropna()
    idx = x.index.intersection(y.index)
    
    # Métricas lineales y monótonas
    pearson_r, _ = stats.pearsonr(df.loc[idx, col], y[idx])
    spearman_r, _ = stats.spearmanr(df.loc[idx, col], y[idx])
    
    # Información Mutua (No lineal / Entropía)
    # Nota: MI requiere que X sea un array 2D, por eso usamos df.loc[idx, [col]]
    mi_score = mutual_info_regression(df.loc[idx, [col]], y[idx], random_state=42)[0]
    
    diagnosticos[col] = {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'mi': mi_score
    }
    
    # Imprimimos la fila de la tabla
    print(f"  {col:<30} | {pearson_r:>+12.4f} | {spearman_r:>+12.4f} | {mi_score:>12.4f}")

# ─── SCATTER PLOTS + RESIDUOS ─────────────────────────────────────
n_vars = len(NUMERICAS)
fig = plt.figure(figsize=(15, 5 * n_vars))
gs = gridspec.GridSpec(n_vars, 2, figure=fig, hspace=0.45, wspace=0.25)

for i, col in enumerate(NUMERICAS):
    idx = df[col].dropna().index.intersection(y.index)
    x_vals = df.loc[idx, col].values.reshape(-1, 1)
    y_vals = y[idx].values
    
    # Regresión lineal simple para trazar la línea y obtener residuos
    reg = LinearRegression().fit(x_vals, y_vals)
    y_pred = reg.predict(x_vals)
    residuos = y_vals - y_pred
    r2 = r2_score(y_vals, y_pred)
    
    d = diagnosticos[col]

    # --- Panel 1: Scatter X vs TIE (Ajuste Lineal) ---
    ax1 = fig.add_subplot(gs[i, 0])
    ax1.scatter(df.loc[idx, col], y[idx], alpha=0.3, s=15, color='#3498DB', edgecolors='none')
    x_line = np.linspace(df.loc[idx, col].min(), df.loc[idx, col].max(), 100).reshape(-1, 1)
    ax1.plot(x_line, reg.predict(x_line), color='#E74C3C', linewidth=2.5, linestyle='-')
    
    ax1.set_xlabel(f'{col}', fontsize=10, fontweight='medium')
    ax1.set_ylabel('TIE (tedr_pct)', fontsize=10, fontweight='medium')
    # Añadimos MI al título del gráfico
    ax1.set_title(f'Ajuste Lineal\nPearson: {d["pearson"]:+.3f} | Spearman: {d["spearman"]:+.3f} | MI: {d["mi"]:.3f}', 
                  fontsize=11, fontweight='bold', color='#2C3E50')
    ax1.tick_params(labelsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- Panel 2: Residuos vs Predichos ---
    ax2 = fig.add_subplot(gs[i, 1])
    ax2.scatter(y_pred, residuos, alpha=0.3, s=15, color='#9B59B6', edgecolors='none')
    ax2.axhline(0, color='#E74C3C', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Predicción del TIE', fontsize=10, fontweight='medium')
    ax2.set_ylabel('Residuo (Error)', fontsize=10, fontweight='medium')
    ax2.set_title(f'Distribución del Residuo\n$R^2$ = {r2:.3f}', 
                  fontsize=11, fontweight='bold', color='#2C3E50')
    ax2.tick_params(labelsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

fig.suptitle('Análisis de Linealidad y Dependencia (Evaluación Manual)',
             fontsize=16, fontweight='bold', y=0.92, color='#2C3E50')

rf_dir = os.path.join(ANALYSIS_DIR, "random forest")
os.makedirs(rf_dir, exist_ok=True)
ruta_grafica = os.path.join(rf_dir, "diagnostico_linealidad_manual.png")
plt.savefig(ruta_grafica, dpi=200, bbox_inches='tight', facecolor='white')

print(f"\nGráfica de diagnóstico manual guardada en: {ruta_grafica}")


## Con transformación de las varoiables 

df["euribor_ref_pct"] = df["euribor_12m_pct"].fillna(df["euribor_3m_pct"]).fillna(0.0)

# ─── 1. APLICAMOS LAS TRANSFORMACIONES ────────────────────────────
df["euribor_sq"] = df["euribor_ref_pct"] ** 2
df["log_importe"] = np.log1p(df["local_curr_orig_risk_amount"])

NUMERICAS_TRANS = [
    "euribor_ref_pct",              # Original
    "euribor_sq",                   # Transformada (Cuadrática)
    "local_curr_orig_risk_amount",  # Original
    "log_importe"                   # Transformada (Logaritmo)
]

diagnosticos = {}
for col in NUMERICAS:
    x = df[col].dropna()
    idx = x.index.intersection(y.index)
    
    pearson_r, _ = stats.pearsonr(df.loc[idx, col], y[idx])
    spearman_r, _ = stats.spearmanr(df.loc[idx, col], y[idx])
    mi_score = mutual_info_regression(df.loc[idx, [col]], y[idx], random_state=42)[0]
    
    diagnosticos[col] = {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'mi': mi_score
    }
    
    print(f"  {col:<30} | {pearson_r:>+12.4f} | {spearman_r:>+12.4f} | {mi_score:>12.4f}")
    
    # Ponemos una pequeña separación visual entre los pares
    if col in ["euribor_sq", "log_importe"]:
        print("  " + "-"*76)

# ─── SCATTER PLOTS + RESIDUOS ─────────────────────────────────────
n_vars = len(NUMERICAS)
fig = plt.figure(figsize=(15, 5 * n_vars))
gs = gridspec.GridSpec(n_vars, 2, figure=fig, hspace=0.45, wspace=0.25)

for i, col in enumerate(NUMERICAS):
    idx = df[col].dropna().index.intersection(y.index)
    x_vals = df.loc[idx, col].values.reshape(-1, 1)
    y_vals = y[idx].values
    
    reg = LinearRegression().fit(x_vals, y_vals)
    y_pred = reg.predict(x_vals)
    residuos = y_vals - y_pred
    r2 = r2_score(y_vals, y_pred)
    
    d = diagnosticos[col]

    # --- Panel 1: Scatter X vs TIE (Ajuste Lineal) ---
    ax1 = fig.add_subplot(gs[i, 0])
    # Damos color distinto a las originales (Azul) y a las transformadas (Verde)
    color_puntos = '#2ECC71' if 'sq' in col or 'log' in col else '#3498DB'
    
    ax1.scatter(df.loc[idx, col], y[idx], alpha=0.3, s=15, color=color_puntos, edgecolors='none')
    x_line = np.linspace(df.loc[idx, col].min(), df.loc[idx, col].max(), 100).reshape(-1, 1)
    ax1.plot(x_line, reg.predict(x_line), color='#E74C3C', linewidth=2.5, linestyle='-')
    
    ax1.set_xlabel(f'{col}', fontsize=10, fontweight='medium')
    ax1.set_ylabel('TIE (tedr_pct)', fontsize=10, fontweight='medium')
    ax1.set_title(f'Ajuste Lineal\nPearson: {d["pearson"]:+.3f} | Spearman: {d["spearman"]:+.3f} | MI: {d["mi"]:.3f}', 
                  fontsize=11, fontweight='bold', color='#2C3E50')
    ax1.tick_params(labelsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- Panel 2: Residuos vs Predichos ---
    ax2 = fig.add_subplot(gs[i, 1])
    ax2.scatter(y_pred, residuos, alpha=0.3, s=15, color='#9B59B6', edgecolors='none')
    ax2.axhline(0, color='#E74C3C', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Predicción del TIE', fontsize=10, fontweight='medium')
    ax2.set_ylabel('Residuo (Error)', fontsize=10, fontweight='medium')
    ax2.set_title(f'Distribución del Residuo\n$R^2$ = {r2:.3f}', 
                  fontsize=11, fontweight='bold', color='#2C3E50')
    ax2.tick_params(labelsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

fig.suptitle('Evaluación de Eficacia de Transformaciones (Antes vs Después)',
             fontsize=16, fontweight='bold', y=0.92, color='#2C3E50')

rf_dir = os.path.join(ANALYSIS_DIR, "random forest")
os.makedirs(rf_dir, exist_ok=True)
ruta_grafica = os.path.join(rf_dir, "diagnostico_transformaciones.png")
plt.savefig(ruta_grafica, dpi=200, bbox_inches='tight', facecolor='white')

print(f"\nGráfica de diagnóstico de transformaciones guardada en: {ruta_grafica}")