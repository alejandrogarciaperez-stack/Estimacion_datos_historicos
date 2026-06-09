import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import joblib
def cargar_tasas_historico(years):
    ruta_12m = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/euribor_12_meses_datos_historicos.csv'
    ruta_3m = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/euribor_3_meses_datos_historicos.csv'
    ruta_tedr = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/tedr_credito_vivienda.csv'
    ruta_tedr_consumo = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/credito_consumo_tedr.csv'
    ruta_pyme = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/tipo_medio_ponderado_pyme.csv'
    ruta_prestamos_corporativos = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/tipo_prestamos_mas_un_millon.csv'
    ruta_tarjeta_credito = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/Tarjetas_credito_tipo.csv'
    
    tasas = {}
    for y in years:
        tasas[y] = {
            'euribor_12m': {}, 'euribor_3m': {}, 'tedr_vivienda': {},
            'tedr_consumo': {}, 'tipo_pyme': {}, 'tipo_corp': {}, 'tarjeta': {}
        }
    
    # Cargar Euribor
    for ruta, clave in [(ruta_12m, 'euribor_12m'), (ruta_3m, 'euribor_3m')]:
        if os.path.exists(ruta):
            df = pd.read_csv(ruta)
            df['DATE'] = pd.to_datetime(df['DATE'])
            for y in years:
                df_y = df[df['DATE'].dt.year == y]
                for _, row in df_y.iterrows():
                    tasas[y][clave][row['DATE'].month] = np.round(float(row.iloc[2]), 3)

    meses_map = {'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6, 
                 'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12}
    
    archivos_tedr = [
        (ruta_tedr, 'tedr_vivienda'),
        (ruta_tedr_consumo, 'tedr_consumo'),
        (ruta_pyme, 'tipo_pyme'),
        (ruta_prestamos_corporativos, 'tipo_corp'),
        (ruta_tarjeta_credito, 'tarjeta')
    ]
    
    for y in years:
        suffix = str(y)[-2:]
        for ruta, clave in archivos_tedr:
            if os.path.exists(ruta):
                with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if f'{suffix}"' in line:
                            partes = line.strip().split(',')
                            if len(partes) == 2:
                                fecha = partes[0].strip('"')
                                if fecha.endswith(suffix):
                                    mes_str = fecha[:3]
                                    if mes_str in meses_map:
                                        try:
                                            tasas[y][clave][meses_map[mes_str]] = float(partes[1])
                                        except:
                                            pass
    return tasas

def generar_datos_historicos(years, n_filas=1000):
    tasas = cargar_tasas_historico(years)
    arquetipos = ['A', 'B', 'C', 'D', 'E']
    probabilidades = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    datos = []
    
    for y in years:
        elecciones = np.random.choice(arquetipos, size=n_filas//len(years), p=probabilidades)
        for arq in elecciones:
            mes = int(np.random.randint(1, 13))
            t = tasas[y]
            
            # Obtener datos usando el mismo formato que 2025
            eur12 = t['euribor_12m'].get(mes, None)
            eur3 = t['euribor_3m'].get(mes, None)
            
            fila = {'year': y, 'contract_month': mes}
            
            if arq == 'A':
                fila.update({'client_segment_id': 'PARTICULARES', 'customer_type_id': 'PERSONA F',
                             'entity_id': 'BANCO COMERCIAL', 'family_id': 'HIPOTECAS',
                             'counterpart_id': np.random.choice(['TIPO_FIJO', 'TIPO_VARIABLE']),
                             'contract_collateral_type': 'INMUEBLES',
                             'local_curr_orig_risk_amount': np.round(np.random.uniform(80000, 400000), 2),
                             'euribor_12m_pct': eur12, 'euribor_3m_pct': None,
                             'tedr_pct': t['tedr_vivienda'].get(mes, None)})
            elif arq == 'B':
                fila.update({'client_segment_id': 'PARTICULARES', 'customer_type_id': 'PERSONA F',
                             'entity_id': 'BANCO COMERCIAL', 'family_id': 'TARJETA',
                             'counterpart_id': 'TARJETA_CLASICA', 'contract_collateral_type': 'NINGUNO',
                             'local_curr_orig_risk_amount': np.round(np.random.uniform(1000, 6000), 2),
                             'euribor_12m_pct': eur12, 'euribor_3m_pct': None,
                             'tedr_pct': t['tarjeta'].get(mes, None)})
            elif arq == 'C':
                fila.update({'client_segment_id': 'PARTICULARES', 'customer_type_id': 'PERSONA F',
                             'entity_id': 'FINANCIERA DE CONSUMO', 'family_id': 'PRESTAMOS',
                             'counterpart_id': 'PRESTAMO_PERSONAL', 'contract_collateral_type': 'PERSONAL',
                             'local_curr_orig_risk_amount': np.round(np.random.uniform(5000, 35000), 2),
                             'euribor_12m_pct': eur12, 'euribor_3m_pct': None,
                             'tedr_pct': t['tedr_consumo'].get(mes, None)})
            elif arq == 'D':
                fila.update({'client_segment_id': 'PYMES', 'customer_type_id': 'JU',
                             'entity_id': 'BANCO COMERCIAL', 'family_id': 'COMERC',
                             'counterpart_id': 'LINEA DE CREDITO', 'contract_collateral_type': 'PERSONAL',
                             'local_curr_orig_risk_amount': np.round(np.random.uniform(20000, 150000), 2),
                             'euribor_12m_pct': None, 'euribor_3m_pct': eur3,
                             'tedr_pct': t['tipo_pyme'].get(mes, None)})
            elif arq == 'E':
                fila.update({'client_segment_id': 'CORPORATIVO', 'customer_type_id': 'JU',
                             'entity_id': 'BANCA E INVERSION/MAYORISTA', 'family_id': 'SINDICA',
                             'counterpart_id': 'PRESTAMO SINDICADO', 'contract_collateral_type': 'MOBILIARIA(ACCIONES/FONDO)',
                             'local_curr_orig_risk_amount': np.round(np.random.uniform(1000000, 15000000), 2),
                             'euribor_12m_pct': None, 'euribor_3m_pct': eur3,
                             'tedr_pct': t['tipo_corp'].get(mes, None)})
            
            # Calcular spread real
            tie = fila.get('tedr_pct')
            e12 = fila.get('euribor_12m_pct')
            e3 = fila.get('euribor_3m_pct')
            
            if tie is not None and not pd.isna(tie):
                tie_dec = tie / 100.0
                if e12 is not None and not pd.isna(e12):
                    ref_dec = e12 / 100.0; m = 12
                elif e3 is not None and not pd.isna(e3):
                    ref_dec = e3 / 100.0; m = 3
                else:
                    ref_dec = 0.0; m = 12
                tin_dec = m * ((1 + tie_dec)**(1/m) - 1)
                spread_dec = tin_dec - ref_dec
                fila['spread_pct_real'] = np.round(spread_dec * 100, 3)
                fila['ref_pct'] = ref_dec * 100
                fila['m_freq'] = m
            else:
                fila['spread_pct_real'] = None
                fila['ref_pct'] = None
                fila['m_freq'] = 12
                
            datos.append(fila)
            
    return pd.DataFrame(datos)

# ... (Todo tu código anterior de las funciones cargar_tasas_historico y generar_datos_historicos se queda igual)

if __name__ == '__main__':
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    print(f"Generando datos para {years}...")
    df_hist = generar_datos_historicos(years, 2000)
    df_hist = df_hist.dropna(subset=['spread_pct_real'])
    print(f"Generados {len(df_hist)} registros válidos.")
    
    # Preparar datos para el modelo (igual que en random_forest_spread.py)
    df_hist["euribor_ref_pct"] = df_hist["euribor_12m_pct"].fillna(df_hist["euribor_3m_pct"]).fillna(0.0)
    df_hist["tiene_euribor_12m"] = df_hist["euribor_12m_pct"].notna().astype(int)
    df_hist["tiene_euribor_3m"]  = df_hist["euribor_3m_pct"].notna().astype(int)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ¡AQUÍ COLOCAS EL NUEVO BLOQUE QUE REEMPLAZA AL ENCODER Y ENTRENAMIENTO VIEJO!
    # ─────────────────────────────────────────────────────────────────────────
    print("\nCargando el pipeline optimizado (Random Forest + Preprocesamiento)...")
    ruta_modelo = r"C:\Proyectos\Empresa\Caso_Pedro\Analisis\mejor_pipeline_rf.pkl"
    
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
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # ... (El resto del código de métricas, prints y gráficos se queda exactamente igual)
    # Comparar TEDR Real vs Predicho por año
    from sklearn.metrics import r2_score, mean_absolute_error
    
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
    ruta_salida = r"C:\Proyectos\Empresa\Caso_Pedro\Analisis\resultados_historicos_2010_2017.csv"
    df_hist.to_csv(ruta_salida, index=False)
    print(f"\nResultados exportados a: {ruta_salida}")
    
    # Mostrar muestra
    print("\nMuestra de predicciones:")
    print(df_hist[['year', 'family_id', 'tedr_pct', 'tedr_pred_pct', 'spread_pct_real', 'spread_pred']].head(10).to_string())
