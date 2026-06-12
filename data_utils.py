import pandas as pd
import numpy as np
import os
import csv
from config import PATHS_CSV

def cargar_tasas(years):
    tasas = {}
    for y in years:
        tasas[y] = {
            'euribor_12m': {}, 'euribor_3m': {}, 'tedr_vivienda': {},
            'tedr_consumo': {}, 'tipo_pyme': {}, 'tipo_corp': {}, 'tarjeta': {}
        }
    
    # Cargar Euribor
    for ruta, clave in [(PATHS_CSV['12m'], 'euribor_12m'), (PATHS_CSV['3m'], 'euribor_3m')]:
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
        (PATHS_CSV['tedr'], 'tedr_vivienda'),
        (PATHS_CSV['tedr_consumo'], 'tedr_consumo'),
        (PATHS_CSV['pyme'], 'tipo_pyme'),
        (PATHS_CSV['prestamos_corporativos'], 'tipo_corp'),
        (PATHS_CSV['tarjeta_credito'], 'tarjeta')
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

def cargar_macro(years):
    macro = {y: {'inflacion': None, 'desempleo': {}, 'pib': {}} for y in years}
    
    # 1. Cargar Inflación
    ruta_inf = PATHS_CSV.get('inflacion')
    if ruta_inf and os.path.exists(ruta_inf):
        with open(ruta_inf, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            headers = []
            for row in reader:
                if not row: continue
                if "Country Name" in row:
                    headers = row
                    continue
                if row and len(row) > 0 and 'espa' in row[0].lower():
                    for y in years:
                        str_y = str(y)
                        if str_y in headers:
                            idx = headers.index(str_y)
                            if idx < len(row) and row[idx].strip():
                                try:
                                    macro[y]['inflacion'] = float(row[idx])
                                except:
                                    pass
                    break

    # 2. Cargar Desempleo
    ruta_des = PATHS_CSV.get('desempleo')
    if ruta_des and os.path.exists(ruta_des):
        df_des = pd.read_csv(ruta_des)
        df_des['observation_date'] = pd.to_datetime(df_des['observation_date'])
        for y in years:
            df_y = df_des[df_des['observation_date'].dt.year == y]
            for _, r in df_y.iterrows():
                m = r['observation_date'].month
                val = r.iloc[1]
                macro[y]['desempleo'][m] = val
                if m+1 <= 12: macro[y]['desempleo'][m+1] = val
                if m+2 <= 12: macro[y]['desempleo'][m+2] = val

    # 3. Cargar PIB (Cálculo YoY)
    ruta_pib = PATHS_CSV.get('pib')
    if ruta_pib and os.path.exists(ruta_pib):
        df_pib = pd.read_csv(ruta_pib)
        df_pib['observation_date'] = pd.to_datetime(df_pib['observation_date'])
        df_pib = df_pib.sort_values('observation_date')
        df_pib['year'] = df_pib['observation_date'].dt.year
        df_pib['month'] = df_pib['observation_date'].dt.month
        
        df_pib['pib_yoy'] = df_pib.iloc[:, 1].pct_change(periods=4) * 100
        
        for y in years:
            df_y = df_pib[df_pib['year'] == y]
            for _, r in df_y.iterrows():
                m = r['month']
                val = r['pib_yoy']
                if not pd.isna(val):
                    macro[y]['pib'][m] = np.round(val, 2)
                    if m+1 <= 12: macro[y]['pib'][m+1] = np.round(val, 2)
                    if m+2 <= 12: macro[y]['pib'][m+2] = np.round(val, 2)
                    
    return macro

def generar_datos(years, n_filas=1000, caos=False):
    tasas = cargar_tasas(years)
    macro = cargar_macro(years)
    arquetipos = ['A', 'B', 'C', 'D', 'E']
    probabilidades = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    datos = []
    # Dividir el número total de filas equitativamente entre los años
    n_por_ano = n_filas // len(years) if len(years) > 0 else n_filas
    
    for y in years:
        elecciones = np.random.choice(arquetipos, size=n_por_ano, p=probabilidades)
        for arq in elecciones:
            mes = int(np.random.randint(1, 13))
            t = tasas[y]
            
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
            
            tie = fila.get('tedr_pct')
            e12 = fila.get('euribor_12m_pct')
            e3 = fila.get('euribor_3m_pct')
            
            # Cruce de arquetipos (10% de error)
            if caos and np.random.rand() < 0.10:
                segmentos = ['PARTICULARES', 'PYMES', 'CORPORATIVO']
                fila['client_segment_id'] = np.random.choice([s for s in segmentos if s != fila['client_segment_id']])
                
                familias = ['HIPOTECAS', 'TARJETA', 'PRESTAMOS', 'COMERC', 'SINDICA']
                if np.random.rand() < 0.5:
                    fila['family_id'] = np.random.choice([f for f in familias if f != fila['family_id']])
            
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
                
                fila['spread_pct'] = np.round(spread_dec * 100, 3)
                
                # Factor de negociación
                if caos:
                    factor_negociacion = np.round(np.random.uniform(-0.4, 0.4), 3)
                    fila['spread_pct'] += factor_negociacion
                
                fila['spread_pct_real'] = fila['spread_pct'] # Compatibilidad
                fila['ref_pct'] = ref_dec * 100
                fila['m_freq'] = m
            else:
                fila['spread_pct'] = None
                fila['spread_pct_real'] = None
                fila['ref_pct'] = None
                fila['m_freq'] = 12
                
            # Añadir variables macroeconómicas reales
            y_macro = macro.get(y, {})
            
            # Inflación (anual)
            inf = y_macro.get('inflacion')
            if inf is None:
                base_rate = tie if (tie is not None and not pd.isna(tie)) else 3.0
                fila['inflation_rate'] = np.round(0.5 * base_rate + np.random.normal(1.0, 0.5), 2)
            else:
                fila['inflation_rate'] = np.round(inf, 2)
                
            # Crecimiento del PIB (trimestral interpolado)
            pib = y_macro.get('pib', {}).get(mes)
            if pib is None:
                base_rate = tie if (tie is not None and not pd.isna(tie)) else 3.0
                fila['gdp_growth'] = np.round(3.5 - 0.3 * base_rate + np.random.normal(0, 0.5), 2)
            else:
                fila['gdp_growth'] = np.round(pib, 2)
                
            # Desempleo (trimestral interpolado)
            des = y_macro.get('desempleo', {}).get(mes)
            if des is None:
                base_rate = tie if (tie is not None and not pd.isna(tie)) else 3.0
                fila['unemployment_rate'] = np.round(10.0 + 0.2 * base_rate + np.random.normal(0, 1.0), 2)
            else:
                fila['unemployment_rate'] = np.round(des, 2)
                
            datos.append(fila)
            
    return pd.DataFrame(datos)
