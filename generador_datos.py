import pandas as pd
import numpy as np
import openpyxl

import os

def cargar_tasas_desde_csv():
    ruta_12m = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/euribor_12_meses_datos_historicos.csv'
    ruta_3m = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/euribor_3_meses_datos_historicos.csv'
    ruta_tedr = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/tedr_credito_vivienda.csv'
    ruta_tedr_consumo = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/credito_consumo_tedr.csv'
    ruta_pyme = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/tipo_medio_ponderado_pyme.csv'
    ruta_prestamos_corporativos = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/tipo_prestamos_mas_un_millon.csv'
    ruta_tarjeta_credito = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/Tarjetas_credito_tipo.csv'
    
    euribor_12m_2025 = {}
    euribor_3m_2025 = {}
    tedr_vivienda_2025 = {}
    tedr_consumo_2025 = {}
    tipo_pyme_2025 = {}
    tipo_prestamos_corporativos_2025 = {}
    tipo_tarjeta_credito_2025 = {}
    
    # Cargar Euribor 12 meses
    if os.path.exists(ruta_12m):
        df_12m = pd.read_csv(ruta_12m)
        df_12m['DATE'] = pd.to_datetime(df_12m['DATE'])
        df_12m_2025 = df_12m[df_12m['DATE'].dt.year == 2025]
        for _, row in df_12m_2025.iterrows():
            mes = row['DATE'].month
            valor = float(row.iloc[2])
            euribor_12m_2025[mes] = np.round(valor, 3)
            
    # Cargar Euribor 3 meses
    if os.path.exists(ruta_3m):
        df_3m = pd.read_csv(ruta_3m)
        df_3m['DATE'] = pd.to_datetime(df_3m['DATE'])
        df_3m_2025 = df_3m[df_3m['DATE'].dt.year == 2025]
        for _, row in df_3m_2025.iterrows():
            mes = row['DATE'].month
            valor = float(row.iloc[2])
            euribor_3m_2025[mes] = np.round(valor, 3)

    # Cargar TEDR Crédito Vivienda
    meses_map = {'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12}
    if os.path.exists(ruta_tedr):
        # Leemos el archivo línea por línea ya que tiene metadatos en las primeras líneas
        with open(ruta_tedr, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '25"' in line:  # Buscamos las fechas que terminan en 25 (ej. "ene25")
                    partes = line.strip().split(',')
                    if len(partes) == 2:
                        fecha = partes[0].strip('"')
                        if fecha.endswith('25'):
                            mes_str = fecha[:3]
                            if mes_str in meses_map:
                                mes = meses_map[mes_str]
                                tedr_vivienda_2025[mes] = float(partes[1])

    # Cargar TEDR Crédito Consumo
    if os.path.exists(ruta_tedr_consumo):
        with open(ruta_tedr_consumo, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '25"' in line:
                    partes = line.strip().split(',')
                    if len(partes) == 2:
                        fecha = partes[0].strip('"')
                        if fecha.endswith('25'):
                            mes_str = fecha[:3]
                            if mes_str in meses_map:
                                mes = meses_map[mes_str]
                                tedr_consumo_2025[mes] = float(partes[1])

    # Cargar Tipo Medio Pyme
    if os.path.exists(ruta_pyme):
        with open(ruta_pyme, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '25"' in line:
                    partes = line.strip().split(',')
                    if len(partes) == 2:
                        fecha = partes[0].strip('"')
                        if fecha.endswith('25'):
                            mes_str = fecha[:3]
                            if mes_str in meses_map:
                                mes = meses_map[mes_str]
                                tipo_pyme_2025[mes] = float(partes[1])  

    # Cargar Tipo de Préstamos más de un millón (Corporativo)
    if os.path.exists(ruta_prestamos_corporativos):
        with open(ruta_prestamos_corporativos, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '25"' in line:
                    partes = line.strip().split(',')
                    if len(partes) == 2:
                        fecha = partes[0].strip('"')
                        if fecha.endswith('25'):
                            mes_str = fecha[:3]
                            if mes_str in meses_map:
                                mes = meses_map[mes_str]
                                tipo_prestamos_corporativos_2025[mes] = float(partes[1])

    # Cargar Tarjeta de Crédito
    if os.path.exists(ruta_tarjeta_credito):
        with open(ruta_tarjeta_credito, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '25"' in line:
                    partes = line.strip().split(',')
                    if len(partes) == 2:
                        fecha = partes[0].strip('"')
                        if fecha.endswith('25'):
                            mes_str = fecha[:3]
                            if mes_str in meses_map:
                                mes = meses_map[mes_str]
                                tipo_tarjeta_credito_2025[mes] = float(partes[1])
            
    return euribor_12m_2025, euribor_3m_2025, tedr_vivienda_2025, tedr_consumo_2025, tipo_pyme_2025, tipo_prestamos_corporativos_2025, tipo_tarjeta_credito_2025

EURIBOR_12M_2025, EURIBOR_3M_2025, TEDR_VIVIENDA_2025, TEDR_CONSUMO_2025, TIPO_PYME_2025, TIPO_PRESTAMOS_CORPORATIVOS_2025, TIPO_TARJETA_CREDITO_2025 = cargar_tasas_desde_csv()


def generar_datos_sinteticos(n_filas=1000):
    # Probabilidades iguales para cada arquetipo (20% cada uno)
    arquetipos = ['A', 'B', 'C', 'D', 'E']
    probabilidades = [0.2, 0.2, 0.2, 0.2, 0.2]

    # Asignar un arquetipo a cada fila
    elecciones = np.random.choice(arquetipos, size=n_filas, p=probabilidades)

    datos = []

    for arq in elecciones:
        # Mes de contratación: aleatorio entre enero (1) y diciembre (12) de 2025
        mes = int(np.random.randint(1, 13))

        if arq == 'A':
            # Arquetipo A: La Hipoteca Tradicional
            # Referencia: Euribor 12M (se revisa anualmente, estándar en España)
            # Sólo aplica a hipotecas a tipo variable; la fija no lleva índice.
            fila = {
                'contract_month':           mes,
                'client_segment_id':        'PARTICULARES',
                'customer_type_id':         'PERSONA F',
                'entity_id':                'BANCO COMERCIAL',
                'family_id':                'HIPOTECAS',
                'counterpart_id':           np.random.choice(['TIPO_FIJO', 'TIPO_VARIABLE']),
                'contract_collateral_type': 'INMUEBLES',
                'local_curr_orig_risk_amount': np.round(np.random.uniform(80_000, 400_000), 2),
                'euribor_12m_pct':          EURIBOR_12M_2025[mes],  # Referencia para hipoteca variable
                'euribor_3m_pct':           None,                   # No aplica a particulares/hipotecas
                'tedr_pct':                 TEDR_VIVIENDA_2025.get(mes, None), # TEDR vivienda
            }

        elif arq == 'B':
            # Arquetipo B: La Tarjeta de Crédito
            # Sin referencia Euribor: aplica un TIN/TAE fijo propio del producto
            fila = {
                'contract_month':           mes,
                'client_segment_id':        'PARTICULARES',
                'customer_type_id':         'PERSONA F',
                'entity_id':                'BANCO COMERCIAL',
                'family_id':                'TARJETA',
                'counterpart_id':           'TARJETA_CLASICA',
                'contract_collateral_type': 'NINGUNO',
                'local_curr_orig_risk_amount': np.round(np.random.uniform(1_000, 6_000), 2),
                'euribor_12m_pct':          EURIBOR_12M_2025[mes],  # No aplica
                'euribor_3m_pct':           None,  # No aplica
                'tedr_pct':                 TIPO_TARJETA_CREDITO_2025.get(mes, None), # Tarjeta de crédito
            }

        elif arq == 'C':
            # Arquetipo C: Préstamos Personales / Coche
            # Sin referencia Euribor: tipo fijo acordado en contrato
            fila = {
                'contract_month':           mes,
                'client_segment_id':        'PARTICULARES',
                'customer_type_id':         'PERSONA F',
                'entity_id':                'FINANCIERA DE CONSUMO',
                'family_id':                'PRESTAMOS',
                'counterpart_id':           'PRESTAMO_PERSONAL',
                'contract_collateral_type': 'PERSONAL',
                'local_curr_orig_risk_amount': np.round(np.random.uniform(5_000, 35_000), 2),
                'euribor_12m_pct':          EURIBOR_12M_2025[mes],  # No aplica
                'euribor_3m_pct':           None,  # No aplica
                'tedr_pct':                 TEDR_CONSUMO_2025.get(mes, None), # TEDR consumo
            }

        elif arq == 'D':
            # Arquetipo D: Línea de Crédito PYME
            # Referencia: Euribor 3M (revisión trimestral, estándar en financiación empresarial)
            fila = {
                'contract_month':           mes,
                'client_segment_id':        'PYMES',
                'customer_type_id':         'JU',
                'entity_id':                'BANCO COMERCIAL',
                'family_id':                'COMERC',
                'counterpart_id':           'LINEA DE CREDITO',
                'contract_collateral_type': 'PERSONAL',
                'local_curr_orig_risk_amount': np.round(np.random.uniform(20_000, 150_000), 2),
                'euribor_12m_pct':          None,                  # No aplica a empresas
                'euribor_3m_pct':           EURIBOR_3M_2025[mes],  # Referencia trimestral
                'tedr_pct':                 TIPO_PYME_2025.get(mes, None), # Tipo medio PYME
            }

        elif arq == 'E':
            # Arquetipo E: Gran Crédito Corporativo / Sindicado
            # Referencia: Euribor 3M (mercado corporativo e interbancario)
            fila = {
                'contract_month':           mes,
                'client_segment_id':        'CORPORATIVO',
                'customer_type_id':         'JU',
                'entity_id':                'BANCA E INVERSION/MAYORISTA',
                'family_id':                'SINDICA',
                'counterpart_id':           'PRESTAMO SINDICADO',
                'contract_collateral_type': 'MOBILIARIA(ACCIONES/FONDO)',
                'local_curr_orig_risk_amount': np.round(np.random.uniform(1_000_000, 15_000_000), 2),
                'euribor_12m_pct':          None,                  # No aplica a corporativos
                'euribor_3m_pct':           EURIBOR_3M_2025[mes],  # Referencia trimestral
                'tedr_pct':                 TIPO_PRESTAMOS_CORPORATIVOS_2025.get(mes, None), # Tipo de préstamos más de un millón
            }
        # Cálculo del spread a partir de la TIE (tedr_pct)
        tie = fila.get('tedr_pct')
        eur12 = fila.get('euribor_12m_pct')
        eur3 = fila.get('euribor_3m_pct')
        
        if tie is not None and not pd.isna(tie):
            tie_dec = tie / 100.0
            if eur12 is not None and not pd.isna(eur12):
                ref_dec = eur12 / 100.0
                m = 12
            elif eur3 is not None and not pd.isna(eur3):
                ref_dec = eur3 / 100.0
                m = 3
            else:
                ref_dec = 0.0
                m = 12
            
            tin_dec = m * ((1 + tie_dec)**(1/m) - 1)
            spread_dec = tin_dec - ref_dec
            fila['spread_pct'] = np.round(spread_dec * 100, 3)
        else:
            fila['spread_pct'] = None

        datos.append(fila)

    df = pd.DataFrame(datos)
    return df


if __name__ == '__main__':
    print("Generando datos sintéticos...")
    df_contratos = generar_datos_sinteticos(1000)

    print(df_contratos.head(10).to_string())

    ruta_salida = r'C:/Proyectos/Empresa/Caso_Pedro/Base_de_datos/contratos_sinteticos.xlsx'
    df_contratos.to_excel(ruta_salida, index=False)
    print(f"\nSe han generado {len(df_contratos)} contratos y se han guardado en '{ruta_salida}'")
    print("\nDistribución por arquetipo (contract_month):")
    print(df_contratos.groupby('client_segment_id')['contract_month'].value_counts().sort_index())
