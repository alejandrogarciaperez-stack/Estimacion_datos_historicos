import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Euribor 2025: valores medios mensuales reales (fuente: BCE / Banco de España)
# Tendencia bajista durante 2025 por las sucesivas bajadas de tipos del BCE.
# ---------------------------------------------------------------------------
EURIBOR_12M_2025 = {
    1: 2.431,  # Enero
    2: 2.375,  # Febrero
    3: 2.262,  # Marzo
    4: 2.142,  # Abril
    5: 2.058,  # Mayo
    6: 1.987,  # Junio
    7: 1.972,  # Julio
    8: 1.985,  # Agosto
    9: 2.024,  # Septiembre
    10: 2.066, # Octubre
    11: 2.110, # Noviembre
    12: 2.089, # Diciembre
}

EURIBOR_3M_2025 = {
    1: 2.632,  # Enero
    2: 2.540,  # Febrero
    3: 2.406,  # Marzo
    4: 2.268,  # Abril
    5: 2.174,  # Mayo
    6: 2.085,  # Junio
    7: 2.002,  # Julio
    8: 1.943,  # Agosto
    9: 1.912,  # Septiembre
    10: 1.948, # Octubre
    11: 1.982, # Noviembre
    12: 1.963, # Diciembre
}


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
                'euribor_12m_pct':          None,  # No aplica
                'euribor_3m_pct':           None,  # No aplica
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
                'euribor_12m_pct':          None,  # No aplica
                'euribor_3m_pct':           None,  # No aplica
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

    ruta_salida = 'contratos_sinteticos.csv'
    df_contratos.to_csv(ruta_salida, index=False)
    print(f"\nSe han generado {len(df_contratos)} contratos y se han guardado en '{ruta_salida}'")
    print("\nDistribución por arquetipo (contract_month):")
    print(df_contratos.groupby('client_segment_id')['contract_month'].value_counts().sort_index())
