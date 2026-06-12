import os
import pandas as pd
from config import DB_DIR
from data_utils import generar_datos

if __name__ == '__main__':
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    print(f"Generando datos sintéticos para regresión: {years}...")
    
    # 5000 filas como solicitó el usuario
    df_contratos = generar_datos(years, 5000)

    print(df_contratos[['year', 'contract_month', 'family_id', 'tedr_pct', 'inflation_rate', 'gdp_growth', 'unemployment_rate']].head(10).to_string())

    ruta_salida_csv = os.path.join(DB_DIR, 'contratos_regresion_2018_2025.csv')
    df_contratos.to_csv(ruta_salida_csv, index=False)
    
    ruta_salida_excel = os.path.join(DB_DIR, 'contratos_regresion_2018_2025.xlsx')
    df_contratos.to_excel(ruta_salida_excel, index=False)
    
    print(f"\nSe han generado {len(df_contratos)} contratos.")
    print(f"Guardado en:\n- {ruta_salida_csv}\n- {ruta_salida_excel}")
    print("\nDistribución por año:")
    print(df_contratos['year'].value_counts().sort_index())

    print("\nGenerando datos sintéticos con CAOS para regresión...")
    df_contratos_caos = generar_datos(years, 5000, caos=True)
    ruta_salida_csv_caos = os.path.join(DB_DIR, 'contratos_regresion_caos_2018_2025.csv')
    df_contratos_caos.to_csv(ruta_salida_csv_caos, index=False)
    ruta_salida_excel_caos = os.path.join(DB_DIR, 'contratos_regresion_caos_2018_2025.xlsx')
    df_contratos_caos.to_excel(ruta_salida_excel_caos, index=False)
    print(f"Guardado caos en:\n- {ruta_salida_csv_caos}\n- {ruta_salida_excel_caos}")
