import os
import pandas as pd
from config import DB_DIR
from data_utils import generar_datos

if __name__ == '__main__':
    print("Generando datos sintéticos para 2025...")
    df_contratos = generar_datos([2025], 1000)

    print(df_contratos.head(10).to_string())

    ruta_salida = os.path.join(DB_DIR, 'contratos_sinteticos.xlsx')
    df_contratos.to_excel(ruta_salida, index=False)
    
    print(f"\nSe han generado {len(df_contratos)} contratos y se han guardado en '{ruta_salida}'")
    print("\nDistribución por arquetipo (contract_month):")
    print(df_contratos.groupby('client_segment_id')['contract_month'].value_counts().sort_index())

    print("\nGenerando datos sintéticos con CAOS para 2025...")
    df_contratos_caos = generar_datos([2025], 1000, caos=True)
    ruta_salida_caos = os.path.join(DB_DIR, 'contratos_sinteticos_caos.xlsx')
    df_contratos_caos.to_excel(ruta_salida_caos, index=False)
    print(f"Se han generado {len(df_contratos_caos)} contratos con caos y se han guardado en '{ruta_salida_caos}'")
