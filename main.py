import pandas as pd 

#Leer archivos csv
#Datos de ventas
ruta_archivo = r"C:\Proyectos\Empresa\Caso_Pedro\Base_de_datos\tipo_interes.csv"
# Leer el archivo raw
df_raw = pd.read_csv(ruta_archivo, encoding='latin1')

# 1. Extraer los metadatos (primeras 5 filas)
# Fila 0: Número secuencial
# Fila 1: Alias de la serie
# Fila 2: Descripción de la serie
# Fila 3: Descripción de las unidades
# Fila 4: Frecuencia
metadatos = df_raw.iloc[:5].copy()

# 2. Extraer las observaciones (de la fila 5 en adelante)
df_datos = df_raw.iloc[5:].copy()

# Renombrar la primera columna como 'Fecha'
df_datos.rename(columns={df_raw.columns[0]: 'Fecha'}, inplace=True)

# Convertir el resto de columnas a valores numéricos (ya que cargaron como texto debido a los metadatos)
for col in df_datos.columns[1:]:
    df_datos[col] = pd.to_numeric(df_datos[col], errors='coerce')

# 3. Opciones para nombrar las columnas (variables)
# Opción A: Mantener los códigos de las series (DF_MESN7...)
# Opción B: Usar los Alias legibles (ej: %BE_19_16.4)
# Opción C: Usar la descripción completa de la serie (ej: Tipo de interés...)

# Usaremos los ALIAS legibles para las columnas para mayor claridad
columnas_alias = ['Fecha'] + list(metadatos.iloc[1, 1:])
df_ordenado = df_datos.copy()
df_ordenado.columns = columnas_alias

# Establecer la fecha como índice para que las filas representen cada observación temporal
df_ordenado.set_index('Fecha', inplace=True)

# Mostrar las primeras filas y columnas de la tabla bien organizada
print("=== TABLA LIMPIA Y ORDENADA (Variables en Columnas, Observaciones en Filas) ===")
print(df_ordenado.iloc[:10, :5])  # Muestra las primeras 10 filas y 5 columnas para verificar

# Guardar la tabla limpia en un nuevo archivo CSV
ruta_limpio = r"C:\Proyectos\Empresa\Caso_Pedro\Base_de_datos\tipo_interes_limpio.csv"
df_ordenado.to_csv(ruta_limpio, encoding='utf-8')
print(f"\nArchivo limpio guardado en: {ruta_limpio}")
