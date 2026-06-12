import os

# Directorio raíz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorios de datos y análisis
DB_DIR = os.path.join(BASE_DIR, 'Base_de_datos')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'Analisis')

# Crear directorios si no existen
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Rutas a los ficheros CSV de tasas
PATHS_CSV = {
    '12m': os.path.join(DB_DIR, 'euribor_12_meses_datos_historicos.csv'),
    '3m': os.path.join(DB_DIR, 'euribor_3_meses_datos_historicos.csv'),
    'tedr': os.path.join(DB_DIR, 'tedr_credito_vivienda.csv'),
    'tedr_consumo': os.path.join(DB_DIR, 'credito_consumo_tedr.csv'),
    'pyme': os.path.join(DB_DIR, 'tipo_medio_ponderado_pyme.csv'),
    'prestamos_corporativos': os.path.join(DB_DIR, 'tipo_prestamos_mas_un_millon.csv'),
    'tarjeta_credito': os.path.join(DB_DIR, 'Tarjetas_credito_tipo.csv'),
    'inflacion': os.path.join(DB_DIR, 'inflacion.csv'),
    'desempleo': os.path.join(DB_DIR, 'desempleo.csv'),
    'pib': os.path.join(DB_DIR, 'crecimiento_pib.csv')
}
