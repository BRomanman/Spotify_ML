# Spotify ML - EP2

Proyecto de Machine Learning para predecir la popularidad de canciones e identificar segmentos musicales mediante K-Means.

## Estructura

- `Data/Raw/`: dataset original.
- `Data/Processed/`: datos separados en Train y Test.
- `Notebook/Exploracion_inicial/`: notebooks de EDA y modelizacion.
- `src/`: codigo reutilizable.
- `output/`: graficos y resultados generados.
- `Docs/`: documentacion.
- `Environment/`: instrucciones y dependencias.

## Flujo

Dataset Raw -> Preprocesamiento -> Train/Test -> EDA -> Modelizacion -> Evaluacion

## Ejecucion

1. Instalar dependencias:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r Environment\\requirements.txt
```

2. Abrir `Notebook/Exploracion_inicial/1_EDA.ipynb`.
3. Ejecutar `Notebook/Exploracion_inicial/2_Modelizacion.ipynb`.

No se utiliza Conda en esta estructura.
