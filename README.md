# Spotify ML - EP2 (Caso C: Spotify Tracks)

Proyecto de Machine Learning (MLY1101) que predice la popularidad de canciones de Spotify (regresión) y descubre
segmentos de canciones (K-Means). Integrantes: Bastián Román, Sebastián Aird, Vicente Contreras e Ignacio Gómez.

## Resumen de resultados

| Tema | Resultado |
|---|---|
| Datos | 114.000 filas → **80.932 canciones únicas** (train 64.745 / test 16.187) |
| Canciones con popularidad 0 | 14,1% en el CSV original; **6,7%** tras eliminar duplicados |
| Random Forest (con ceros) | R² 0,42 · error medio 10,3 puntos · 64% de las canciones con error ≤ 10 puntos |
| Random Forest (sin ceros, evaluado sin ceros) | R² 0,52 · error medio 8,7 puntos |
| Regresión Lineal (línea base) | R² 0,40 con ceros · 0,50 sin ceros: casi iguala al Random Forest |
| Multi-hot de géneros | Mejora un poco el Random Forest (R² 0,45 con ceros; 0,54 sin ceros) |
| Uso práctico | Revisando el 20% mejor puntuado se captura el 54–60% de los éxitos (el azar daría 20%) |
| K-Means | K=5 por interpretabilidad (silhouette 0,161; K=3 es el mejor estadísticamente: 0,257) |

Detalle completo, limitaciones y sesgos en `Informe_Tecnico_EP2_Spotify.md`.

## Estructura

```text
Spotify_ML/Spotify_ML_eva2/
├── Data/
│   ├── Raw/Spotify_Tracks_Dataset.csv
│   └── Processed/{Train/train.csv, Test/test.csv}
├── Notebook/Exploracion_inicial/
│   ├── 1_EDA.ipynb
│   └── 2_Modelizacion.ipynb
├── src/
│   ├── preprocess.py   # deduplicado, limpieza, split, preprocesador
│   ├── train.py        # modelos, entrenamiento y persistencia
│   ├── test.py         # evaluación de modelos guardados
│   ├── metrica.py      # métricas y KPIs de negocio
│   └── grafico.py      # gráficos (matplotlib)
├── output/
│   ├── Graficos/       # PNG usados en el informe y la presentación
│   ├── Modelos/        # métricas, KPIs, clusters y 4 modelos .joblib
│   └── Tablas/         # tablas CSV de respaldo del EDA y la modelización
├── Docs/               # 1_EDA.md, 2_modelos.md
├── Environment/requirements.txt
├── Informe_Tecnico_EP2_Spotify.md
└── README.md
```

## Notebooks

1. **`1_EDA.ipynb`**: carga, calidad, duplicados, preparación (una fila por canción), distribución de popularidad,
   comparación de las canciones con popularidad 0 contra el resto (audio y género), correlaciones con y sin ceros,
   división train/test.
2. **`2_Modelizacion.ipynb`**: Regresión Lineal y Random Forest entrenados con y sin ceros y evaluados en 4 escenarios,
   gráfico real vs. predicho y diagnóstico de la línea horizontal, aporte del género, KPIs de negocio y uso para
   Spotify, persistencia de modelos, variante multi-hot de géneros, K-Means (con y sin ceros) y PCA.
   Incluye diagramas de cajas (ceros vs. resto, géneros y clusters).

Cada notebook se entrega ejecutado, con sus resultados visibles.

## Decisiones importantes

- **Una fila por canción** antes de dividir train/test, en tres pasos (mismo `track_id`, mismo audio con otro
  `track_id`, mismo título + artista). Así una canción no puede estar a la vez en train y test (fuga de datos).
- **Los ceros se analizan aparte**: todos los gráficos comparativos usan naranjo = con ceros y azul = sin ceros.
  La marca `es_cero` se calcula dentro de los notebooks y no se guarda en los CSV (sería una copia del objetivo).
- **Variables numéricas estandarizadas** (necesario para la Regresión Lineal; sin ello daba R² ≈ 0,01 por mala convergencia del ajuste).
- **Random Forest** con `max_depth=None, min_samples_leaf=10, max_features=0.33`: la configuración anterior
  (`max_depth=15`) producía una línea horizontal en el gráfico real vs. predicho porque los árboles gastaban su
  profundidad separando géneros (113 columnas one-hot).

## Artefactos en `output/`

```text
output/Modelos/
├── regresion_lineal_con_ceros.joblib   regresion_lineal_sin_ceros.joblib
├── random_forest_con_ceros.joblib      random_forest_sin_ceros.joblib
├── metricas_modelos.csv  kpis_negocio.csv  perfil_clusters.csv
└── silhouette_scores.csv  generos_por_cluster.csv
```

Los `.joblib` dependen de la versión de scikit-learn con que se guardaron (1.9.1). Si se cargan con otra versión,
conviene volver a ejecutar `2_Modelizacion.ipynb` (Run All; cada Random Forest tarda ~1 minuto).

## Instalación y ejecución

Python 3.12. El README está en la raíz del repositorio; el proyecto vive en la carpeta `Spotify_ML_eva2/`.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r Environment\requirements.txt
```

Abrir el proyecto en VS Code, seleccionar `.venv\Scripts\python.exe` como intérprete y ejecutar en orden
`1_EDA.ipynb` y luego `2_Modelizacion.ipynb`.

Duoc UC · MLY1101 Machine Learning · EP2
