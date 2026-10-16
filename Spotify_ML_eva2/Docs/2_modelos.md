# Modelos

## Supervisado (regresión)
- Regresión Lineal (línea base) y Random Forest Regressor (`max_depth=None`, `min_samples_leaf=10`, `max_features=0.33`).
- Variables numéricas imputadas (mediana) y estandarizadas; género en one-hot.
- Cada modelo se entrena **con** y **sin** ceros y se evalúa en 4 escenarios (test completo / test sin ceros).
- Métricas: MAE, RMSE, R² y KPIs de negocio (% de canciones con error ≤ 10 puntos, precisión y lift del 20% mejor puntuado).

| Escenario | RF MAE | RF R² | Lineal R² |
|---|---|---|---|
| Con ceros, test completo | 10,31 | 0,423 | 0,396 |
| Con ceros, test sin ceros | 9,21 | 0,469 | 0,450 |
| Sin ceros, test sin ceros | 8,74 | 0,523 | 0,495 |
| Sin ceros, test completo | 10,45 | 0,372 | 0,339 |

- Diagnóstico de la línea horizontal: con `max_depth=15` el 37% de las predicciones caía en ≈ 39; sin límite de
  profundidad el valor más repetido concentra 0,6%.
- Solo con audio el R² es 0,16 (Random Forest) y 0,06 (lineal); con género 0,42 y 0,40.
- Variante multi-hot (todos los géneros de cada canción): Random Forest R² 0,451 / 0,489 / 0,540 / 0,385 en los cuatro escenarios.

## No supervisado
- K-Means con K=5 sobre 11 variables de audio estandarizadas (80.932 canciones).
- Método del codo para K entre 2 y 8, validado con silhouette: K=3 es el mejor (0,257); K=5 vale 0,161 y se elige
  por interpretabilidad.
- Perfil de clusters con y sin ceros, géneros predominantes y PCA para visualización.

## Artefactos generados
- `output/Modelos/`: 4 modelos `.joblib` (`regresion_lineal_*` y `random_forest_*`, `con_ceros` / `sin_ceros`),
  `metricas_modelos.csv`, `kpis_negocio.csv`, `perfil_clusters.csv`, `silhouette_scores.csv`, `generos_por_cluster.csv`.
- `output/Tablas/`: tablas CSV de respaldo del EDA y de la modelización.
- `output/Graficos/`: gráficos PNG.
