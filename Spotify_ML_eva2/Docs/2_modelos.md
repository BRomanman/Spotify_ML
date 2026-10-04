# Modelos

## Supervisado
- Regresion Lineal
- Random Forest Regressor
- Metricas: MAE, RMSE y R2

## No supervisado
- K-Means con K=5
- Metodo del codo para K entre 2 y 8, validado con silhouette score
  (favorece K=3 con 0.251; K=5 = 0.166, elegido por interpretabilidad de los clusters, no por ser el optimo estadistico)
- Perfil de clusters
- Generos predominantes
- PCA para visualizacion

## Artefactos generados
- `output/Modelos/regresion_lineal.joblib`, `random_forest.joblib`: modelos entrenados persistidos
- `output/Modelos/metricas_modelos.csv`, `silhouette_scores.csv`, `perfil_clusters.csv`
