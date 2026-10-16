# EDA

La exploración revisa dimensiones, tipos, estadísticas, valores faltantes, duplicados, la distribución de popularidad,
las variables musicales, la popularidad por género y las correlaciones, siempre **con y sin las canciones de
popularidad 0**.

## Preparación

- Se eliminan `Unnamed: 0`, `track_id`, `track_name`, `album_name` y `artists`; `popularity` es el objetivo y
  `track_genre` la variable categórica.
- Se deja **una fila por canción** en tres pasos (la primera ocurrencia se conserva):

| Paso | Filas | Ceros |
|---|---|---|
| Dataset original | 114.000 | 16.020 (14,1%) |
| Sin `track_id` repetido | 89.741 | 9.448 (10,5%) |
| Sin audio idéntico con otro `track_id` | 83.479 | 5.598 (6,7%) |
| Sin título + artista repetido | 80.932 | 5.425 (6,7%) |

- Split 80/20 (`random_state=42`): train 64.745 y test 16.187 filas. `es_cero` no se guarda en los CSV.

Este criterio (una fila por canción) difiere del EP1 y es intencional: evita que la misma canción, o la misma
grabación con otro `track_id`, quede a la vez en train y en test. Justificación en la sección 4 de `1_EDA.ipynb`.

## Hallazgos sobre los ceros

- Las canciones repetidas tienen 39,9% de ceros frente a 5,2% de las no repetidas.
- En audio las diferencias son pequeñas (más acústicos, menos enérgicos y más silenciosos).
- El audio casi no permite predecirlos (AUC 0,68; con género 0,89).
- Se concentran en pocos géneros: 13 géneros tienen 20% o más de ceros y 32 géneros menos de 1%.
- Las correlaciones con la popularidad son débiles con y sin ceros (|r| ≈ 0,20 como máximo).
