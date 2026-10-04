# Evaluación Parcial N°2 — Caso C: Inteligencia musical y predicción de popularidad de canciones

**Asignatura:** MLY1101 – Machine Learning
**Evaluación:** Evaluación Parcial N°2 – Presentación y defensa técnica del proyecto (40%)
**Caso:** C — Spotify Tracks
**Integrantes:** Bastián Román, Sebastián Aird, Ignacio Gómez y Vicente Contreras

Este informe documenta la etapa de modelamiento (supervisado y no supervisado) que continúa el trabajo de
preparación de datos y análisis exploratorio realizado en la Evaluación Parcial N°1. Es un documento
reproducible: cada resultado reportado aquí proviene de ejecutar `Notebook/Exploracion_inicial/1_EDA.ipynb`
y `Notebook/Exploracion_inicial/2_Modelizacion.ipynb` sobre `Data/Raw/Spotify_Tracks_Dataset.csv`.

---

## 1. Descripción del problema de negocio

En una plataforma de música digital existe una gran cantidad de canciones con características acústicas,
rítmicas y de producción diferentes. El problema de negocio es analizar qué características están asociadas
con la popularidad de una canción, predecir su nivel de popularidad y, adicionalmente, **descubrir segmentos
o "tipos de canción"** que no están explícitos en los metadatos originales, pero que pueden ser útiles para
curaduría de contenido y armado de playlists.

La variable objetivo del problema supervisado es `popularity` (escala 0–100, continua), por lo que se trata
de un **problema de regresión**. El problema no supervisado busca descubrir agrupaciones (clusters) de
canciones con perfiles sonoros similares, sin usar la etiqueta `popularity` ni `track_genre` como criterio
de agrupación.

**Preguntas de negocio:**
1. ¿Podemos estimar la popularidad de una canción a partir de sus características musicales y metadata?
2. ¿Existen segmentos naturales de canciones (más allá del género autodeclarado) que agrupen perfiles
   sonoros similares y que sean útiles para la curaduría de contenido?

*(Encuadre propuesto para esta evaluación; no corresponde a funcionalidades reales de Spotify.)*

## 2. Objetivos del proyecto

**Objetivo general:** desarrollar y comparar modelos de Machine Learning supervisados que permitan predecir
la popularidad de una canción, y aplicar una técnica no supervisada que permita identificar segmentos
relevantes dentro del catálogo, usando el dataset Spotify Tracks.

**Objetivos específicos:**
1. Preparar los datos para el modelamiento (limpieza, codificación de variables categóricas, separación
   train/test sin fuga de información).
2. Implementar y comparar al menos dos modelos de aprendizaje supervisado de regresión.
3. Implementar una técnica de aprendizaje no supervisado (clustering) para descubrir patrones/segmentos.
4. Evaluar el desempeño de los modelos supervisados con métricas apropiadas a un problema de regresión.
5. Interpretar los resultados obtenidos y traducirlos en recomendaciones para el contexto de negocio.
6. Documentar el proceso completo siguiendo la metodología CRISP-DM.

## 3. Definición de KPIs

| KPI | Definición | Propósito |
|---|---|---|
| MAE | Error absoluto medio entre popularidad real y predicha | Medir el error promedio de predicción, en las mismas unidades que `popularity` |
| RMSE | Raíz del error cuadrático medio | Penalizar con más fuerza los errores de predicción grandes |
| R² | Proporción de la varianza de `popularity` explicada por el modelo | Evaluar la capacidad explicativa global del modelo |
| Inercia (WCSS) | Suma de distancias al cuadrado dentro de cada cluster | Guiar la elección del número de clusters K (método del codo) |
| Silhouette score | Cohesión interna vs. separación entre clusters | Validar cuantitativamente la calidad/elección de K |

## 4. Descripción de las fuentes de datos

- **Fuente:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle), construido a partir de la Web API de Spotify.
- **Volumen:** 114.000 registros × 21 columnas — 1.000 canciones por cada uno de los 114 géneros musicales presentes.
- **Variables principales:** identificación (`track_id`, `artists`, `album_name`, `track_name`), variable
  objetivo (`popularity`), duración (`duration_ms`), contenido explícito (`explicit`), 10 características de
  audio (`danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`,
  `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`) y `track_genre`.
- **Calidad general:** solo 1 fila con metadatos faltantes (`artists`, `track_name`, `album_name`, 0.001%
  cada una); 0 filas duplicadas si se consideran todas las columnas, pero 24.259 `track_id` repetidos
  (21,3%) porque una misma canción puede estar catalogada en varios géneros (ver sección 5.2).

## 5. Preparación y análisis exploratorio de los datos (EDA)

### 5.1 Distribución de la variable objetivo
`popularity` tiene media 33.2, mediana 35 y desviación estándar 22.3, con una cola larga hacia valores bajos
(patrón típico de la industria musical: pocas canciones muy populares, muchas con popularidad baja o nula).

### 5.2 Duplicados y decisión de preparación
El dataset no tiene filas 100% duplicadas, pero 24.259 `track_id` se repiten porque la misma canción puede
pertenecer a varios géneros. Para este proyecto se decidió **conservar una sola fila por canción**
(`track_id` único, 89.741 filas finales), a diferencia del criterio usado en el EP1 (que conservaba las
canciones multi-género y solo eliminaba 450 duplicados reales por par `track_id`+`track_genre`).

Este cambio de criterio es intencional: al dejar una sola fila por canción **antes** de dividir train/test,
se elimina por construcción el riesgo de que la misma canción (mismas características de audio) aparezca en
train y en test bajo un género distinto — riesgo de fuga de datos que ya se había identificado en el EP1. El
costo de esta decisión es perder la señal de multi-categorización por género (cada canción queda asociada
solo a su primera ocurrencia en el CSV original). Ver la justificación completa en `1_EDA.ipynb`, sección 8.1.

### 5.3 Relación entre variables musicales y popularidad
Las correlaciones lineales con `popularity` son débiles para todas las variables de audio (|r| < 0.10),
reforzando el hallazgo del EP1: la popularidad depende mayoritariamente de factores externos al sonido de la
canción (fama del artista, marketing, posicionamiento en playlists), no de sus características acústicas.

### 5.4 Preparación aplicada antes del modelamiento
1. Eliminación de columnas no predictivas (`Unnamed: 0`, `track_id`, `track_name`, `album_name`, `artists`).
2. Deduplicado a una fila por canción (sección 5.2).
3. División train/test (80/20, `random_state=42`).
4. Codificación de `track_genre` mediante One-Hot Encoding, ajustado **solo con datos de entrenamiento**
   dentro de un `Pipeline` de scikit-learn (sin fuga de información hacia test).
5. Imputación de valores faltantes (mediana para numéricas, moda para categóricas) dentro del mismo pipeline.

Resultado: `Data/Processed/Train/train.csv` (71.792 filas) y `Data/Processed/Test/test.csv` (17.949 filas),
15 columnas cada uno.

## 6. Modelamiento

### 6.1 Modelos supervisados (comparación de 2 modelos de regresión)

| Modelo | MAE | RMSE | R² | Justificación |
|---|---|---|---|---|
| Regresión Lineal | 17.05 | 20.37 | 0.007 | Línea base interpretable; mide si existe una relación aditiva simple |
| Random Forest Regressor (`n_estimators=100`, `max_depth=15`, `min_samples_leaf=2`) | 14.53 | 18.37 | 0.192 | Captura relaciones no lineales/interacciones entre variables (p. ej. género × energía) |

El Random Forest reduce el error y casi triplica el R² respecto a la Regresión Lineal, confirmando que
existen relaciones no lineales que el modelo lineal no puede representar. Aun así, **ningún modelo alcanza
un poder predictivo alto**: el mejor de los dos deja sin explicar más del 80% de la varianza de `popularity`.

### 6.2 Aprendizaje no supervisado: K-Means

Se aplicó K-Means sobre 11 variables de audio estandarizadas (`StandardScaler`), usando el conjunto completo
(train+test, 89.741 canciones), ya que el clustering no requiere separación train/test al no usar la
etiqueta `popularity`.

- **Elección de K:** se probaron valores de K entre 2 y 8 mediante el método del codo (inercia) y se validó
  cuantitativamente con *silhouette score* (muestra de 10.000 canciones). El silhouette favorece
  numéricamente K=3 (0.251) sobre K=5 (0.166), pero se optó por **K=5** por interpretabilidad de negocio: a
  K=3 los segmentos mezclan perfiles sonoros distintos, mientras que a K=5 cada cluster es claramente
  diferenciable (ver tabla siguiente). Esta es una decisión explícita de compromiso, documentada como tal
  (no se presenta K=5 como el óptimo estadístico).
- **Perfil de los 5 clusters resultantes:**

| Cluster | Tamaño | Perfil dominante | Popularidad media |
|---|---|---|---|
| 0 | 7.119 | sleep, new-age, ambient, classical — baja energía, muy acústico | 28.6 |
| 1 | 30.797 | forro, salsa, kids, afrobeat, disco — alta bailabilidad/energía | 33.8 |
| 2 | 24.036 | black-metal, grindcore, drum-and-bass, heavy-metal — máxima energía | 32.7 |
| 3 | 19.782 | tango, honky-tonk, cantopop, romance, acoustic — acústico tradicional | 33.1 |
| 4 | 8.007 | comedy, emo, dancehall, hardcore — 95.9% contenido explícito | **36.4** |

El cluster 4 concentra casi todo el contenido explícito del dataset y tiene, a la vez, la mayor popularidad
promedio — un patrón que complementa el hallazgo del EP1 sobre la relación entre `explicit` y `popularity`.

## 7. Evaluación e interpretación de resultados (IE8)

- **Desempeño limitado pero consistente con el EDA:** el R² bajo de ambos modelos no es un error de
  implementación, sino el reflejo directo de que las variables de audio disponibles tienen poca relación con
  la popularidad. El Random Forest, al capturar no linealidades, mejora el error pero no resuelve el límite
  estructural de la información disponible.
- **Recomendación de negocio:** no usar el modelo de regresión como predictor único para decisiones críticas
  de catálogo (su error esperado, 14–17 puntos de popularidad, es demasiado alto). Sí es útil como señal
  complementaria de bajo peso, y como argumento para incorporar variables externas al audio (datos del
  artista, señales de redes sociales, presencia en playlists editoriales) en una siguiente iteración.
- **Valor del clustering:** a diferencia del modelo supervisado, la segmentación no supervisada sí entrega
  un resultado directamente accionable: 5 perfiles sonoros interpretables que pueden usarse para curaduría de
  playlists independientemente de si se puede predecir popularidad con precisión.

## 8. Sesgos, ética y privacidad

- **Sesgos:** muestreo artificialmente balanceado por género (1.000 canciones c/u, no representa la
  proporción real del catálogo); el deduplicado a una fila por canción (sección 5.2) introduce un sesgo
  adicional, ya que cada canción multi-género queda arbitrariamente asociada a un solo género (el primero en
  el CSV), lo que puede subrepresentar ciertos cruces de género.
- **Ética:** curar contenido según clusters o popularidad predicha podría marginar artistas emergentes o
  géneros de nicho si se usa sin supervisión humana; el cluster dominado por contenido explícito no debe
  interpretarse como "el contenido explícito causa popularidad" (correlación, no causalidad) ni usarse para
  decisiones de censura o promoción automática.
- **Privacidad:** el dataset no contiene datos personales de oyentes (sin IDs de usuario, ubicación ni
  historial de escucha), solo metadatos públicos de canciones/artistas. Uso restringido al ámbito académico.

## 9. Metodología (CRISP-DM)

| Fase | Estado en este proyecto |
|---|---|
| 1. Comprensión del negocio | Completado — problema, objetivos y KPIs (secciones 1-3) |
| 2. Comprensión de los datos | Completado — descripción de fuente y EDA (`1_EDA.ipynb`, secciones 4-5 de este informe) |
| 3. Preparación de los datos | Completado — limpieza, deduplicado documentado, codificación, split train/test (sección 5.4) |
| 4. Modelado | Completado — 2 modelos supervisados de regresión + K-Means no supervisado (sección 6) |
| 5. Evaluación | Completado — métricas MAE/RMSE/R², silhouette score, interpretación de negocio (secciones 6-7) |
| 6. Despliegue | Fuera del alcance de esta evaluación |

## 10. Conclusiones

El proyecto compara exitosamente dos modelos supervisados de regresión (Regresión Lineal vs. Random Forest)
y aplica una técnica no supervisada (K-Means) con justificación cuantitativa (silhouette score) y cualitativa
(interpretabilidad de negocio) para el número de clusters elegido. El hallazgo central es consistente a lo
largo de todo el proyecto (EP1 y EP2): las características de audio por sí solas tienen un poder predictivo
limitado sobre la popularidad, pero sí permiten segmentar el catálogo en perfiles sonoros útiles para
curaduría de contenido, que es un resultado de negocio igualmente valioso aunque no provenga del modelo de
regresión.

## Estructura del proyecto

```
Spotify_ML_eva2/
├── Data/
│   ├── Raw/Spotify_Tracks_Dataset.csv
│   └── Processed/{Train/train.csv, Test/test.csv}
├── Notebook/Exploracion_inicial/
│   ├── 1_EDA.ipynb
│   └── 2_Modelizacion.ipynb
├── src/ (preprocess.py, train.py, test.py, metrica.py, grafico.py)
├── output/
│   ├── Graficos/ (histogramas, correlaciones, comparación de modelos, codo, PCA, etc.)
│   └── Modelos/ (metricas_modelos.csv, perfil_clusters.csv, silhouette_scores.csv,
│                 regresion_lineal.joblib, random_forest.joblib)
├── Docs/
└── README.md
```
