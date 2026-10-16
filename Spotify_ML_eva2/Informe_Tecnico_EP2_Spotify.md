# Evaluación Parcial N°2 — Caso C: Inteligencia musical y predicción de popularidad de canciones

**Asignatura:** MLY1101 – Machine Learning
**Evaluación:** Evaluación Parcial N°2 – Presentación y defensa técnica del proyecto (40%)
**Caso:** C — Spotify Tracks
**Integrantes:** Bastián Román, Sebastián Aird, Ignacio Gómez y Vicente Contreras

Este informe documenta la etapa de modelamiento (supervisado y no supervisado) que continúa el trabajo de
preparación de datos y análisis exploratorio realizado en la Evaluación Parcial N°1. Es un documento
reproducible: cada resultado reportado aquí proviene de ejecutar `Notebook/Exploracion_inicial/1_EDA.ipynb`
y `Notebook/Exploracion_inicial/2_Modelizacion.ipynb` sobre `Data/Raw/Spotify_Tracks_Dataset.csv`
(las tablas de respaldo quedan en `output/Tablas/` y `output/Modelos/`).

**Cambios incorporados tras la revisión con el profesor**

1. **Los ceros de popularidad se analizan por separado.** El EDA y el modelamiento se hacen *con ceros* y *sin ceros*, y todos los gráficos comparativos usan la misma convención de color (naranjo = con ceros, azul = sin ceros).
2. **Se explicó y corrigió la línea horizontal** del gráfico real vs. predicho del Random Forest (era un defecto de configuración del modelo; sección 6.2).
3. **KPIs entendibles para negocio** y un caso de uso concreto para Spotify (secciones 3 y 7).
4. **Se detectó y eliminó una fuga de datos adicional**: la misma grabación publicada con distintos `track_id` quedaba en train y en test (sección 5.1).
5. **Se corrigió la Regresión Lineal**: en una versión anterior le faltaba estandarizar las variables numéricas y daba R² ≈ 0,01; bien configurada llega a R² ≈ 0,40 (sección 6.1).
6. **Se probó la variante multi-hot de géneros** propuesta por el profesor (sección 6.4) y se agregaron **diagramas de cajas** a los gráficos de los ceros, los géneros y los clusters.

---

## 1. Descripción del problema de negocio

En una plataforma de música digital existe una gran cantidad de canciones con características acústicas,
rítmicas y de producción diferentes. El problema de negocio es analizar qué características están asociadas
con la popularidad de una canción, estimar su nivel de popularidad para **priorizar qué canciones revisar o
promover primero** y, adicionalmente, **descubrir segmentos o "tipos de canción"** que no están explícitos en
los metadatos originales, pero que pueden ser útiles para curaduría de contenido y armado de playlists.

La variable objetivo del problema supervisado es `popularity` (escala 0–100, continua), por lo que se trata
de un **problema de regresión**. El problema no supervisado busca descubrir agrupaciones (clusters) de
canciones con perfiles sonoros similares, sin usar la etiqueta `popularity` ni `track_genre` como criterio
de agrupación.

**Preguntas de negocio:**
1. ¿Podemos estimar la popularidad de una canción a partir de sus características musicales y metadata?
2. ¿Qué tan distintas son las canciones con popularidad 0 y qué pasa con el modelo si se las incluye o se las excluye?
3. ¿Existen segmentos naturales de canciones (más allá del género autodeclarado) que agrupen perfiles
   sonoros similares y que sean útiles para la curaduría de contenido?

*(Encuadre propuesto para esta evaluación; no corresponde a funcionalidades reales de Spotify.)*

## 2. Objetivos del proyecto

**Objetivo general:** desarrollar y comparar modelos de Machine Learning supervisados que permitan predecir
la popularidad de una canción, y aplicar una técnica no supervisada que permita identificar segmentos
relevantes dentro del catálogo, usando el dataset Spotify Tracks.

**Objetivos específicos:**
1. Preparar los datos para el modelamiento (limpieza, codificación de variables categóricas, separación
   train/test sin fuga de información).
2. Caracterizar las canciones con popularidad 0 y medir el efecto de incluirlas o excluirlas del modelo.
3. Implementar y comparar al menos dos modelos de aprendizaje supervisado de regresión.
4. Implementar una técnica de aprendizaje no supervisado (clustering) para descubrir patrones/segmentos.
5. Evaluar el desempeño de los modelos supervisados con métricas apropiadas a un problema de regresión.
6. Interpretar los resultados obtenidos y traducirlos en recomendaciones para el contexto de negocio.
7. Documentar el proceso completo siguiendo la metodología CRISP-DM.

## 3. Definición de KPIs

**KPIs técnicos**

| KPI | Definición | Propósito |
|---|---|---|
| MAE | Error absoluto medio entre popularidad real y predicha | Medir el error promedio de predicción, en las mismas unidades que `popularity` |
| RMSE | Raíz del error cuadrático medio | Penalizar con más fuerza los errores de predicción grandes |
| R² | Proporción de la varianza de `popularity` explicada por el modelo | Evaluar la capacidad explicativa global del modelo |
| Inercia (WCSS) | Suma de distancias al cuadrado dentro de cada cluster | Guiar la elección del número de clusters K (método del codo) |
| Silhouette score | Cohesión interna vs. separación entre clusters | Validar cuantitativamente la calidad/elección de K |

**KPIs de negocio** (entendibles sin conocimientos técnicos; archivo `output/Modelos/kpis_negocio.csv`)

| KPI | Valor | Qué significa |
|---|---|---|
| Canciones con popularidad 0 | 16.020 en el CSV original (14,1%); 5.425 tras limpiar (6,7%) | Cuánto del catálogo no tiene popularidad registrada |
| Datos que se descartan al entrenar sin ceros | 4.348 canciones (6,7% del entrenamiento) | Costo de excluir los ceros |
| Error promedio del modelo (MAE, Random Forest) | 10,3 puntos (con ceros) · 8,7 puntos (sin ceros, sobre canciones con popularidad > 0) | Cuántos puntos de popularidad se equivoca, en una escala 0–100 |
| Canciones predichas con error ≤ 10 puntos | 64% (con ceros) · 68% (sin ceros) | De cada 100 canciones, cuántas se predicen con error pequeño |
| Éxitos capturados al priorizar el 20% mejor puntuado | 54% (con ceros) · 60% (sin ceros); el azar daría 20% | Cuánto mejor que elegir al azar es usar el modelo para ordenar canciones |

## 4. Descripción de las fuentes de datos

- **Fuente:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle), construido a partir de la Web API de Spotify.
- **Volumen:** 114.000 registros × 21 columnas — 1.000 canciones por cada uno de los 114 géneros musicales presentes.
- **Variables principales:** identificación (`track_id`, `artists`, `album_name`, `track_name`), variable
  objetivo (`popularity`), duración (`duration_ms`), contenido explícito (`explicit`), 12 características de
  audio y producción (`danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`,
  `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`) y `track_genre`.
- **Calidad general:** solo 1 fila con metadatos faltantes (`artists`, `track_name`, `album_name`, 0,001%
  cada una); 0 filas duplicadas si se consideran todas las columnas, pero **muchas canciones repetidas**: 24.259
  `track_id` repetidos (21,3%), 6.262 filas más con audio idéntico a otra y 2.547 con el mismo título y artista
  (sección 5.1). Además, **16.020 filas (14,1%) tienen popularidad 0**.

## 5. Preparación y análisis exploratorio de los datos (EDA)

### 5.1 Duplicados y decisión de preparación

El CSV repite la misma canción de tres maneras. Cada una se limpia en un paso, conservando siempre la primera fila:

| Paso | Filas que quedan | Ceros | % de ceros |
|---|---|---|---|
| Dataset original | 114.000 | 16.020 | 14,1% |
| Sin `track_id` repetido (misma canción en varios géneros) | 89.741 | 9.448 | 10,5% |
| Sin audio idéntico con otro `track_id` (misma grabación en otro álbum/single) | 83.479 | 5.598 | 6,7% |
| Sin título + artista repetido (otra versión de la canción) | **80.932** | **5.425** | **6,7%** |

Se deja **una sola fila por canción antes de dividir train/test**, lo que elimina por construcción la fuga de
datos: una misma canción no puede quedar a la vez en train y en test. La primera versión de este proyecto
solo deduplicaba por `track_id` (89.741 filas); al revisar el modelo se midió que el 9,6% del test tenía en
train una canción con audio idéntico (con otro `track_id`), lo que inflaba el desempeño. Esta decisión difiere
del EP1, donde se conservaban las canciones multi-género y solo se eliminaban 450 duplicados reales.

**Costo de la decisión:** cada canción queda asociada al género de su primera aparición (113 de los 114 géneros
conservan canciones) y se descarta el 29% de las filas. Las conclusiones valen para "canciones únicas". Detalle
en `1_EDA.ipynb`, secciones 3.2 y 4.

**Hallazgo sobre los ceros:** las canciones repetidas tienen una proporción de ceros mucho mayor que las no
repetidas (**39,9% vs. 5,2%**), y dos tercios de los ceros originales (10.595 de 16.020) desaparecen con los
duplicados. Esto sugiere que muchos ceros son copias de canciones ya publicadas (reediciones, recopilatorios).

### 5.2 Distribución de la variable objetivo: los ceros

Con las 80.932 canciones únicas, `popularity` tiene media 34,6, mediana 35 y desviación estándar 19,4. Las
**5.425 canciones con popularidad 0 (6,7%)** forman una barra aislada, desconectada del resto de la
distribución (que parte en 1 y tiene dos jorobas, hacia 20 y hacia 45). Sin los ceros (75.507 canciones) la
media sube a 37,1, la mediana a 37 y la desviación baja a 17,7. A partir de aquí se trabaja con dos vistas:
**con ceros** (80.932) y **sin ceros** (75.507). Gráfico: `output/Graficos/distribucion_popularidad.png`.

### 5.3 ¿Cómo son las canciones con popularidad 0?

- **Audio:** las diferencias son pequeñas. Los ceros son algo menos enérgicos (diferencia estandarizada
  d = −0,27; energía media 0,57 vs. 0,64), más acústicos (d = +0,26) y más silenciosos (d = −0,22; −9,8 dB vs.
  −8,5 dB). El resto de las variables casi no cambia (|d| ≤ 0,17). Los diagramas de cajas muestran lo mismo:
  las cajas de ambos grupos se solapan casi por completo. Gráficos: `ceros_vs_resto_audio.png`,
  `cajas_audio_ceros.png`, `ceros_diferencia_estandarizada.png`.
- **No se pueden predecir con el audio:** un clasificador para adivinar si una canción es un cero obtiene un
  AUC de 0,68 usando solo audio (0,5 = azar) y de 0,89 al agregar el género.
- **Se concentran en pocos géneros:** 13 géneros tienen 20% o más de ceros (iranian 65%, romance 61%, latin 55%,
  jazz 52%, country 42%), 32 géneros tienen menos de 1% y 4 no tienen ninguno. Los 15 géneros con más ceros
  reúnen el 58% de todos los ceros. Gráficos: `ceros_por_genero.png`, `cajas_popularidad_genero.png`.
- **Efecto en la popularidad por género:** la media cambia mucho con y sin ceros (soul 40,7 → 59,6; electro
  46,0 → 61,0; dance 38,2 → 55,9). Gráfico: `popularidad_por_genero.png`.

### 5.4 Relación entre variables musicales y popularidad

Las correlaciones lineales con `popularity` son débiles **con y sin ceros**: ninguna variable pasa de
|r| ≈ 0,20. La más fuerte es `instrumentalness` (−0,17 con ceros; −0,20 sin ceros). Quitar los ceros refuerza
algunas correlaciones (speechiness −0,065 → −0,089; energy −0,002 → −0,043) pero no cambia el panorama. El
hallazgo del EP1 se mantiene: el sonido por sí solo se asocia poco con la popularidad. Gráfico:
`correlacion_popularidad.png`.

### 5.5 Preparación aplicada antes del modelamiento

1. Eliminación de columnas no predictivas (`Unnamed: 0`, `track_id`, `track_name`, `album_name`, `artists`).
2. Deduplicado a una fila por canción en tres pasos (sección 5.1).
3. División train/test (80/20, `random_state=42`). La marca `es_cero` se calcula solo dentro de los notebooks y
   **no se guarda en los CSV** (sería una copia de la variable objetivo).
4. Codificación de `track_genre` mediante One-Hot Encoding (113 columnas), ajustado **solo con datos de
   entrenamiento** dentro de un `Pipeline` de scikit-learn (sin fuga de información hacia test).
5. Imputación de valores faltantes (mediana para numéricas, moda para categóricas) dentro del mismo pipeline.

Resultado: `Data/Processed/Train/train.csv` (64.745 filas; 4.348 ceros, 6,72%) y `Data/Processed/Test/test.csv`
(16.187 filas; 1.077 ceros, 6,65%), 16 columnas cada uno (15 variables + `popularity`).

## 6. Modelamiento

### 6.1 Modelos supervisados: con ceros y sin ceros

Se entrenan dos modelos de regresión, cada uno **con** y **sin** las canciones de popularidad 0 en train, y se
evalúan en cuatro escenarios: **A** (entrenado con ceros, test completo, 16.187 canciones), **B** (entrenado
con ceros, test sin ceros, 15.110), **C** (entrenado sin ceros, test sin ceros) y **D** (entrenado sin ceros,
test completo). La comparación justa del efecto de excluir los ceros es **B vs. C**.

| Modelo | Escenario | MAE | RMSE | R² |
|---|---|---|---|---|
| Regresión Lineal | A · con ceros, test completo | 10,58 | 15,06 | 0,396 |
| Regresión Lineal | B · con ceros, test sin ceros | 9,40 | 13,09 | 0,450 |
| Regresión Lineal | C · sin ceros, test sin ceros | 9,10 | 12,54 | 0,495 |
| Regresión Lineal | D · sin ceros, test completo | 10,81 | 15,75 | 0,339 |
| Random Forest | A · con ceros, test completo | 10,31 | 14,71 | 0,423 |
| Random Forest | B · con ceros, test sin ceros | 9,21 | 12,85 | 0,469 |
| Random Forest | **C · sin ceros, test sin ceros** | **8,74** | **12,18** | **0,523** |
| Random Forest | D · sin ceros, test completo | 10,45 | 15,36 | 0,372 |

- **Regresión Lineal** (línea base, con variables numéricas estandarizadas): R² 0,40 con ceros y 0,50 sin ceros. Casi iguala al Random Forest porque el efecto principal, el del género, es aditivo. *Nota de corrección:* sin estandarizar, `duration_ms` (≈ 200.000) frente a columnas 0/1 deja el sistema mal condicionado y el ajuste no converge (R² ≈ 0,01); esa versión inicial subestimaba la línea base.
- **Random Forest** (`n_estimators=100`, `max_depth=None`, `min_samples_leaf=10`, `max_features=0.33`): suma relaciones no lineales e interacciones (p. ej. género × audio); su ventaja sobre la lineal es pequeña (≈ 0,03 de R²). **Entrenar sin ceros mejora** el R² de 0,469 a 0,523 y el error de 9,2 a 8,7 puntos sobre las mismas canciones (B vs. C), pero **usado sobre canciones que pueden ser cero (D) rinde peor** que el modelo con ceros (R² 0,372 vs. 0,423): nunca aprendió que existe el 0. La Regresión Lineal muestra el mismo patrón (B 0,450 → C 0,495; D 0,339).

Gráficos: `comparacion_modelos.png`, `predicciones_random_forest.png`.

### 6.2 Gráfico real vs. predicho y la línea horizontal

En la versión anterior del Random Forest (`max_depth=15`, `min_samples_leaf=2`) el gráfico real vs. predicho
mostraba una línea horizontal. El diagnóstico (`1_EDA`/`2_Modelizacion`, sección 2.5; gráfico
`diagnostico_linea_horizontal.png`) mostró que era un defecto del modelo, no del dataset:

- El género está codificado en 113 columnas 0/1 y cada árbol, con solo 15 niveles de profundidad, **gasta sus
  niveles separando géneros uno por uno**; las canciones de géneros "intermedios" llegan al nivel 15 sin haber
  usado el audio y reciben un valor casi constante.
- Evidencia: el 37% de las canciones del test recibía ≈ 39,0; todas las canciones de la línea terminaban en el
  nivel 15 y el 80% de las divisiones de su camino eran por género; el género acaparaba el 75% de la importancia.
- **Corrección:** sin límite de profundidad y con `max_features=0.33`. La línea desaparece (el valor más
  repetido concentra 0,6%) y el R² sube de 0,29 a 0,42 (MAE de 12,5 a 10,3).

**Lectura del gráfico corregido:** el modelo acierta mejor en la zona central (error medio de 6,1 puntos en
canciones con popularidad 21–40) y se aplana hacia el promedio en los extremos: sobrepredice las poco populares
y subpredice las muy populares (MAE de 21,0 en el tramo 61–80 y de 40,8 en el tramo 81–100). Las canciones con
popularidad 0 quedan en una columna vertical (MAE de 25,8): el audio no las distingue. Entrenar sin ceros
mejora los tramos altos (61–80: 21,0 → 16,4; 81–100: 40,8 → 31,9) pero empeora los ceros (25,8 → 34,5).

### 6.3 ¿Cuánto aporta el género?

Sin la columna de género (solo audio), el Random Forest obtiene **R² = 0,16** (MAE 14,0) frente a **0,42**
(MAE 10,3) con género, y la Regresión Lineal **0,06** frente a **0,40**. En el Random Forest final el género suma
el 57% de la importancia. El sonido por sí solo explica poco de la popularidad; lo que más ayuda es saber en qué
género está la canción. Que la lineal llegue casi al nivel del Random Forest indica que el efecto del género es
sobre todo aditivo; el Random Forest saca más del audio (0,16 vs. 0,06), pero su aporte total es pequeño.

### 6.4 Variante multi-hot: conservar todos los géneros de cada canción

En la versión principal cada canción conserva el género de su primera aparición. La variante propuesta por el
profesor junta, para cada canción única, todos los géneros en los que aparecía (una columna 0/1 por género;
`clean_dataset_multihot` en `src/preprocess.py`), con las mismas 80.932 canciones y el mismo split.

- El 16,5% de las canciones aparece en más de un género. A más géneros, más ceros: 5,3% de ceros con un género,
  10,2% con dos, 19,8% con tres y 31,6% con cuatro o más.
- **Random Forest:** mejora en los cuatro escenarios, de forma pequeña pero consistente: R² 0,423 → **0,451** (A),
  0,469 → **0,489** (B), 0,523 → **0,540** (C) y 0,372 → **0,385** (D); el error baja entre 0,2 y 0,3 puntos.
- **Regresión Lineal:** efecto mixto (0,396 → 0,408 en A y 0,339 → 0,357 en D; 0,450 → 0,437 en B y 0,495 → 0,490 en C).
- **Advertencia:** el multi-hot incluye de forma implícita *cuántos géneros* tiene la canción, algo que depende de
  cómo se armó el dataset y no se conoce al lanzar una canción nueva; parte de la mejora podría venir de ahí.
- **Decisión:** la versión principal sigue usando un género (consistencia con `Data/Processed` y los modelos
  guardados); el multi-hot se recomienda como mejora si se rehace el proyecto. Detalle en `2_Modelizacion.ipynb`, sección 2.9
  (`output/Tablas/comparacion_multihot.csv`, `output/Graficos/comparacion_multihot.png`).

### 6.5 Aprendizaje no supervisado: K-Means

Se aplicó K-Means sobre 11 variables de audio estandarizadas (`StandardScaler`), usando el conjunto completo
(train + test, 80.932 canciones), ya que el clustering no usa `popularity` ni `track_genre`. Por eso los
clusters son los mismos con y sin ceros; lo que cambia es la lectura de popularidad de cada uno.

- **Elección de K:** se probaron valores de K entre 2 y 8 con el método del codo (sin quiebre nítido) y se validó
  con *silhouette score* (muestra de 10.000 canciones): K=3 → **0,257 (máximo)**, K=2 → 0,229, K=8 → 0,184, K=7 → 0,181,
  K=6 → 0,172, K=4 → 0,162 y **K=5 → 0,161**. K=5 **no es el óptimo estadístico**; se eligió por
  interpretabilidad de negocio (perfiles distinguibles para curaduría) y se declara como una decisión de
  compromiso. La contrapartida es que los clusters se solapan (silhouette 0,16). Gráfico: `metodo_codo.png`.
- **Perfil de los 5 clusters:**

| Cluster | Canciones | Perfil sonoro | Géneros predominantes | Popularidad media (con / sin ceros) | % ceros |
|---|---|---|---|---|---|
| 0 | 27.186 | Bailable y alegre (bailabilidad 0,67, energía 0,74) | forro, salsa, kids, afrobeat, chicago-house | 35,7 / 38,2 | 6,6% |
| 1 | 22.222 | Intenso y oscuro (energía ≈ 0,8, valence 0,30, tempo 135) | black-metal, grindcore, drum-and-bass, heavy-metal, hardstyle | 33,2 / 34,8 | 4,5% |
| 2 | 17.613 | Acústico y suave (acousticness 0,68, energía 0,39) | tango, cantopop, romance, honky-tonk, acoustic | 35,3 / 38,2 | 7,6% |
| 3 | 7.246 | Hablado / explícito (95,5% explícito, speechiness 0,25) | comedy, emo, j-dance, hardcore, sad | **38,0 / 40,2** | 5,5% |
| 4 | 6.665 | Ambient / instrumental (loudness −21 dB, instrumentalness 0,78) | sleep, new-age, ambient, classical, iranian | 29,2 / 33,7 | **13,3%** |

El cluster 3 (explícito/hablado) tiene la mayor popularidad media y el cluster 4 (ambient) la menor, a la vez
que la mayor proporción de ceros. Al quitar los ceros, la diferencia entre el cluster más y menos popular se
achica de 8,8 a 6,5 puntos. Los ceros se distribuyen por todo el mapa PCA (no forman una región propia).
Gráficos: `clusters_pca.png`, `clusters_ceros.png`, `cajas_popularidad_cluster.png`.

## 7. Evaluación e interpretación de resultados (IE8)

- **Desempeño:** el Random Forest explica entre el 42% (con ceros) y el 52% (sin ceros) de la variabilidad de la
  popularidad, con un error medio de 8,7 a 10,3 puntos; es un resultado moderado. La Regresión Lineal queda
  apenas por debajo (R² 0,40 con ceros, 0,50 sin ceros), así que la ventaja del Random Forest es pequeña (≈ 0,03);
  la lineal tiene la ventaja de ser más simple de explicar. Casi todo el poder viene del género: solo con audio
  el R² es 0,16 (Random Forest) y 0,06 (lineal). Consistente con el EDA, el sonido por sí solo explica poco.
- **Para qué sirve (y para qué no):** no es un predictor de la popularidad exacta (64% de las canciones con error
  ≤ 10 puntos; error de 41 puntos en el tramo 81–100; no anticipa ceros: AUC 0,68). Sí sirve para **priorizar**:
  al revisar primero el 20% mejor puntuado se captura el **54%** de los éxitos reales con el modelo con ceros
  (2,7 veces el azar) y el **60%** con el modelo sin ceros (3,0 veces el azar). Ejemplo: un equipo editorial que
  puede revisar 1.000 de 5.000 canciones encontraría ~540–600 de los 1.000 éxitos en lugar de ~200 al azar.
  Gráfico: `uso_modelo_ganancia.png`.
- **Cómo lo usaría Spotify:** (1) curaduría editorial: elegir qué canciones escuchar primero para armar playlists;
  (2) cola de revisión para descubrimiento de catálogo; (3) combinarlo con los segmentos de K-Means para armar
  playlists por perfil sonoro. Propuesta no implementada: un esquema en dos etapas (detectar ceros por reglas y
  aplicar luego el modelo sin ceros). No se midió impacto en retención ni ingresos.
- **Limitaciones de excluir los ceros:** se pierde el 6,7% del entrenamiento; el modelo sin ceros cae de R² 0,52
  a 0,37 si luego aparecen ceros reales (D); las conclusiones valen solo para canciones con popularidad > 0; y no
  se sabe si un 0 significa "nadie la escuchó" o "dato no registrado".
- **Valor del clustering:** la segmentación entrega 5 perfiles sonoros interpretables y accionables para curaduría
  de playlists, independientemente de la precisión del modelo supervisado.

## 8. Sesgos, ética y privacidad

- **Sesgos:** muestreo artificialmente balanceado por género (1.000 canciones c/u, no representa la proporción
  real del catálogo); el deduplicado a una fila por canción (sección 5.1) asocia cada canción a un solo género
  (el primero en el CSV; la variante multi-hot de la sección 6.4 conserva todos); el 0 mezcla posiblemente canciones sin escuchas con datos no registrados y se concentra
  en pocos géneros, por lo que excluirlo puede sesgar la lectura de esos géneros; el split aleatorio por canción
  puede dejar canciones del mismo artista en train y test (no se evaluó una división por artista).
- **Ética:** curar contenido según clusters o popularidad predicha podría marginar artistas emergentes o géneros
  de nicho si se usa sin supervisión humana; el cluster de contenido explícito no debe interpretarse como "el
  contenido explícito causa popularidad" (correlación, no causalidad) ni usarse para decisiones de censura o
  promoción automática.
- **Privacidad:** el dataset no contiene datos personales de oyentes (sin IDs de usuario, ubicación ni historial
  de escucha), solo metadatos públicos de canciones/artistas. Uso restringido al ámbito académico.

## 9. Metodología (CRISP-DM)

| Fase | Estado en este proyecto |
|---|---|
| 1. Comprensión del negocio | Completado — problema, objetivos y KPIs técnicos y de negocio (secciones 1-3) |
| 2. Comprensión de los datos | Completado — fuente, calidad, duplicados y análisis de los ceros (`1_EDA.ipynb`, secciones 4-5) |
| 3. Preparación de los datos | Completado — deduplicado en tres pasos, codificación, split train/test sin fuga (sección 5.5) |
| 4. Modelado | Completado — 2 modelos supervisados (con y sin ceros), variante multi-hot y K-Means (sección 6) |
| 5. Evaluación | Completado — MAE/RMSE/R², KPIs de negocio, silhouette, diagnóstico de la línea horizontal, interpretación (secciones 6-7) |
| 6. Despliegue | Fuera del alcance; se propone un esquema en dos etapas como siguiente paso |

## 10. Conclusiones

El Random Forest supera a la Regresión Lineal, pero por poco (R² 0,42 vs. 0,40 con ceros): casi todo el poder
predictivo viene del género (solo con audio: 0,16 y 0,06). Excluir los ceros mejora el rendimiento sobre canciones
con popularidad > 0 (Random Forest: R² 0,47 → 0,52) pero el modelo pierde capacidad si después aparecen ceros
reales (R² 0,37), por lo que la decisión de incluirlos o no debe ir acompañada de sus limitaciones. Conservar
todos los géneros de cada canción (multi-hot) mejora un poco el Random Forest (R² 0,45 con ceros; 0,54 sin ceros). Los ceros no
se pueden predecir con el audio, se concentran en pocos géneros y en canciones repetidas. La línea horizontal del
gráfico original era un defecto del modelo inicial (profundidad limitada con género en *one-hot*), que se diagnosticó y
corrigió. El modelo sirve para priorizar canciones (captura 54–60% de los éxitos revisando el 20% mejor puntuado) y no
para predecir la popularidad exacta; el K-Means entrega cinco perfiles sonoros útiles para curaduría, aunque con
separación estadística débil (K=5 se elige por interpretabilidad; K=3 tiene mejor silhouette).

**Próximos pasos:** adoptar la variante multi-hot, esquema en dos etapas para los ceros, variables externas al audio
(artista, redes, playlists), validación con una división por artista y modelos de boosting.

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
│   ├── Graficos/ (distribución de popularidad, ceros vs resto, diagramas de cajas, correlaciones,
│   │              comparación de modelos y de multi-hot, real vs predicho, diagnóstico de la línea,
│   │              codo/silhouette, clusters, ganancia)
│   ├── Modelos/  (metricas_modelos.csv, kpis_negocio.csv, perfil_clusters.csv, silhouette_scores.csv,
│   │              generos_por_cluster.csv, 4 archivos .joblib: regresion_lineal y random_forest,
│   │              con_ceros y sin_ceros)
│   └── Tablas/   (dedup_pasos.csv, ceros_*.csv, correlaciones_popularidad.csv, error_por_tramo.csv, etc.)
├── Docs/
└── README.md
```
