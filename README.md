# Spotify ML - EVA 2

Proyecto de Machine Learning desarrollado para la evaluacion EVA 2, utilizando un dataset de canciones de Spotify.

El proyecto contempla las etapas de exploracion de datos, preprocesamiento, modelizacion supervisada y no supervisada, evaluacion de modelos y generacion de resultados.

---

## Objetivo

Analizar un conjunto de datos de canciones de Spotify para identificar patrones y relaciones entre sus caracteristicas, utilizando tecnicas de Machine Learning.

El proyecto busca aplicar un flujo completo de trabajo de Ciencia de Datos, desde la carga y exploracion de los datos hasta la construccion y evaluacion de modelos.

---

## Dataset

El dataset utilizado contiene informacion sobre canciones de Spotify y sus caracteristicas.

Entre las variables disponibles se encuentran:

- `popularity`
- `duration_ms`
- `explicit`
- `danceability`
- `energy`
- `key`
- `loudness`
- `mode`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `track_genre`

El dataset original se encuentra en:

```text
Data/Raw/Spotify_Tracks_Dataset.csv

Los datos procesados se almacenan en:

Data/Processed/Train/train.csv
Data/Processed/Test/test.csv

Estructura del proyecto

Spotify_eva2/
│
├── Data/
│   ├── Raw/
│   │   └── Spotify_Tracks_Dataset.csv
│   │
│   └── Processed/
│       ├── Train/
│       │   └── train.csv
│       └── Test/
│           └── test.csv
│
├── Notebook/
│   └── Exploracion_inicial/
│       ├── 1_EDA.ipynb
│       └── 2_Modelizacion.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── grafico.py
│   ├── metrica.py
│   ├── train.py
│   └── test.py
│
├── output/
│   ├── Graficos/
│   └── Modelos/
│
├── Docs/
│   ├── estructura_inicial.md
│   ├── 1_EDA.md
│   └── 2_modelos.md
│
├── Environment/
│   ├── main.md
│   └── requirements.txt
│
├── Informe_Tecnico_EP2_Spotify.md
│
├── .gitignore
└── README.md

El informe tecnico (`Informe_Tecnico_EP2_Spotify.md`) es el entregable principal de la EP2: documenta,
en un solo archivo Markdown reproducible, el problema de negocio, objetivos, KPIs, fuentes de datos,
preparacion/EDA, modelado (supervisado y no supervisado), evaluacion/interpretacion de resultados,
etica y la metodologia CRISP-DM aplicada.

Flujo del proyecto

El proyecto sigue el siguiente flujo:

Dataset original
      |
      v
Data/Raw
      |
      v
Preprocesamiento
      |
      v
Data/Processed
      |
      +----------+
      |          |
      v          v
    Train       Test
      |          |
      +----+-----+
           |
           v
      Modelizacion
           |
           +----------------+
           |                |
           v                v
      Modelos             Clustering
    supervisados        no supervisado
           |                |
           +-------+--------+
                   |
                   v
              Evaluacion
                   |
                   v
               Resultados

Notebooks
1. EDA

Archivo:

Notebook/Exploracion_inicial/1_EDA.ipynb

En este notebook se realiza la exploracion inicial del dataset.

Se consideran actividades como:

    Carga de datos.

    Revision de dimensiones.

    Revision de tipos de datos.

    Identificacion de valores faltantes.

    Analisis estadistico.

    Analisis de variables.

    Visualizaciones.

    Identificacion de patrones y relaciones.

El dataset tiene 24.259 `track_id` repetidos (21.3%) porque una misma cancion puede pertenecer a varios
generos. La preparacion deja una sola fila por cancion para evitar fuga de datos entre train y test; la
justificacion completa de este criterio esta documentada en la seccion 8.1 del propio notebook.

2. Modelizacion

Archivo:

Notebook/Exploracion_inicial/2_Modelizacion.ipynb

En este notebook se desarrolla la etapa de Machine Learning.

Se consideran:

    Preparacion de los datos.

    Separacion de datos de entrenamiento y prueba.

    Transformacion de variables.

    Modelos supervisados.

    Modelos no supervisados.

    Clustering.

    Reduccion de dimensionalidad.

    Evaluacion de resultados.

Se comparan 2 modelos supervisados de regresion (Regresion Lineal y Random Forest) y se aplica K-Means
como tecnica no supervisada. El numero de clusters (K=5) se valida cuantitativamente con silhouette score:
el score favorece K=3 (0.251) por sobre K=5 (0.166), pero se mantiene K=5 por interpretabilidad de negocio
(clusters mas diferenciables); el detalle de esta decision esta en el notebook y en el informe tecnico.
Los modelos entrenados quedan persistidos en `output/Modelos/*.joblib` para su reutilizacion.

Codigo fuente

La carpeta src contiene las funciones utilizadas durante el proyecto.
preprocess.py

Contiene funciones relacionadas con:

    Limpieza de datos.

    Preparacion del dataset.

    Transformacion de variables.

    Separacion de datos.

    Guardado de datasets procesados.

grafico.py

Contiene funciones utilizadas para generar graficos y visualizaciones.

Los resultados pueden almacenarse en:

output/Graficos/

metrica.py

Contiene funciones relacionadas con las metricas utilizadas para evaluar los modelos.
train.py

Contiene funciones y procesos relacionados con el entrenamiento de modelos.
test.py

Contiene funciones y procesos relacionados con la evaluacion de los modelos utilizando datos de prueba.
Resultados

Los resultados generados durante el proyecto se almacenan en:

output/
├── Graficos/
└── Modelos/

Los graficos generados se guardan en:

output/Graficos/

Los modelos entrenados se guardan en:

output/Modelos/
├── regresion_lineal.joblib
├── random_forest.joblib
├── metricas_modelos.csv
├── perfil_clusters.csv
└── silhouette_scores.csv

Tecnologias utilizadas

    Python

    Pandas

    NumPy

    Scikit-learn

    SciPy

    Matplotlib

    Seaborn

    Jupyter Notebook

    VS Code

    Git

    GitHub

Requisitos

Se utiliza Python 3.12.

Las dependencias del proyecto se encuentran en:

Environment/requirements.txt

Instalacion

Clonar el repositorio:

git clone URL_DEL_REPOSITORIO

Entrar al proyecto:

cd Spotify_eva2

Crear el entorno virtual:

python -m venv .venv

Instalar las dependencias:

.venv\Scripts\python.exe -m pip install -r Environment\requirements.txt

Ejecucion

Una vez instaladas las dependencias, abrir el proyecto en VS Code.

Seleccionar como interprete de Python:

.venv\Scripts\python.exe

Luego ejecutar los notebooks en el siguiente orden:
Paso 1

Notebook/Exploracion_inicial/1_EDA.ipynb

Paso 2

Notebook/Exploracion_inicial/2_Modelizacion.ipynb

Organizacion del proyecto

El proyecto separa los datos, notebooks, codigo fuente, resultados y documentacion para facilitar el mantenimiento y la reproduccion del trabajo.

La estructura permite diferenciar:

    Datos originales.

    Datos procesados.

    Exploracion de datos.

    Modelizacion.

    Codigo reutilizable.

    Resultados.

    Documentacion.

    Dependencias del proyecto.

Autores

Bastian Roman
Sebastian Aird
Vicente Contreras
Ignacio Gomez

Proyecto desarrollado para la asignatura de Ciencia de Datos / Machine Learning.

Duoc UC

EVA 2 - Machine Learning
