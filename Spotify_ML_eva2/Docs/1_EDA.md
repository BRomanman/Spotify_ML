# EDA

La exploracion revisa dimensiones, tipos, estadisticas, valores faltantes, duplicados, distribucion de popularidad, variables musicales, popularidad por genero y correlaciones.

La preparacion elimina `Unnamed: 0`, `track_id`, `track_name`, `album_name` y `artists`; elimina registros repetidos por `track_id` (114.000 -> 89.741 filas); conserva `popularity` como objetivo y `track_genre` como variable categorica.

El criterio de duplicados (una fila por cancion, en vez del criterio `track_id`+`track_genre` usado en la EP1) es intencional: evita que la misma cancion quede en train y en test bajo distinto genero. El detalle esta en la seccion 8.1 de `1_EDA.ipynb`.
