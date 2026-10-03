# EDA

La exploracion revisa dimensiones, tipos, estadisticas, valores faltantes, duplicados, distribucion de popularidad, variables musicales, popularidad por genero y correlaciones.

La preparacion elimina `Unnamed: 0`, `track_id`, `track_name`, `album_name` y `artists`; elimina registros repetidos por `track_id`; conserva `popularity` como objetivo y `track_genre` como variable categorica.
