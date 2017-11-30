
USE landing_zone;

INSERT INTO metadata_wind_power(
  id,
  numero_de_medidas,
  nombre_estacion,
  unidades_direccion_viento,
  unidades_velocidad_viento
  )
  VALUES(1, 4, "TORRE CLIMATOLOGICA POZO", "grados", "m/sg");

INSERT INTO metadata_pv(
  id,
  nombre_estacion,
  numero_de_medidas,
  humedad_relativa,
  irradiancia_difusa_horizontal,
  irradiancia_global_horizontal,
  temperatura_ambiente,
  irradiancia_ultravioleta,
  presion_barometrica,
  pluviometria
  )
  VALUES(1, "Torre Climatológica", 7, "%", "W/m^2", "W/m^2", "ºC", "W/m^2",
         "bar", "%");
