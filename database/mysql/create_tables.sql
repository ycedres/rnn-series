
CREATE DATABASE IF NOT EXISTS landing_zone;

USE landing_zone;

/* wind power metadata */
DROP TABLE IF EXISTS metadata_wind_power;

CREATE TABLE IF NOT EXISTS metadata_wind_power(
id int,
numero_de_medidas int,
nombre_estacion varchar(255),
unidades_direccion_viento varchar(255),
unidades_velocidad_viento varchar(255)
);

/* wind power values */
DROP TABLE IF EXISTS wind_power_1;

CREATE TABLE IF NOT EXISTS wind_power_1(
  id int,
  measure_date varchar(255),
  measure_time varchar(255),
  direccion_viento_40_mtos float(2,2),
  viento_10_mtos float(2,2),
  viento_20_mtos float(2,2),
  viento_40_mtos float(2,2)
);

/* PV metadata */
DROP TABLE IF EXISTS metadata_pv;

CREATE TABLE IF NOT EXISTS metadata_pv(
id int,
nombre_estacion varchar(255),
numero_de_medidas int,
humedad_relativa varchar(255),
irradiancia_difusa_horizontal varchar(255),
irradiancia_global_horizontal varchar(255),
temperatura_ambiente varchar(255),
irradiancia_ultravioleta varchar(255),
presion_barometrica varchar(255),
pluviometria varchar(255)
);

/* PV values */
DROP TABLE IF EXISTS pv_1;

CREATE TABLE IF NOT EXISTS pv_1(
  id int,
  measure_date varchar(255),
  measure_time varchar(255),
  humedad_relativa float(2,2),
  irradiancia_difusa_horizontal float(2,2),
  irradiancia_global_horizontal float(2,2),
  temperatura_ambiente float(2,2),
  irradiancia_ultravioleta float(2,2),
  presion_barometrica float(3,2),
  pluviometria float(2,2)
);
