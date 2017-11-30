"""
Insert values inside tables:
 * pv_1
 * wind_power_1
"""
import numpy as np
import pandas as pd
import pymysql

from sqlalchemy import create_engine

torre_me = '../torrePozoIzquierdo_JUL_AGO_2004/torreME.txt'
station_ms = '../torrePozoIzquierdo_JUL_AGO_2004/torreMS.txt'

# **** Torre ME
# 1.- Load the file
def insert_torre_me():
    # load df
    df_torre_me = pd.read_csv(torre_me, skiprows=12,
                              delim_whitespace=True,
                              header=None,
                              names=['measure_date', 'measure_time', 'direccion_viento_40_mtos',
                                     'viento_10_mtos', 'viento_20_mtos', 'viento_40_mtos'])
    # insert table id
    df_torre_me['id'] = 1

    # 2.- Insert in mySQL
    # create engine
    engine =   create_engine("mysql+mysqldb://root:001975@0.0.0.0/landing_zone")
    # insert df into mysql
    df_torre_me.to_sql('wind_power_1',
                       engine,
                       if_exists='replace')
    print('load data inside mysql')

# *** Torre MS
def insert_torre_ms():
    # 1.- load df
    df_station_ms = pd.read_csv(station_ms,
                                skiprows=17,
                                delim_whitespace=True,
                                header=None,
                                names=[
                                    'measure_date',
                                    'measure_time',
                                    'humedad_relativa',
                                    'irradiancia_difusa_horizontal',
                                    'irradiancia_global_horizontal',
                                    'temperatura_ambiente',
                                    'irradiancia_ultravioleta',
                                    'presion_barometrica',
                                    'pluviometria'])
    # insert table id
    df_station_ms['id']=1
    # 2.- insert into mysql
    # create engine
    engine =   create_engine("mysql+mysqldb://root:001975@0.0.0.0/landing_zone")
    # insert df into mysql
    df_station_ms.to_sql('pv_1',
                         engine,
                         if_exists='replace')
    print('load data inside mysql')

if __name__=="__main__":
    insert_torre_me()
    insert_torre_ms()
