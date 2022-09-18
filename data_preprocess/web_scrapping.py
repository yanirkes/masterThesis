from selenium import webdriver
import pandas as pd
import numpy as np
import bs4 as bs
import urllib
import time
import threading as thrd
import utils.constant_class as inf_
import html5lib
import lxml

import subprocess

s = subprocess.check_output("curl https://www.acb.com/club/plantilla/id/12/temporada_id/2003")
df = pd.read_html(io=s)

col_name = ["NUMBERP", "JUGADOR", "POSICIÓN", "NACIONALIDAD", "LICENCIA", "ALTURA", "EDAD", "TEMP."]
full_col_name = ["NUMBERP", "JUGADOR", "POSICIÓN", "NACIONALIDAD", "LICENCIA", "ALTURA", "EDAD", "TEMP.", "YEAR", "TEAM"]
years = [2003+i for i in range(0,11) ]
eng_col = [
            'numberp'
           ,  "player"
           ,  "position"
           ,  "nationality"
           ,  "license"
           ,  "hieght"
           ,  "age"
           ,  'temp.'
           ,  'year'
           ,  'team'
]

def create_df_player_par_by_year(ids, year):
    temp = pd.DataFrame(columns = full_col_name)
    for id in ids:
        df_ = pd.read_html(f'https://www.acb.com/club/plantilla-lista/id/{id}/temporada_id/{year}',
                          attrs={'class': 'roboto defecto tabla_plantilla equipo_plantilla tabla_ancho_completo mb30'})
        df_2 = []
        try:
            df_2 = pd.read_html(f'https://www.acb.com/club/plantilla-lista/id/{id}/temporada_id/{year}',
                          attrs={"class":"roboto defecto tabla_plantilla plantilla_bajas clasificacion tabla_ancho_completo"})
            print(df_2[0].columns)
            df_2[0]["ALTURA"] = np.nan
            df_2[0]["EDAD"] = np.nan
            df_2[0]["TEMP."] = np.nan
        except ValueError:
            pass
        source_2 = urllib.request.urlopen(
            f'https://www.acb.com/club/plantilla-lista/id/{id}/temporada_id/{year}').read()
        soup_2 = bs.BeautifulSoup(source_2, 'html.parser')
        team_name = soup_2.find_all("h3", class_='roboto_condensed_bold mayusculas')[0].next
        if len(df_) == 0:
            raise Exception("There is a problem creating df", df_)
        if len(df_2) != 0:
            df_.append(df_2[0])
        for i in  range(len(df_)):
            df_1 = pd.DataFrame(df_[i])
            df_1.columns = col_name
            df_1['TEAM'] = team_name
            df_1['YEAR'] = year
            temp = pd.concat([temp, df_1], axis=0).reset_index(drop=True)

    return temp

def create_table_player_par(year, data):
    source = urllib.request.urlopen(f'https://www.acb.com/club/index/temporada_id/{year}').read()
    soup = bs.BeautifulSoup(source, 'html.parser')
    # get the list of ids, each id = a team
    ids_response = soup.find_all("a", href=True, class_ = "clase_mostrar_block960 equipo_logo primer_logo")
    # create a list of ids
    ids = [i['href'].split('/')[-1] for i in ids_response]
    # Add missing ids
    ids = ids + inf_.constant.missing_ids_webscrapping
    ids = set(ids)
    data.append(create_df_player_par_by_year(ids, year))
    # data.append(create_df_player_par_by_year([11], year))

def cut_name_string(name):
    name_length = int(len(name) / 2)
    return name[0:name_length]

if __name__ == "__main__":
    # sys.setrecursionlimit(25000)
    start_time = time.time()

    data = list()
    th = []
    for ind, year in enumerate(years):
        thread = thrd.Thread(name='th%s' % ind, target=create_table_player_par(year,data ))
        thread.start()
        th.append(thread)

    for j in th:
        j.join()
    print(time.time() - start_time)
    print(data)

    full_data = pd.concat(data).reset_index(drop = True)
    full_data.loc[:, 'JUGADOR'] = full_data.loc[:, 'JUGADOR'].apply(lambda x: cut_name_string(x))
    full_data.columns = eng_col

    full_data.to_csv("player_detail_4.csv")















source = urllib.request.urlopen(f'https://www.acb.com/club/index/temporada_id/{2005}').read()
soup = bs.BeautifulSoup(source, 'html.parser')
# get the list of ids, each id = a team
ids_response = soup.find_all("a", href=True, class_ = "clase_mostrar_block960 equipo_logo primer_logo")
# create a list of ids
ids = [i['href'].split('/')[-1] for i in ids_response]


temp = pd.DataFrame(columns = full_col_name)
for id in ids:
    df_ = pd.read_html(f'https://www.acb.com/club/plantilla-lista/id/{id}/temporada_id/{2005}',
                      attrs={'class': 'roboto defecto tabla_plantilla equipo_plantilla tabla_ancho_completo mb30'})
    source_2 = urllib.request.urlopen(
        f'https://www.acb.com/club/plantilla-lista/id/{id}/temporada_id/{2005}').read()
    soup_2 = bs.BeautifulSoup(source_2, 'html.parser')
    team_name = soup_2.find_all("h3", class_='roboto_condensed_bold mayusculas')[0].next
    if len(df_) == 0:
        raise Exception("There is a problem creating df", df_)
    for i in  range(len(df_)):
        df_1 = pd.DataFrame(df_[i])
        df_1.columns = col_name
        df_1['TEAM'] = team_name
        df_1['YEAR'] = 2005
        temp = pd.concat([temp, df_1], axis=0).reset_index(drop=True)

source = urllib.request.urlopen(f'https://www.acb.com/club/plantilla-lista/id/71/temporada_id/2006').read()
soup = bs.BeautifulSoup(source, 'html.parser')


teams = ["Leche Rio", "FC Barcelona","Etosa Alicante", "Casademont Girona", "Real Madrid","DKV Joventut", "Caprabo Lleida", "Unelco Tenerife","Unicaja",
         "Forum Valladolid", "Adecco Estudiantes","Polaris World Murcia", "Jabones Pardo Fuenlabrada", "Caja San Fernando","Ricoh Manresa",
         "Pamesa Cerámica Valencia", "Auna G. Canaria","TAU Cerámica", "CB Granada", "Winterthur FC Barcelona","Lagun Aro Bilbao Basket",
         "Plus Pujol Lleida", "Pamesa Valencia","Gran Canaria", "Akasvayu Girona", "Alta Gestión Fuenlabrada","Fórum Valladolid", "Gran Canaria Grupo Dunas",
         "Leche Río","Llanera Menorca", "Polaris World CB Murcia", "ViveMenorca","Bruesa GBC", "MMT Estudiantes", "Grupo Capitol Valladolid","Grupo Begar León",
         "Kalise Gran Canaria", "Cajasol","iurbentia Bilbao Basket", "AXA FC Barcelona", "CAI Zaragoza","Alta Gestión Fuenlabrada", "CB Murcia", "Regal FC Barcelona",
         "Lagun Aro GBC", "Suzuki Manresa", "Ayuda en Acción Fuenlabrada","Club Baloncesto Murcia", "Caja Laboral", "Bizkaia Bilbao Basket","Meridiano Alicante",
         "Blancos de Rueda Valladolid", "Xacobeo Blu:Sens","Power Electronics Valencia", "ASEFA Estudiantes", "Gran Canaria 2014","Baloncesto Fuenlabrada", "Menorca Basquet",
         "Assignia Manresa","Asefa Estudiantes", "Valencia Basket", "Blusens Monbus","Banca Civica", "Lucentum Alicante", "Gescrap Bizkaia","Mad-Croc Fuenlabrada",
         "FIATC Mutua Joventut", "UCAM Murcia","FC Barcelona Regal", "CB Canarias", "UCAM Murcia CB","Valencia Basket Club", "Bàsquet Manresa",
         "Herbalife Gran Canaria","Uxue Bilbao Basket", "FIATC Joventut" ]