#!/usr/bin/env python
# coding: utf-8

# In[8]:


import random
import pandas as pd
import streamlit as st

# Cargar el contenido del archivo CSS
#with open('styles.css', 'r') as file:
#    css_code = file.read()

#st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)

import base64


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()



def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        font-family: sans-serif;
        font-size: 24px;  
        font-weight: bold;  
        color: white; 
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('C:/Users/USUARIO/Desktop/IRONHACK/repos/AgeofSigmar/Images/wallpaper_streamlit.png')

# Función para lanzar un dado
def roll(min, max):
    return random.randint(min, max)

# Función para simular una prueba de valentía durante la fase de choque de batalla
def bravery_test(unit):
    bravery_value = unit['Bravery']
    roll_result = roll(1, 6)
    test_result = roll_result + unit['Wounds']
    return test_result <= bravery_value, test_result

# Función para simular la batalla entre dos ejércitos
def battle(army1, army2):
    log = []
    current = army1.copy()
    enemy = army2.copy()

    while True:
        for i in range(current['Attacks']):
            hitRand = roll(1, 6)
            woundRand = roll(1, 6)
            saveRand = roll(1, 6)

            if 'Rend' in current and current['Rend'] > 0:
                saveRand -= current['Rend']

            attack_number = i + 1
            attack_description = f"{current['Unit']} ({current['Wounds']}) is making the {attack_number} attack against {enemy['Unit']}."

            if hitRand >= current['To Hit'] and woundRand >= current['To Wound'] and saveRand >= enemy['Save']:
                damage_caused = current['Damage']
                log.append(f"{attack_description} The attack caused {damage_caused} points of damage.")
                enemy['Wounds'] -= damage_caused
            else:
                log.append(f"{attack_description} The attack missed.")

        if enemy['Wounds'] <= 0:
            log.append(f"{enemy['Unit']} has been defeated.")
            break

        if 'Bravery' in enemy and enemy['Wounds'] > 0:
            bravery_result, test_result = bravery_test(enemy)
            if bravery_result:
                log.append("Your unit has bravely withstood the attack.")
            else:
                log.append("Your unit has been morally overwhelmed and flees in search of safety.")
                log.append(f"{current['Unit']} has won the battle as {enemy['Unit']} flees.")
                return log

        current, enemy = enemy, current

    return log

# Función para ejecutar la batalla y mostrar el resultado
def run_battle(army1, army2):
    return battle(army1, army2)

st.title("Age of Sigmar Battle")

# Cargar los datos y crear las listas de ejércitos
units = pd.read_excel("AgeofSigmardata.xlsx", sheet_name="Units' Warscroll and Price")
weapons = pd.read_excel("AgeofSigmardata.xlsx", sheet_name="Units' Weapons")
sigmar = pd.merge(units, weapons, on='Unit')

army_list = []
for index, row in sigmar.iterrows():
    army = {
        'Unit': row['Unit'],
        'Faction': row['Faction'],
        'Move': row['Move'],
        'Wounds': row['Wounds'],
        'Save': row['Save'],
        'Bravery': row['Bravery'],
        'Price': row['Price'],
        'Points': row['Points'],
        'Main weapon': row['Main weapon'],
        'Kind': row['Kind'],
        'Range': row['Range'],
        'Attacks': row['Attacks'],
        'To Hit': row['To Hit'],
        'To Wound': row['To Wound'],
        'Rend': row['Rend'],
        'Damage': row['Damage'],
    }
    army_list.append(army)

# Crear una barra lateral en Streamlit para la selección de ejércitos
st.sidebar.header("Army Selection")
army1_name = st.sidebar.selectbox("Select Army 1", [army['Unit'] for army in army_list])
army2_name = st.sidebar.selectbox("Select Army 2", [army['Unit'] for army in army_list])

# Encontrar los ejércitos seleccionados por nombre
army1 = next(army for army in army_list if army['Unit'] == army1_name)
army2 = next(army for army in army_list if army['Unit'] == army2_name)

# Botón para ejecutar la batalla
if st.button("Run Battle"):
    log = run_battle(army1, army2)
    st.write("Battle Simulation Result:")
    for entry in log:
        st.write(entry)



# In[16]:


# get_ipython().system('jupyter nbconvert --to script battle_simulation.ipynb')


# In[14]:


# Ruta de acceso al archivo de la aplicación Streamlit
#app_file_path = "C:\\Users\\USUARIO\\Desktop\\IRONHACK\\repos\\AgeofSigmar\\battle_simulation.py"

# Ejecutar la aplicación Streamlit desde Jupyter Notebook
#subprocess.run(["C:\\Users\\USUARIO\\anaconda3\\envs\\myenv_geopandas\\Scripts\\streamlit.cmd", "run", app_file_path])

