import streamlit as st
import random
import pandas as pd

# Función para tirar un dado
def roll(min, max):
    return random.randint(min, max)

# Función para realizar una prueba de Bravery durante la fase de Battle Shock
def bravery_test(unit, wounds_suffered):
    bravery_value = unit['Bravery']
    roll_result = roll(1, 6)
    test_result = roll_result + wounds_suffered
    
    if test_result <= bravery_value:
        return True, test_result
    else:
        return False, test_result

# Función para simular la batalla entre dos ejércitos
def battle(army1, army2):
    log = []

    # Tirar dados para determinar el turno inicial de cada ejército
    while True:
        roll1 = roll(1, 6)
        roll2 = roll(1, 6)

        if roll1 > roll2:
            current, enemy = army1, army2
            log.append(f"{army1['Faction']} won the dice roll and starts first.")
            break
        elif roll2 > roll1:
            current, enemy = army2, army1
            log.append(f"{army2['Faction']} won the dice roll and starts first.")
            break
        else:
            log.append("It's a tie! Repeating the dice roll to determine the starting turn.")

    while True:
        for i in range(current['Attacks']):
            hitRand = roll(1, 6)  # Dado de 6 caras
            woundRand = roll(1, 6)  # Dado de 6 caras
            saveRand = roll(1, 6)  # Dado de 6 caras

            # Aplicar el valor de Rend si existe
            if 'Rend' in current and current['Rend'] > 0:
                saveRand -= current['Rend']

            numero_ataque = i + 1  # Calcular el número del ataque
            descripcion_ataque = f"{current['Unit']} ({current['Wounds']}) is making the {numero_ataque} attack against {enemy['Unit']}."

            if hitRand >= current['To Hit'] and woundRand >= current['To Wound'] and saveRand >= current['Save']:
                damage_caused = current['Damage']
                log.append(f"{descripcion_ataque} The attack caused {damage_caused} points of damage.")
                enemy['Wounds'] -= damage_caused
            else:
                log.append(f"{descripcion_ataque} The attack missed.")

        if enemy['Wounds'] <= 0:
            log.append(f"{enemy['Unit']} has been defeated.")
            break

        # Realizar una prueba de Bravery para el ejército enemigo durante la fase de Battle Shock
        if 'Bravery' in enemy and enemy['Wounds_Suffered'] > 0:
            resultado_bravery, resultado_prueba = bravery_test(enemy, enemy['Wounds_Suffered'])
            if resultado_bravery:
                log.append("Your unit has bravely withstood the attack.")
            else:
                log.append("Your unit has been morally overwhelmed and flees in search of safety.")

        # Cambiar de turno
        current, enemy = enemy, current

    return log

# Cargar los datos de tus ejércitos desde un archivo Excel
units = pd.read_excel("AgeofSigmardata.xlsx", sheet_name="Units' Warscroll and Price")
weapons = pd.read_excel("AgeofSigmardata.xlsx", sheet_name="Units' Weapons")
sigmar = pd.merge(units, weapons, on='Unit')

# Crear instancias de ejércitos desde el DataFrame 'sigmar'
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
        'Rend': row['Rend'],  # Asegúrate de que esta columna exista en tu DataFrame
        'Damage': row['Damage'],
        'Wounds_Suffered': 0  # Inicialmente, no se han sufrido heridas
    }
    army_list.append(army)

# Selecciona dos ejércitos de la lista para la batalla (ajusta los índices según tus necesidades)
army1 = army_list[0]
army2 = army_list[1]

log = battle(army1, army2)

# Crear la interfaz de usuario de Streamlit
st.title("Simulación de Batalla")
st.header("Registro de Batalla")
for entry in log:
    st.write(entry)


