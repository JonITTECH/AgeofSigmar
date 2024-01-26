#!/usr/bin/env python
# coding: utf-8

# ### hacer un tableau interactivo, en el que al escoger la facción o la alianza te salgan cuáles son las estimaciones de precios

# # Libraries

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import streamlit as st


# # Data load

# We extract the information from the two pages of the Excel file

# In[2]:


units = pd.read_excel("AgeofSigmardata.xlsx", sheet_name="Units' Warscroll and Price")
weapons = pd.read_excel("AgeofSigmardata.xlsx", sheet_name="Units' Weapons")
sigmar =  pd.merge(units, weapons, on='Unit')


# ## Characteristics

# #### Unit Characteristics

# **Grand Alliance**: Refers to the larger faction or coalition under which various armies and factions unite.

# **Faction**: Represents a specific army or group within a Grand Alliance. Each faction has its unique characteristics, units, and abilities.

# **Move**: The distance a unit can move in inches during its movement phase.

# **Wounds**: The total number of wounds a unit can endure before being removed from play.

# **Save**: The value needed on a six-sided die to successfully save against an attack.

# **Bravery**: This statistic represents the unit's morale or courage on the battlefield. When a unit suffers wounds, it may be required to take a Bravery test during the Battle Shock phase. To determine if the unit passes the test, roll a six-sided die and add the number of wounds the unit suffered that turn. If the result is less than or equal to the unit's Bravery value, it passes the test and remains in play. If the result exceeds the Bravery value, the unit may suffer additional casualties as models flee from the battlefield.

# **Price**: The cost in terms of in-game currency (e.g., points, gold) to include the unit in your army.

# **Points**: The actual in-game points assigned to the unit, which are used to balance forces in a match.
# 

# #### Weapon Characteristics

# **Kind**: The type of weapon, specifying its characteristics and potential special rules.

# **Range**: The maximum distance in inches at which the weapon can be used effectively.

# **Attacks**: The number of attacks the weapon can make in a single round of combat.

# **To Hit**: The value needed on a six-sided die to successfully hit the target with the weapon, once you successfully hit them.

# **To Wound**: The value needed on a six-sided die to successfully wound the target with the weapon.

# **Rend**: The negative modifier applied to the opponent's save roll when the weapon inflicts damage.

# **Damage**: The amount of damage each successful hit with the weapon inflicts on the target.

# # Data Transformation

# **'To Hit'** means the value needed on a die to hit the target, so we are going to convert it to a percentage over 1 (2 decimals)

# In[3]:


sigmar['To Hit'] = round(1 - sigmar['To Hit'] / 6, 2)


# **'To Wound'** means the value needed on a die to wound the target, once you successfully hit them, so we are going to change it as we did with 'To Hit'

# In[4]:


sigmar['To Wound'] = round(1 - sigmar['To Wound'] / 6, 2)


# We want to create a column that tells us the probability of hitting the target and causing a wound: **'To Hit & Wound'**

# In[5]:


sigmar['To Hit & Wound'] = round(sigmar['To Hit'] * sigmar['To Wound'], 2)


# We would like to have a column that tells us the overall expectation of generating damage: **'Expected Total Damage'**

# In[6]:


sigmar['Expected Total Damage'] = sigmar['Damage'] * sigmar['To Hit & Wound'] * sigmar['Attacks']


# **'Rend'** is a variable that depends on the opponent's 'Save' value, subtracting from it, so we are going to make it positive without applying it to any common calculation, as the 'Save' variable has too many options

# In[7]:


sigmar['Rend'] = sigmar['Rend'].abs()


# As **'Save'** indicates the number from which you save yourself, I'm changing the column so it calculates, based on the value in 'Save', what is the probability of saving yourself

# In[8]:


sigmar['Save'] = round((7 - sigmar['Save']) / 6, 2)


# Regarding **'Bravery'**, let's alculate the probability of rolling a number less than or equal to x on a six-sided die (after adding 1 of damage). As the main 'Expected Total Damage' is 0.50, we're using 1 instead, so it's more significative for this measure.

# In[9]:


sigmar['Bravery'] =  round((sigmar['Bravery'] - 1) / 6, 2)


# Let's just fix these little things as well

# In[10]:


sigmar['Kind'] = sigmar['Kind'].str.strip()


# In[11]:


sigmar['Kind'] = sigmar['Kind'].replace('Misile weapon', 'Missile weapon')


# # Data Exploration

# In[12]:


sigmar.shape


# In[13]:


sigmar.head()


# #### Top 3 Best Weapons

# In[14]:


top3_best_weapons = sigmar.sort_values(by='Expected Total Damage', ascending=False).head(3)

top3_best_weapons


# #### Top 3 Best Units

# We can consider many factors, but the most important ones are the expected damage and the wounds they can suffer, as well as the ability to avoid attacks (save). That's why we have tested all combinations, and we would always have the same top 3 as the best weapons  
# Could we also take into account movement and range, but this is much more related to general strategies when developing combat, and we will not consider it in this analysis.

# In[15]:


top3_units = sigmar[sigmar['Expected Total Damage'] > 0.8].sort_values(by=['Wounds', 'Expected Total Damage', 'Save'], ascending=[False, False, False]).head(3)
top3_units


# Even with this filter added, the top 3 units remain the same, only changing the order of the first two.

# ### Count of Units per Faction/ Faction per Units

# In[16]:


units_by_faction_alliance = sigmar.groupby(['Grand Alliance', 'Faction']).size().reset_index(name='Unit Count')
factions_by_alliance = sigmar.groupby('Grand Alliance')['Faction'].nunique().reset_index(name='Faction Count')

units_by_faction_alliance = units_by_faction_alliance.sort_values(by='Unit Count', ascending=False)
factions_by_alliance = factions_by_alliance.sort_values(by='Faction Count', ascending=False)

print("Units per faction and alliance:")
print(units_by_faction_alliance)
print("\nFactions per alliance:")
print(factions_by_alliance)


# #### Interpretation of Units per Faction and Alliance

# The alliance with the highest representation in terms of unit count is "Order," with Sylvaneth and Free Cities having the most units (5 each)

# The Stormcast Eternals, also from the Order alliance, follow closely with 4 units

# Tzeentch from the Chaos alliance has the highest unit count among Chaos factions, with 4 units

# Legions of Nagash from the Death alliance and several other factions have only 1 unit each

# #### Interpretation of FactionS per Alliance

# The alliance with the most factions is "Order," with 7 unique factions

# Chaos follows closely with 6 unique factions

# Death has 4 unique factions, and Destruction has only 3 unique factions

# ### Mean values per Grand Alliance

# In[17]:


factions_mean =  sigmar.groupby('Faction')[['Move','Wounds','Save', 'Bravery','Price','Points','Range','Attacks','To Hit','To Wound','Rend','Damage','To Hit & Wound','Expected Total Damage']].mean()
alliances_mean = sigmar.groupby('Grand Alliance')[['Move','Wounds','Save', 'Bravery','Price','Points','Range','Attacks','To Hit','To Wound','Rend','Damage','To Hit & Wound','Expected Total Damage']].mean()

# we don't have enough data to see the st dev of the factions, so let's check the alliances
alliances_std = sigmar.groupby('Grand Alliance')[['Move','Wounds','Save','Bravery','Price','Points','Range','Attacks','To Hit','To Wound','Rend','Damage','To Hit & Wound','Expected Total Damage']].std()


# In[18]:


alliances_mean


# **Chaos** has the highest score in mobility and it boasts the best price. However, its Rend points are very low, as the amount of Attacks.

# **Death** doesn't stand out as especially at anything but for being one of the most courageous (considering they are already deceased and have little to fear).

# **Destruction** has the worst movement and a limited ability to dodge damage, but stands out remarkably in the amount of damage needed to eliminate them, the number of attacks, and the expected damage. On the other hand, they are the most expensive and have the highest points, so you may have few units in your army.
# 

# **Order** is very good at avoiding damage, they are the bravest, and they have units with the longest attack range

# ### Top 5 best Factions per each Value

# In[19]:


columns_to_find_top5 = ['Move', 'Wounds', 'Save', 'Bravery', 'Price', 'Expected Total Damage']


# In[20]:


# Dictionary to store top 5 DataFrames for each column
top5_dataframes = {}

# Find top 5 factions for each column and create DataFrames
for column in columns_to_find_top5:
    if column in factions_mean.columns:
        # Sort in reverse order if it is the 'Price' column
        if column == 'Price':
            top5_dataframes[f'Top5_{column}_Factions'] = pd.DataFrame({f'Top5_{column}_Factions': factions_mean[column].nsmallest(5).index})
        else:
            top5_dataframes[f'Top5_{column}_Factions'] = pd.DataFrame({f'Top5_{column}_Factions': factions_mean[column].nlargest(5).index})
    else:
        print(f"Column '{column}' not found in factions_mean.")

# Concatenate all DataFrames
top5_factions_df = pd.concat([pd.DataFrame(top5_dataframes[column]) for column in top5_dataframes.keys()], axis=1)

top5_factions_df


# ### Top 4 best Alliances per each value

# In[21]:


# Dictionary to store top 4 DataFrames for each column
top4_alliances_dataframes = {}

# Find top 4 factions for each column and create DataFrames in alliances_mean
for column in columns_to_find_top5:
    if column in alliances_mean.columns:
        # Ordenar inversamente si es la columna 'Price'
        if column == 'Price':
            top4_alliances_dataframes[f'Top4_{column}_Alliances'] = pd.DataFrame({f'Top4_{column}_Alliances': alliances_mean[column].nsmallest(5).index})
        else:
            top4_alliances_dataframes[f'Top4_{column}_Alliances'] = pd.DataFrame({f'Top4_{column}_Alliances': alliances_mean[column].nlargest(5).index})
    else:
        print(f"Column '{column}' not found in alliances_mean.")

# Concatenate all DataFrames
top4_alliances_df = pd.concat([pd.DataFrame(top4_alliances_dataframes[column]) for column in top4_alliances_dataframes.keys()], axis=1)

top4_alliances_df



# ### Correlations among numeric variables

# In[22]:


numerical_data = sigmar.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numerical_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Heatmap - Correlation Matrix (Numerical Variables)')
plt.show()


# In[23]:


correlation_matrix


# #### Interpretation

# Regarding the interpretation of these correlations, some are more significant than others. Let's exclude the following ones due to their obviousness:
# 
# To Hit & Wound and Expected Total Damage (0.613)
# To Hit and To Hit & Wound (0.837)
# Expected Total Damage and Damage (0.757)  

# Now, let's focus on others that are more notable:

# **Wounds and Points (0.576)**: There is a strong positive correlation between the number of wounds a unit has and its points value. Units with more wounds tend to have higher points costs.

# **Expected Total Damage and Attacks (0.658)**: The expected total damage a unit can deal has a strong positive correlation with the number of attacks it can make. Units with more attacks are likely to have higher expected total damage.

# **Save and Points (0.578)**: The save characteristic of a unit shows a moderate positive correlation with its points value. Units with better save characteristics may have higher points costs.

# ## Melee and Missile Unit Characteristics

# ### Statistics of Kind of Weapons

# In[24]:


mele_weapons = sigmar[sigmar['Kind'] == 'Mele weapon']
missile_weapons = sigmar[sigmar['Kind'] == 'Missile weapon']


# #### Total number of melee and missile units

# In[25]:


total_units = sigmar['Kind'].value_counts()

print("Total Units:")
print(total_units)


# We only have 7 missile units, let's see which factions and alliances they belong to

# #### Number of units per faction and alliance for Missile weapons

# In[26]:


# Number of units per faction for Missile weapons
faction_alliance_units_missile = sigmar[sigmar['Kind'] == 'Missile weapon'].groupby(['Grand Alliance', 'Faction', 'Kind']).size().reset_index(name='Unit Count')

print("\nUnits per Faction and Alliance (Missile weapons only):")
print(faction_alliance_units_missile)


# In[27]:


# Number of units per alliance for Missile weapons
alliance_units_missile = sigmar[sigmar['Kind'] == 'Missile weapon'].groupby(['Grand Alliance', 'Kind']).size().reset_index(name='Unit Count')

print("\nUnits per Alliance (Missile weapons only):")
print(alliance_units_missile)


# #### Interpretation

# We can see that we are fortunate to have at least one unit for each alliance, although they are not evenly distributed. The Order alliance has the most units of this type (3).

# On the other hand, we have one unit per faction, which also aids in the analysis. However, we only have 7 units in total to analyze, which may not have significant meaning, and we may not be able to draw decisive conclusions.

# ### General comparison between melee and missile weapons means

# In[28]:


mele_stats = mele_weapons[['Move', 'Wounds', 'Save', 'Bravery', 'Price', 'Points', 'Range', 'Attacks', 'To Hit', 'To Wound', 'Rend', 'Damage', 'To Hit & Wound', 'Expected Total Damage']].mean()
missile_stats = missile_weapons[['Move', 'Wounds', 'Save', 'Bravery', 'Price', 'Points', 'Range', 'Attacks', 'To Hit', 'To Wound', 'Rend', 'Damage', 'To Hit & Wound', 'Expected Total Damage']].mean()

combined_stats = pd.DataFrame({'Mele weapons': mele_stats, 'Missile weapons': missile_stats})
combined_stats = combined_stats.transpose()


# In[29]:


combined_stats


# #### Interpretation

#  Missile weapons generally have a higher Move value compared to Mele weapons. On average, also have slightly higher Wounds compared to Mele weapons, as well as generally have a higher average Points value. Besides, theys are better at Rend values,Damageand the Expected Total Damage.han Mele weapons.

# Mele weapons tend to have a higher average number of Attacks.

# We can conclude that, for the sample we have taken, missile units are much more powerful in almost every aspect than melee units

# ### Means of Misile weapons grouped by their Faction

# Since we only have three examples of units with missile weapons, and these belong to 7 different factions, let's examine them in detail

# In[30]:


missile_weapon_stats_factions = sigmar[sigmar['Kind'] == 'Missile weapon'].groupby(['Grand Alliance', 'Faction'])[['Move', 'Wounds', 'Save', 'Bravery', 'Price', 'Points', 'Range', 'Attacks', 'To Hit', 'To Wound', 'Rend', 'Damage', 'To Hit & Wound', 'Expected Total Damage']].mean()

missile_weapon_stats_factions


# #### Interpretation

# Entre these missile units, we see some that stand out, such as the Tzeentch faction, with a high movement (12) and the highest expected damage (1.44, considering it performs 3 attacks), also having a very long range. On the other hand, it is a rather weak unit with only one wound, little ability to defend against an attack, and it will easily flee.
# Some units stand out for their wounds, such as the Ogor Mawtribes, and others for their price, like those of Sylvaneth.

# ### Means of Misile/Mele weapons grouped by their Alliance and their differences

# In[31]:


means_columns = ['Move', 'Wounds', 'Save', 'Bravery', 'Price', 'Points', 'Range', 'Attacks', 'To Hit', 'To Wound', 'Rend', 'Damage', 'To Hit & Wound', 'Expected Total Damage']

missile_weapon_stats_alliance = sigmar[sigmar['Kind'] == 'Missile weapon'].groupby(['Grand Alliance'])[means_columns].mean().reset_index()
melee_weapon_stats_alliance = sigmar[sigmar['Kind'] == 'Mele weapon'].groupby(['Grand Alliance'])[means_columns].mean().reset_index()

missile_weapon_stats_alliance['Weapon Type'] = 'Missile'
melee_weapon_stats_alliance['Weapon Type'] = 'Melee'

combined_stats = pd.concat([missile_weapon_stats_alliance, melee_weapon_stats_alliance], ignore_index=True)

column_order = ['Grand Alliance', 'Weapon Type'] + means_columns
combined_stats = combined_stats[column_order]

combined_stats


# We are going to visually compare these metrics, starting with 'Wounds', 'Save', 'Bravery', 'Attacks', 'To Hit', 'To Wound', 'Rend', 'Damage', 'To Hit & Wound', 'Expected Total Damage', and then creating another set of four charts to compare 'Move', 'Price', 'Points', and 'Range'.

# In[32]:


categories = ['Wounds', 'Save', 'Bravery', 'Attacks', 'To Hit', 'To Wound', 'Rend', 'Damage', 'To Hit & Wound', 'Expected Total Damage']
colors = {'Missile': 'blue', 'Melee': 'orange'}

num_alliances = len(combined_stats['Grand Alliance'].unique())
bar_width = 0.35
index = np.arange(len(categories))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, alliance in enumerate(combined_stats['Grand Alliance'].unique()):
    row, col = divmod(i, 2)  # Calcula la fila y columna para el subplot
    ax = axes[row, col]
    
    missile_means = combined_stats[(combined_stats['Grand Alliance'] == alliance) & (combined_stats['Weapon Type'] == 'Missile')][categories].values.flatten()
    melee_means = combined_stats[(combined_stats['Grand Alliance'] == alliance) & (combined_stats['Weapon Type'] == 'Melee')][categories].values.flatten()
    
    bar1 = ax.bar(index, missile_means, bar_width, label='Missile', color=colors['Missile'])
    bar2 = ax.bar(index + bar_width, melee_means, bar_width, label='Melee', color=colors['Melee'])
    
    ax.set_ylabel('Mean Values')
    ax.set_title(f'Mean Values of Missile and Melee Units - {alliance}')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories, rotation='vertical')  # Rotar los nombres de las columnas verticalmente
    ax.legend()

    for bars in [bar1, bar2]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# **Analysis of these melee and missile unit variables for the Chaos alliance**

# Regarding the Chaos alliance, we observe that its missile units can endure more wounds, as well as having a greater number of attacks, higher damage, and a higher probability of penetrating the enemy's armor. Due to the combination of all these factors, the expected damage output is significantly higher among their missile units. Interestingly, the probability of dealing damage once they have hit the enemy is very similar.

# **Analysis of these melee and missile unit variables for the Death alliance**

# In contrast to Chaos, melee units can withstand more wounds, have a higher number of attacks, and higher probabilities in all phases of the attack, resulting in a total damage expectation that doubles that of missile units. On the other hand, missile units have higher bravery and armor penetration.

# **Analysis of these melee and missile unit variables for the Destruction alliance**

# For this alliance, except for the values of wounds that can be resisted, where missile units stand out significantly, for all other values, melee units are either tied or stand out. It is the only alliance where the armor-penetration capability is superior in melee units. The number of attacks is almost triple, as is the damage they inflict, which is reflected in the total damage expectation.

# **Analysis of these melee and missile unit variables for the Order alliance**

# It is an alliance where melee and missile units are quite balanced. Nevertheless, melee units can withstand more wounds and have a higher average number of attacks than missile units. For everything else, especially in the damage they cause and in the overall damage expectation, missile units stand out.

# In[33]:


categories = ['Move','Price', 'Points', 'Range']
colors = {'Missile': 'blue', 'Melee': 'orange'}

num_alliances = len(combined_stats['Grand Alliance'].unique())
bar_width = 0.35
index = np.arange(len(categories))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, alliance in enumerate(combined_stats['Grand Alliance'].unique()):
    row, col = divmod(i, 2)  # Calcula la fila y columna para el subplot
    ax = axes[row, col]
    
    missile_means = combined_stats[(combined_stats['Grand Alliance'] == alliance) & (combined_stats['Weapon Type'] == 'Missile')][categories].values.flatten()
    melee_means = combined_stats[(combined_stats['Grand Alliance'] == alliance) & (combined_stats['Weapon Type'] == 'Melee')][categories].values.flatten()
    
    bar1 = ax.bar(index, missile_means, bar_width, label='Missile', color=colors['Missile'])
    bar2 = ax.bar(index + bar_width, melee_means, bar_width, label='Melee', color=colors['Melee'])
    
    ax.set_ylabel('Mean Values')
    ax.set_title(f'Mean Values of Missile and Melee Units - {alliance}')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories, rotation='vertical')  # Rotar los nombres de las columnas verticalmente
    ax.legend()

    for bars in [bar1, bar2]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# For these metrics, we will overlook that the range is higher for missile units, but it is worth noting that in Chaos and Order units, it is almost double that of Destruction and Death units.

# **Analysis of these melee and missile unit variables for the Chaos alliance**

# They tend to be very balanced, although the missile units have a bit more movement and slightly more points than the melee units.

# **Analysis of these melee and missile unit variables for the Death alliance**

# 
# In terms of price and movement, the missile units are higher, but the melee units have a much higher point value (which could be considered a disadvantage, as it allows fewer units in your army) The missile units have higher movement..

# **Analysis of these melee and missile unit variables for the Destruction alliance**

# They tend to be very balanced, although the mele units have a bit more movement and slightly more points than the melee units.

# **Analysis of these melee and missile unit variables for the Order alliance**

# The missile units have higher movement.They also have many more points than the melee units. Otherwise, they are balanced.

# ### Standard Deviation of Misile/Mele weapons grouped by their Alliance

# In[34]:


missile_weapon_stats_alliance_std = sigmar[sigmar['Kind'] == 'Missile weapon'].groupby(['Grand Alliance'])[['Move', 'Wounds', 'Save', 'Bravery', 'Price', 'Points', 'Range', 'Attacks', 'To Hit', 'To Wound', 'Rend', 'Damage', 'To Hit & Wound', 'Expected Total Damage']].std().reset_index()
mele_weapon_stats_alliance_std = sigmar[sigmar['Kind'] == 'Mele weapon'].groupby(['Grand Alliance'])[['Move', 'Wounds', 'Save', 'Bravery', 'Price', 'Points', 'Range', 'Attacks', 'To Hit', 'To Wound', 'Rend', 'Damage', 'To Hit & Wound', 'Expected Total Damage']].std().reset_index()


# In[35]:


missile_weapon_stats_alliance_std


# In[36]:


mele_weapon_stats_alliance_std


# As there is a huge difference between the values on the columns, I thought about normalization, but it turned out that there was no standard deviation in the data after the normalization. Therefore, we will carefully analyze the data without normalizing

# In[37]:


# First, let's prepare the data we want to examine by grouping them by alliances and separating them by weapon type

chaos_mele_weapons = sigmar[(sigmar['Grand Alliance'] == 'Chaos') & (sigmar['Kind'] == 'Mele weapon')]
death_mele_weapons = sigmar[(sigmar['Grand Alliance'] == 'Death') & (sigmar['Kind'] == 'Mele weapon')]
destruction_mele_weapons = sigmar[(sigmar['Grand Alliance'] == 'Destruction') & (sigmar['Kind'] == 'Mele weapon')]
order_mele_weapons = sigmar[(sigmar['Grand Alliance'] == 'Order') & (sigmar['Kind'] == 'Mele weapon')]

chaos_missile_weapons = sigmar[(sigmar['Grand Alliance'] == 'Chaos') & (sigmar['Kind'] == 'Missile weapon')]
death_missile_weapons = sigmar[(sigmar['Grand Alliance'] == 'Death') & (sigmar['Kind'] == 'Missile weapon')]
destruction_missile_weapons = sigmar[(sigmar['Grand Alliance'] == 'Destruction') & (sigmar['Kind'] == 'Missile weapon')]
order_missile_weapons = sigmar[(sigmar['Grand Alliance'] == 'Order') & (sigmar['Kind'] == 'Missile weapon')]


# This is the **failed normalization attempt**

# In[38]:


# Create DataFrames for mele weapons for each alliance
# chaos_mele_normalized = chaos_mele_weapons.copy()
# chaos_mele_normalized[numeric_columns] = scaler.fit_transform(chaos_mele_weapons[numeric_columns])

# death_mele_normalized = death_mele_weapons.copy()
# death_mele_normalized[numeric_columns] = scaler.fit_transform(death_mele_weapons[numeric_columns])

# destruction_mele_normalized = destruction_mele_weapons.copy()
# destruction_mele_normalized[numeric_columns] = scaler.fit_transform(destruction_mele_weapons[numeric_columns])

# order_mele_normalized = order_mele_weapons.copy()
# order_mele_normalized[numeric_columns] = scaler.fit_transform(order_mele_weapons[numeric_columns])

# Create DataFrames for missile weapons for each alliance
# chaos_missile_normalized = chaos_missile_weapons.copy()
# chaos_missile_normalized[numeric_columns] = scaler.fit_transform(chaos_missile_weapons[numeric_columns])

# death_missile_normalized = death_missile_weapons.copy()
# death_missile_normalized[numeric_columns] = scaler.fit_transform(death_missile_weapons[numeric_columns])

# destruction_missile_normalized = destruction_missile_weapons.copy()
# destruction_missile_normalized[numeric_columns] = scaler.fit_transform(destruction_missile_weapons[numeric_columns])

# order_missile_normalized = order_missile_weapons.copy()
# order_missile_normalized[numeric_columns] = scaler.fit_transform(order_missile_weapons[numeric_columns])


# In[39]:


# Standard deviation for melee weapons
chaos_mele_std = chaos_mele_weapons.select_dtypes(include=['float64', 'int64']).std()
death_mele_std = death_mele_weapons.select_dtypes(include=['float64', 'int64']).std()
destruction_mele_std = destruction_mele_weapons.select_dtypes(include=['float64', 'int64']).std()
order_mele_std = order_mele_weapons.select_dtypes(include=['float64', 'int64']).std()

# Standard deviation for missile weapons
chaos_missile_std = chaos_missile_weapons.select_dtypes(include=['float64', 'int64']).std()
death_missile_std = death_missile_weapons.select_dtypes(include=['float64', 'int64']).std()
destruction_missile_std = destruction_missile_weapons.select_dtypes(include=['float64', 'int64']).std()
order_missile_std = order_missile_weapons.select_dtypes(include=['float64', 'int64']).std()



# In[40]:


# Let's put it together
mele_std_data = {
    'Chaos': chaos_mele_std,
    'Death': death_mele_std,
    'Destruction': destruction_mele_std,
    'Order': order_mele_std
}

alliances_mele_std = pd.DataFrame(mele_std_data)

missile_std_data = {
    'Chaos': chaos_missile_std,
    'Death': death_missile_std,
    'Destruction': destruction_missile_std,
    'Order': order_missile_std
}

alliances_missile_std = pd.DataFrame(missile_std_data)


# In[41]:


alliances_mele_std


# In[42]:


alliances_missile_std


# #### Interpretation

# **Mele units**

# Move: Chaos has the highest variability, indicating diverse mobility.  
# Wounds: Destruction units show the most variability, suggesting a broad range of durability.  
# Save: Chaos and Order are relatively stable, while Death and Destruction vary more.  
# Points & Price: High standard deviations in all alliances, especially Order.  
# Range: Consistent across factions.  
# Attacks: Chaos and Death show slightly higher variability.  
# To Hit & To Wound: Generally stable across alliances.  
# Rend & Damage: Moderate variability, especially in Destruction.

# **Missile units**

# Move: Significant variability in Chaos and Order, limited data for Death.  
# Wounds, Save, Bravery: Limited data for Death, low variability in available data  Chaos.

# Points & Price: High standard deviations in all alliances, especially Order.  
# Range: Extremely high variability, especially in Chaos.  
# Attacks: Limited data for Death, moderate variability elsewhere.  
# To Hit & To Wound: Generally stable across alliances.  
# Rend & Damage: Moderate variability, especially in Chaos.  

# These standard deviations reveal the degree of diversity and variability within each alliance, providing insights into their unit characteristics. It's important to note that some standard deviations might be influenced by the limited data available for certain alliances or specific units. Additionally, the interpretation is subject to the context and objectives of your analysis.
