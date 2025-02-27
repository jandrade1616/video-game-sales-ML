#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Paso 1. Abre el archivo de datos y estudia la información general 
import pandas as pd
import matplotlib.pyplot as plt

ruta_archivo = '/datasets/games.csv'
datos_juegos = pd.read_csv(ruta_archivo)

print(datos_juegos.head())
print(datos_juegos.info())
print(datos_juegos.describe())


# In[2]:


# Reemplazar los nombres de las columnas a minúsculas
datos_juegos.columns = datos_juegos.columns.str.lower()
print(datos_juegos.columns)

print(datos_juegos == 'TBD')
# No Exite ningun valor TBD


# In[3]:


# Convertir 'year_of_release' a entero (int), ignorando nulos
datos_juegos['year_of_release'] = pd.to_numeric(datos_juegos['year_of_release'], errors='coerce').astype('Int64')
datos_juegos['critic_score'] = pd.to_numeric(datos_juegos['critic_score'], errors='coerce').astype('Int64')
datos_juegos['user_score'] = pd.to_numeric(datos_juegos['user_score'], errors='coerce')
print(datos_juegos.dtypes)


# In[4]:


#Manejo de valores ausentes.
# Revisar los valores nulos en cada columna
valores_nulos = datos_juegos.isnull().sum()
print(valores_nulos)


# In[7]:


critic_score_median = datos_juegos['critic_score'].median()
user_score_median = datos_juegos['user_score'].median()

datos_juegos['critic_score'].fillna(critic_score_median, inplace=True)
datos_juegos['user_score'].fillna(user_score_median, inplace=True)

valores_nulos = datos_juegos.isnull().sum()
print(valores_nulos)


# In[8]:


#Ventas totales
datos_juegos['total_sales'] = datos_juegos['na_sales'] + datos_juegos['eu_sales'] + datos_juegos['jp_sales'] + datos_juegos['other_sales']
print(datos_juegos[['name', 'total_sales']].head())


# In[9]:


# Lazamientos por año
lanzamientos_por_ano = datos_juegos['year_of_release'].value_counts().sort_index()

print(lanzamientos_por_ano)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
lanzamientos_por_ano.plot(kind='bar')
plt.title('Número de Juegos Lanzados por Año')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Número de Juegos')
plt.show()


# In[10]:


# Calcular las ventas totales por plataforma
ventas_por_plataforma = datos_juegos.groupby('platform')['total_sales'].sum().sort_values(ascending=False)

print(ventas_por_plataforma)
plataformas_principales = ventas_por_plataforma.head(5).index
print(f"Plataformas principales: {plataformas_principales}")


# In[11]:


import matplotlib.pyplot as plt

# Seleccionar los últimos 5 años del dataset
anio_inicio = 2012
anio_fin = 2016

# Filtrar los datos para los años de interés
datos_recientes = datos_juegos[(datos_juegos['year_of_release'] >= anio_inicio) & 
                               (datos_juegos['year_of_release'] <= anio_fin)]

plataformas_principales = ['PS2', 'X360', 'PS3', 'Wii', 'DS']
datos_principales_plataformas = datos_recientes[datos_recientes['platform'].isin(plataformas_principales)]

ventas_por_ano_plataforma = datos_principales_plataformas.pivot_table(
    index='year_of_release', columns='platform', values='total_sales', aggfunc='sum')

plt.figure(figsize=(14, 7))
ventas_por_ano_plataforma.plot()

plt.xticks(range(anio_inicio, anio_fin + 1), rotation=60, fontsize=10)
plt.title('Tendencias de Ventas por Año para Plataformas Principales (2012-2016)')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Ventas Totales (millones)')
plt.legend(title='Plataforma', loc='upper left')
plt.tight_layout()
plt.show()


# In[10]:


datos_relevantes = datos_juegos[(datos_juegos['year_of_release'] >= 2013) & (datos_juegos['year_of_release'] <= 2016)]

ventas_totales_relevantes = datos_relevantes.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
print(ventas_totales_relevantes)
plataformas_rentables = ventas_totales_relevantes.head(5).index
print(f"Plataformas potencialmente rentables: {plataformas_rentables}")



ventas_por_ano_plataforma = datos_relevantes.pivot_table(
    index='year_of_release', columns='platform', values='total_sales', aggfunc='sum')

plt.figure(figsize=(14, 7))
ventas_por_ano_plataforma.plot(ax=plt.gca())
plt.title('Tendencias de Ventas por Año para Plataformas (2013-2016)')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Ventas Totales (millones)')
plt.legend(title='Plataforma', loc='upper right')


plt.xticks(ticks=range(2013, 2017), labels=range(2013, 2017), rotation=0)
plt.show()




# In[11]:


# Crear un diagrama de caja para ventas globales por plataforma
datos_relevantes = datos_juegos[(datos_juegos['year_of_release'] >= 2013) & (datos_juegos['year_of_release'] <= 2016)]

plt.figure(figsize=(12, 6))
datos_relevantes.boxplot(column='total_sales', by='platform')
plt.title('Diagrama de Caja de Ventas Globales por Plataforma (2013-2016)')
plt.suptitle('')
plt.xlabel('Plataforma')
plt.ylabel('Ventas Totales (millones)')
plt.xticks(rotation=45)
plt.show()


# In[20]:


import numpy as np

# Convertir columnas relevantes a float64 explícitamente
datos_xone['critic_score'] = datos_xone['critic_score'].astype(np.float64)
datos_xone['user_score'] = datos_xone['user_score'].astype(np.float64)
datos_xone['total_sales'] = datos_xone['total_sales'].astype(np.float64)

# Verificar que las conversiones se hicieron correctamente
print(datos_xone.dtypes)

# Calcular la correlación nuevamente
correlacion_criticos_xone = datos_xone['critic_score'].corr(datos_xone['total_sales'])
correlacion_usuarios_xone = datos_xone['user_score'].corr(datos_xone['total_sales'])

print(f'Correlación entre Critic Score y Ventas para Xbox One: {correlacion_criticos_xone}')
print(f'Correlación entre User Score y Ventas para Xbox One: {correlacion_usuarios_xone}')

plt.figure(figsize=(10, 5))
plt.scatter(datos_xone['critic_score'], datos_xone['total_sales'], label='Critic Score', alpha=0.6, color='blue')
plt.scatter(datos_xone['user_score'], datos_xone['total_sales'], label='User Score', alpha=0.6, color='orange')
plt.title('Impacto de Reseñas en Ventas para Xbox One')
plt.xlabel('Puntaje')
plt.ylabel('Ventas Totales (millones)')
plt.legend()
plt.show()


# In[27]:


import matplotlib.pyplot as plt


top_50_juegos = datos_relevantes.groupby('name')['total_sales'].sum().nlargest(50).index
datos_top_50 = datos_relevantes[datos_relevantes['name'].isin(top_50_juegos)]
ventas_top_50_comparadas = datos_top_20.groupby(['name', 'platform'])['total_sales'].sum().reset_index()
print(ventas_top_50_comparadas.head(50))

plt.figure(figsize=(15, 8))
for name in ventas_top_50_comparadas['name'].unique():
    subset = ventas_top_50_comparadas[ventas_top_50_comparadas['name'] == name]
    plt.plot(subset['platform'], subset['total_sales'], marker='o', label=name)

plt.title('Comparación de Ventas de los 50 Juegos Más Vendidos en Diferentes Plataformas')
plt.xlabel('Plataforma')
plt.ylabel('Ventas Totales (millones)')
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()


# In[26]:


# Agrupar por género para obtener las ventas totales por género
ventas_por_genero = datos_relevantes.groupby('genre')['total_sales'].sum().reset_index()
ventas_por_genero = ventas_por_genero.sort_values(by='total_sales', ascending=False)

# Mostrar los resultados
print(ventas_por_genero)

# Graficar la distribución de ventas por género
plt.figure(figsize=(12, 6))
plt.bar(ventas_por_genero['genre'], ventas_por_genero['total_sales'], color='skyblue')
plt.title('Ventas Totales por Género de Juego')
plt.xlabel('Género')
plt.ylabel('Ventas Totales (millones)')
plt.xticks(rotation=45)
plt.show()


# In[28]:


# Calcular ventas totales por plataforma en cada región
plataformas_na = datos_relevantes.groupby('platform')['na_sales'].sum().sort_values(ascending=False).head(5)
plataformas_ue = datos_relevantes.groupby('platform')['eu_sales'].sum().sort_values(ascending=False).head(5)
plataformas_jp = datos_relevantes.groupby('platform')['jp_sales'].sum().sort_values(ascending=False).head(5)

print("Plataformas principales en NA:")
print(plataformas_na)
print("\nPlataformas principales en UE:")
print(plataformas_ue)
print("\nPlataformas principales en JP:")
print(plataformas_jp)


# In[29]:


# Calcular ventas totales por género en cada región
generos_na = datos_relevantes.groupby('genre')['na_sales'].sum().sort_values(ascending=False).head(5)
generos_ue = datos_relevantes.groupby('genre')['eu_sales'].sum().sort_values(ascending=False).head(5)
generos_jp = datos_relevantes.groupby('genre')['jp_sales'].sum().sort_values(ascending=False).head(5)

print("Géneros principales en NA:")
print(generos_na)
print("\nGéneros principales en UE:")
print(generos_ue)
print("\nGéneros principales en JP:")
print(generos_jp)


# In[30]:


# Calcular ventas totales por clasificación ESRB en cada región
esrb_na = datos_relevantes.groupby('rating')['na_sales'].sum().sort_values(ascending=False)
esrb_ue = datos_relevantes.groupby('rating')['eu_sales'].sum().sort_values(ascending=False)
esrb_jp = datos_relevantes.groupby('rating')['jp_sales'].sum().sort_values(ascending=False)

print("Ventas por clasificación ESRB en NA:")
print(esrb_na)
print("\nVentas por clasificación ESRB en UE:")
print(esrb_ue)
print("\nVentas por clasificación ESRB en JP:")
print(esrb_jp)


# In[16]:


# Prueba de hipótesis para Xbox One vs. PC
from scipy import stats

# Nivel de significancia
alfa = 0.05

xone_scores = datos_relevantes[datos_relevantes['platform'] == 'XOne']['user_score'].dropna()
pc_scores = datos_relevantes[datos_relevantes['platform'] == 'PC']['user_score'].dropna()

# Prueba de Levene para comprobar igualdad de varianzas
stat_levene, p_value_levene = stats.levene(xone_scores, pc_scores)


if p_value_levene > alfa:
    print(f"No se rechaza la hipótesis nula de Levene: Las varianzas son iguales (p-value = {p_value_levene}).")
    equal_var = True
else:
    print(f"Se rechaza la hipótesis nula de Levene: Las varianzas no son iguales (p-value = {p_value_levene}).")
    equal_var = False

# Prueba t de Student con el resultado de Levene
t_stat_1, p_value_1 = stats.ttest_ind(xone_scores, pc_scores, equal_var=equal_var)
print(f"Prueba de Hipótesis 1 (Xbox One vs PC): t-statistic = {t_stat_1}, p-value = {p_value_1}")

# Interpretación del resultado
if p_value_1 < alfa:
    print("Rechazamos la hipótesis nula: Las calificaciones promedio de Xbox One y PC son diferentes.")
else:
    print("No se rechaza la hipótesis nula: No hay suficiente evidencia para decir que las calificaciones promedio de Xbox One y PC son diferentes.")


# In[17]:


from scipy import stats

# Nivel de significancia
alfa = 0.05

accion_scores = datos_relevantes[datos_relevantes['genre'] == 'Action']['user_score'].dropna()
deportes_scores = datos_relevantes[datos_relevantes['genre'] == 'Sports']['user_score'].dropna()

# Prueba de Levene para comprobar igualdad de varianzas
stat_levene, p_value_levene = stats.levene(accion_scores, deportes_scores)


if p_value_levene > alfa:
    print(f"No se rechaza la hipótesis nula de Levene: Las varianzas son iguales (p-value = {p_value_levene}).")
    equal_var = True
else:
    print(f"Se rechaza la hipótesis nula de Levene: Las varianzas no son iguales (p-value = {p_value_levene}).")
    equal_var = False


t_stat_2, p_value_2 = stats.ttest_ind(accion_scores, deportes_scores, equal_var=equal_var)
print(f"Prueba de Hipótesis 2 (Acción vs Deportes): t-statistic = {t_stat_2}, p-value = {p_value_2}")


if p_value_2 < alfa:
    print("Rechazamos la hipótesis nula: Las calificaciones promedio de Acción y Deportes son diferentes.")
else:
    print("No se rechaza la hipótesis nula: No hay suficiente evidencia para decir que las calificaciones promedio de Acción y Deportes son diferentes.")

