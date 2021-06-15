
import numpy as np #Support for creating large matrices and multidimensional vectors, with a large collection of mathematical functions
from numpy.core.fromnumeric import size #Create vectors, matrices, and mathematical functions to operate on these
import pandas as pd #Data manipulation and analysis
import matplotlib.pyplot as plt #Generation of graphs from data contained in lists
import  plotly.offline as pyo  #Work with plotly offline and generate html file
import plotly.express as px #Optimized "plotly" to work with data frames
import seaborn as sns #Provides a high-level graphical interface
sns.set() #Gives modeling to the figures


#Load the dataset using Pandas functionalities

data_complete = pd.read_csv("~/Documentos/ml_stanford_rb/DatosInmobiliaria.csv")
data_complete = pd.DataFrame(data_complete)

# We select the three most abundant classes of property types and the region with the most published properties and create a new data frame with those instances.
# We show the first five data

data_map = data_complete[((data_complete['property_type'] == "Departamento") |(data_complete['property_type'] == "Casa") | (data_complete['property_type'] == "PH"))  & (data_complete['l2'] == "Capital Federal")] 
data_map.head()


#we see some basic statistical details in the dataset

data_map.describe()


print(data_map["lat"].count()) #The count() method returns the number of times the specified element appears in the list
print(data_map["lon"].count())

before_count_lat=data_map["lat"].isnull().sum() #This function takes a scalar or array-like object and indicates if values ​​are missing
                                                #With "sum" we see the total amount of null values
print("la cantidad de valores nulos en lat es: ",before_count_lat)

before_count_lon=data_map["lon"].isnull().sum() #This function takes a scalar or array-like object and indicates if values ​​are missing
                                                #With "sum" we see the total amount of null values
print("la cantidad de valores nulos en lon es: ",before_count_lon)

data_map.dropna(subset=["lat","lon"],inplace=True) #data_map.dropna = Remove missing values
                                                   #subset=["lat","lon"] = Labels along other axis to consider

#We recheck the null values ​​after modifying the dataset, this should give us 0

after_count_lat=data_map["lat"].isnull().sum()
print("la cantidad de valores nulos en lat es: ",after_count_lat)

after_count_lon=data_map["lon"].isnull().sum()
print("la cantidad de valores nulos en lon es: ",after_count_lon)

#We reviewed the total data we had of the different types of properties

data_map["property_type"].value_counts()

# In a Mapbox scatterplot, each row in the data_frame is represented by a symbol mark on a Mapbox map

fig = px.scatter_mapbox(data_map, lat="lat", lon="lon", color="property_type",hover_data=['price'],hover_name="title",   # data_map = this argument must be passed for column names to be used (and not keyword names)
                        animation_frame="l3",zoom=10,                                                                    # lat="lat" = the name of a column in data_frame, or a Pandas Series or array_like object. The values ​​in this column or array_like are used to place the marks according to latitude on a map
                        title="Mapa de Capital Federal para Departamentos,PH y Casa"                                     # lon="lon" = the name of a column in data_frame, or a Pandas Series or array_like object. The values ​​in this column or array_like are used to place the marks based on longitude on a map
                       )                                                                                                 # color="property_type" = the name of a column in data_frame, or a Pandas Series or array_like object. The values ​​in this column or array_like are used to color the marks
                                                                                                                         # hover_data=['price'] = The values ​​in these columns appear as additional data in the tooltip
                                                                                                                         # hover_name="title" = el nombre de una columna en data_frame, o un objeto Pandas Series o array_like. Los valores de esta columna o array_like aparecen en negrita en la información sobre herramientas flotante
                                                                                                                         # animation_frame="l3" = el nombre de una columna en data_frame, o un objeto Pandas Series o array_like. Los valores de esta columna o array_like se utilizan para asignar marcas a los fotogramas de animación.
                                                                                                                         # zoom=10 = Sets the zoom level of the map
                                                                                                                         # title="Mapa de Capital Federal para Departamentos,PH y Casa = the title of the figure
# fig.update_layout = used to update multiple nested properties of a figure's design
# we bring to the code a map that will help us show the data

fig.update_layout(mapbox_style="open-street-map")

# Instantly, interactive charts overlaid on the Python-created dataset can be shared

pyo.offline.plot(fig, filename = "Mapa de Capital Federal para Departamentos,PH y Casa.html")

fig.show()
