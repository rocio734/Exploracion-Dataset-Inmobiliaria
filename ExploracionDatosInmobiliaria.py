#I propose to create a Machine Learning model that, given the characteristics of the property, predicts its sale price

#We import the necessary libraries

import numpy as np #Support for creating large matrices and multidimensional vectors, with a large collection of mathematical functions
from numpy.core.fromnumeric import size #Create vectors, matrices, and mathematical functions to operate on these
import pandas as pd #Data manipulation and analysis
import matplotlib.pyplot as plt #Generation of graphs from data contained in lists
import seaborn as sns #Provides a high-level graphical interface
from matplotlib.ticker import ScalarFormatter #Format the tick values ​​as a number
from matplotlib import gridspec #Creates cells of equal size by adjusting the relative heights and widths of rows and columns
import scipy as sp #It provides more utility functions for optimization, stats and signal processing
from scipy import stats #Has probability distributions and statistical functions
from sklearn import metrics #Implement functions that evaluate prediction error for specific purposes
from sklearn.model_selection import train_test_split #Divide matrices into test subsets and random train
from sklearn.metrics import mean_squared_error #Regression loss of mean square error
from sklearn.linear_model import LinearRegression #Linear regression by ordinary least squares
from sklearn.metrics import mean_absolute_error #Regression loss of mean absolute error
from statsmodels.stats.outliers_influence import variance_inflation_factor #Is a measure of collinearity between the predictor variables within a multiple regression
from sklearn import preprocessing #Changes the raw feature vectors to a more suitable representation for later estimators.
from sklearn.feature_selection import RFE #Select features recursively considering smaller and smaller feature sets
from sklearn.svm import SVR #Vector regression with epsilon support
import math #Be able to use mathematical functions
import plotly.express as px #Optimized "plotly" to work with data frames
import  plotly.offline as pyo  #Work with plotly offline and generate html file

sns.set() #Gives modeling to the figures

#Load the dataset using Pandas functionalities

data = pd.read_csv("~/Documentos/ml_stanford_rb/DatosInmobiliaria.csv")
data = pd.DataFrame(data)

#We show how many rows and columns it has, and its first five instances

data.head()
print(data.head())

data.tail()
print(data.tail())

data.shape

#Show the column names and how many missing values ​​are per column


data.isnull().any()     #Functions that return a boolean value indicating whether the passed in argument value is in fact missing data.

#We see if there are null values ​​with "insull" and how many null values ​​there are with "sum"


count=data["lat"].isnull().sum()
print("la cantidad de valores nulos en lat es: ",count)

count1=data["lon"].isnull().sum()
print("la cantidad de valores nulos en lon es: ",count1)

count2=data["bathrooms"].isnull().sum()
print("la cantidad de valores nulos en bathrooms es: ",count2)

count3=data["surface_total"].isnull().sum()
print("la cantidad de valores nulos en surface_total es: ",count3)

count4=data["surface_covered"].isnull().sum()
print("la cantidad de valores nulos en surface_covered es: ",count4)

#We see how many property types are published based on the data and how many instances of each property type are in the dataset

tipos = data["property_type"].unique()  #Encuentra los valores unicos que hay en esa categoria
tipos = tipos.tolist() #to convert a string type series to a list
print(tipos)

instancias = data["property_type"].value_counts() #The value_counts () method returns a string containing the unique value counts. 
                                                  #This means that, for any column in a data frame, this method returns the count of unique entries in that column.
print(instancias)

#A graph is presented with the types of properties and their quantities

plt.figure(figsize=(10,5))  #Width, height in inches
plt.yscale('log')  #Is used to set the y-axis scale/ Log= Log scale
plt.gca().yaxis.set_major_formatter(ScalarFormatter()) #We define the scale without decimals
ax = sns.countplot(data = data, x = "property_type")   #Show the counts of observations in each categorical bin using bars
                                                       #The data on the x-axis will be equal to the property type in the dataset
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha="right")  #Methods to rotate x-axis mark label text and not overlap values// we turn 40 degrees to the right
                                                                 #"get_xticklabels" sets the rotation property of the xtick label object.
plt.show()

#Region of publications and distribution graphics according to GBA and according to neighborhoods
 
fig= plt.subplots(figsize=(20,18),constrained_layout=True) #constrained_layout automatically adjusts subplots and decorations like legends and colorbars 
                                                     #so that they fit in the figure window while still preserving, as best they can, the logical layout requested by the user.
                                                     #Figsize= Width, height in inches
grid = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) #A grid layout to place subplots within a figure
                                                     #(The number of rows and columns in the grid // Defines the relative heights of the rows. Each column gets a relative height of height_ratios[i] / sum(height_ratios))
ax1=plt.subplot(grid[0]) #set grid spacing
sns.countplot(data=data,y="l2",order=data["l2"].value_counts().index,ax=ax1,color="g") #Sns.countplot= the length of the bar is proportional to the number of elements represented by it
                                                                                     #value_counts()= method returns the count of unique entries in that column l2
                                                                                     #index,ax= set graphics grid spacing to green

ax1.set_yticklabels(ax1.get_yticklabels(),fontsize="medium")     #Set the ytick labels with the string label list // make the labels on the "y" axis medium in size

ax1.set_title("Distribucion segun el G.B.A.", fontsize= 'large') #Title located in the x-axis direction with large size
ax2=plt.subplot(grid[1]) #set grid spacing= 1
sns.countplot(data=data,x="l3",order=data["l3"].value_counts().index,ax=ax2,color="b")  #Sns.countplot= the length of the bar is proportional to the number of elements represented by it
                                                                                     #value_counts()= method returns the count of unique entries in that column l3
                                                                                     #index,ax= set graphics grid spacing to green



ax2.set_title("Distribucion segun barrios", fontsize= 'large') #Title located in the x-axis direction with large size
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90,ha="right") #Methods to rotate x-axis mark label text and not overlap values// we turn 90 degrees to the right
                                                                  #"get_xticklabels" sets the rotation property of the xtick label object.
plt.yticks(fontsize= 11) #I set the size of the y-axis labels
ax1.grid()
ax2.grid()
plt.show()

#We selected the three most abundant classes of property types and the region with the most published properties and create a new data frame with those instances.
#We show the number of rows and columns it has

data_2 = data[((data['property_type'] == "Departamento") |(data['property_type'] == "Casa") | (data['property_type'] == "PH"))  & (data['l2'] == "Capital Federal")] 
data_2.shape
print(data_2.shape)

#For a first look at the presented variables we use "PairPlot".We are going to study the distribution and pairwise relationships of the variables for each property.
#We will see a joint model of the relationships between property and variable

figz= plt.figure(constrained_layout=True)
mask_cols= ["property_type","price","surface_covered","surface_total","rooms","bedrooms","bathrooms"]  #Mask columns of a 2D array that contain masked values
graph=sns.pairplot(data_2[mask_cols],hue="property_type") #By default, this function will create an axis grid so that each numeric variable in the data will be shared 
                                                          #between the y-axes in a single row and the x-axes in a single column. Diagonal plots are treated differently: 
                                                          # A univariate distribution plot is drawn to show the marginal distribution of the data in each column.
                                                          #hue="property_type" = Variable in data to map plot aspects to different colors
graph.fig.set_size_inches(16,8) #Is used to set the figure size in inches // These parameters are the (width, height) of the figure in inches

plt.grid()
plt.show()

#We convert the above into a new dataset as another variable

data_3 = data_2[mask_cols].copy()
data_3 = data_3.reindex(sorted(data_3.columns), axis=1)

#With "describe" and "Pairplot" we can see some of the relationships within the dataset

data_3.describe()
print(data_3.describe)

#Some characteristics that we see is that the number 14 in bathrooms is surely "outlier", 21 rooms is "outlier", the covered area and the total of 126,062 is "outlier"
#That the minimum number of rooms is 0 indicates that there may be studio apartments

#Maximum number of bathrooms=14

data_3["bathrooms"].value_counts().sort_index() #To get the output, the left numeric column in ascending order
#Values ​​greater than 6 are not very representative

#Maximum number of rooms= 26

data_3["rooms"].value_counts().sort_index() #To get the output, the left numeric column in ascending order
#Values ​​greater than 7 are not very representative

#We plot the outliers and the total area.
#We plot outliers and area covered.

fig,(ax1,ax2) = plt.subplots(2,constrained_layout=True,figsize=(10,10)) #define the number of rows and columns of the subplot grid 
                                                                        #The restrained design perfectly fits the graphics within your figure
                                                                        #We give you the size of the figure
ax1.set_title("outliers y superficie total")
ax2.set_title("outliers y superficie cubierta")

sns.boxplot(data=data_3,x="surface_total",ax=ax1) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it

sns.boxplot(data=data_3,x="surface_covered",ax=ax2) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
plt.grid()
plt.show()

#Values ​​greater than 20000 are outlier


#Some instances have very large or very small total area values ​​and make it difficult to display correctly
#We use IQR 
#The use of the 0.01 quantile is to avoid having negative values ​​in the minimum surface

unicos = data_3["property_type"].unique() #We call as "unicos" variable the values ​​of data set 3 attributed to "property_type"

for x in unicos:  #If you want to operate on each element of "property_type", we use a loop "for"

#IQR= The interquartile range (IQR) is the difference between the 75th percentile (0.75 quantile) and the 25th percentile (0.25 quantile) 
#The IQR can be used to detect outliers in the data

    Q1 = data_3[data_3["property_type"]==x]["surface_total"].quantile(0.25)
    Q3 = data_3[data_3["property_type"]==x]["surface_total"].quantile(0.75)
    IQR = Q3 - Q1
    surface_min = data_3[data_3["property_type"]==x]["surface_total"].quantile(0.01) #We could have defined it as surface_min = (Q1 - 1.5 * IQR) but we use 
                                                                                     #the quantile 0.01 is to avoid having negative values ​​in the minimum surface
    surface_max = Q3 + (IQR*1.5) #It is a new range of decision that we choose to see the outliers// Upper Bound
                                 #When the scale is taken as 1.5, according to the IQR method, any data that is beyond 2.7σ from the mean (μ), anywhere, 
                                 #will be considered an outlier. And this decision range is the closest to what the Gaussian Distribution tells us, that is, 3σ.


    print(x)
    print("la Superficie maxima es {} y la superficie minima es {} y el IQR {}" .format(surface_max,surface_min,IQR))
    print("-----------")

#We create a new variable with the recently obtained data

dpto_New=data_3[data_3["property_type"]=="Departamento"]
ph_New=data_3[data_3["property_type"]=="PH"]
casa_New=data_3[data_3["property_type"]=="Casa"]


#we make the total surface less than the maximum surface and the total surface greater than the minimum surface

dpto_New= dpto_New[(dpto_New.surface_total <= 177.0) & (dpto_New.surface_total >=25.0) ]
ph_New=ph_New[(ph_New.surface_total <= 275.0) & (ph_New.surface_total >=35.0) ]
casa_New=casa_New[(casa_New.surface_total <= 579.0) & (casa_New.surface_total >=56.0) ]



figure,(ax1,ax2,ax3) = plt.subplots(3,constrained_layout=True,figsize=(10,10)) #define the number of rows and columns of the subplot grid 
                                                                        #The restrained design perfectly fits the graphics within your figure
                                                                        #We give you the size of the figure
ax1.set_title("Relacion tipo de propiedad y superficie total")
ax2.set_title("Relacion tipo de propiedad y superficie total")
ax3.set_title("Relacion tipo de propiedad y superficie total")
sns.boxplot(data=dpto_New,x="surface_total",y="property_type",ax=ax1) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=ph_New,x="surface_total",y="property_type",ax=ax2)   #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=casa_New,x="surface_total",y="property_type",ax=ax3) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
ax1.grid()
ax2.grid()
ax3.grid()
plt.show()

#Properties cannot have surface_covered greater than surface_total. If that happens, you need to filter those instances
#A filter is performed on the data where the covered area is greater than the total area

conc_1=[dpto_New,ph_New,casa_New] #I create a variable that has the new values ​​of the properties
data_4= pd.concat(conc_1)         #Concatenate pandas objects along a particular axis with optional set logic along the other axes
                                  #We concatenate the data of "conc_1" in a variable
data_4.describe()  
print(data_4.describe)


#In the graph we see that we have properties where the covered surface is greater than the total surface

sns.scatterplot(data=data_4, x='surface_total', y='surface_covered')
plt.grid()
plt.show()


#We make the arrangement

data_4.drop(data_4.loc[data_4['surface_covered'] > data_4['surface_total']].index,inplace=True ,axis=0) #data_4.drop = we remove elements // data_4.loc = Access a group of rows and columns by label 

#The graph below shows that after filtering, no property has a covered area greater than the total area

sns.scatterplot(data=data_4, x='surface_total', y='surface_covered')
plt.grid()
plt.show()


#The price range that the properties take is very wide so we study the distribution of that variable and filter it by a reasonable value that allows us obtain understandable graphics.
#We use the IQR.
#The use of the 0.01 quantile is to avoid having negative values ​​in the minimum price.

unique= data_4["property_type"].unique()  # We call as "unique" variable the values ​​of data set 4 attributed to "property_type"

#IQR= The interquartile range (IQR) is the difference between the 75th percentile (0.75 quantile) and the 25th percentile (0.25 quantile) 
#The IQR can be used to detect outliers in the data

for x in unique:

    Q1 = data_4[data_4["property_type"]==x]["price"].quantile(0.25)
    Q3 = data_4[data_4["property_type"]==x]["price"].quantile(0.75)
    IQR = Q3 - Q1
    precio_min = data_4[data_4["property_type"]==x]["price"].quantile(0.01) #We could have defined it as price_min = (Q1 - 1.5 * IQR) but we use 
                                                                            #the quantile 0.01 is to avoid having negative values ​​in the minimum price
    precio_max = Q3 + (IQR*1.5)  #It is a new range of decision that we choose to see the outliers// Upper Bound
                                 #When the scale is taken as 1.5, according to the IQR method, any data that is beyond 2.7σ from the mean (μ), anywhere, 
                                 #will be considered an outlier. And this decision range is the closest to what the Gaussian Distribution tells us, that is, 3σ
    print(x)
    print("el precio maximo es {}, el precio minimo es {} y el IQR {}" .format(precio_max,precio_min,IQR))
    print("-------------------------------------------------------------------")
#We obtain maximum price, minimum price and the IQR of each type of property


#We change the name of the variables with the data obtained recently
dpto_New=data_4[data_4["property_type"]=="Departamento"]
ph_New=data_4[data_4["property_type"]=="PH"]
casa_New=data_4[data_4["property_type"]=="Casa"]

#I define the prices of each property lower than the maximum prices and higher than the minimum prices

dpto_New= dpto_New[(dpto_New.price <= 440750.0) & (dpto_New.price >=62000.0) ]
ph_New=ph_New[(ph_New.price <= 447650.0) & (ph_New.price >=69900.0) ]
casa_New=casa_New[(casa_New.price <= 785000.0) & (casa_New.price >=99000.0) ]


#We see the relationship between property type and price by making a box plot using our IQR obtained

figure,(ax1,ax2,ax3) = plt.subplots(3,constrained_layout=True,figsize=(10,10)) #define the number of rows and columns of the subplot grid 
                                                                               #The restrained design perfectly fits the graphics within your figure
                                                                               #We give you the size of the figure

ax1.set_title("Relacion tipo de propiedad y precio(En miles de Dolares)")
ax2.set_title("Relacion tipo de propiedad y precio(En miles de Dolares)")
ax3.set_title("Relacion tipo de propiedad y precio(En  Dolares)")

sns.boxplot(data=dpto_New,x="price",y="property_type",ax=ax1)  #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=ph_New,x="price",y="property_type",ax=ax2)    #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=casa_New,x="price",y="property_type",ax=ax3)  #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
ax1.grid()
ax2.grid()
ax3.grid()
plt.show()
#We obtain the maximum and minimum number of bedrooms in the properties and the IQR

#We create a list the maximum and minimum 

rooms_min_list=[]
rooms_max_list=[]

#As the values ​​are very high, we restrict it

unique= data_4["property_type"].unique() # We call as "unique" variable the values ​​of data set 4 attributed to "property_type"

#IQR= The interquartile range (IQR) is the difference between the 75th percentile (0.75 quantile) and the 25th percentile (0.25 quantile) 
#The IQR can be used to detect outliers in the data

for x in unique:

    Q1 = data_4[data_4["property_type"]==x]["rooms"].quantile(0.25)
    Q3 = data_4[data_4["property_type"]==x]["rooms"].quantile(0.75)
    IQR = Q3 - Q1
    rooms_min = data_4[data_4["property_type"]==x]["rooms"].quantile(0.01)   ##We could have defined it as rooms_min = (Q1 - 1.5 * IQR) but we use 
                                                                             #the quantile 0.01 is to avoid having negative values ​​in the minimum rooms
    rooms_max = Q3 + (IQR*1.5)   #It is a new range of decision that we choose to see the outliers// Upper Bound
                                 #When the scale is taken as 1.5, according to the IQR method, any data that is beyond 2.7σ from the mean (μ), anywhere, 
                                 #will be considered an outlier. And this decision range is the closest to what the Gaussian Distribution tells us, that is, 3σ
    
    rooms_min_list.append(Q1 - (IQR*1.5)) #We name the list we had before with the new values
    rooms_max_list.append(Q3 + (IQR*1.5)) #The append () method of the FormData interface appends a new value onto an existing key inside a FormData object
    
    print(x)
    print("el numero maximo de rooms es {}, el numero minimo de rooms es {} y el IQR {}" .format(rooms_max,rooms_min,IQR))
    print("-------------------------------------------------------------------")

#We see that the values ​​are not integers, so we use the ceil, to raise them to the nearest integer

for i in rooms_max_list:
    ceil_max=math.ceil(i)
    print(ceil_max)


    for i in rooms_min_list:
        ceil_min =math.ceil(i)
    print(ceil_min)


#I am adjusting the new values ​​obtained to the variables we had

dpto_New=data_4[data_4["property_type"]=="Departamento"]
ph_New=data_4[data_4["property_type"]=="PH"]
casa_New=data_4[data_4["property_type"]=="Casa"]

#I define the number of bedrooms in the properties less than the maximum value and greater than the minimum value

dpto_New= dpto_New[(dpto_New.rooms <= 5.0) & (dpto_New.rooms >=1.0) ]
ph_New=ph_New[(ph_New.rooms <= 6.0) & (ph_New.rooms >=2.0) ]
casa_New=casa_New[(casa_New.rooms <= 9.0) & (casa_New.rooms >=1.0) ]


#We see the relationship between property type and price by making a box plot using our IQR obtained

figure,(ax1,ax2,ax3) = plt.subplots(3,constrained_layout=True,figsize=(10,10)) #define the number of rows and columns of the subplot grid 
                                                                               #The restrained design perfectly fits the graphics within your figure
                                                                               #We give you the size of the figure

ax1.set_title("Relacion tipo de propiedad y rooms")
ax2.set_title("Relacion tipo de propiedad y rooms")
ax3.set_title("Relacion tipo de propiedad y rooms")

sns.boxplot(data=dpto_New,x="rooms",y="property_type",ax=ax1)  #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=ph_New,x="rooms",y="property_type",ax=ax2)    #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=casa_New,x="rooms",y="property_type",ax=ax3)  #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it

ax1.grid()
ax2.grid()
ax3.grid()
plt.show()


#To get the output, the left numeric column in ascending order
data_4["bathrooms"].value_counts().sort_index()


#Maximum number of bathrooms=14
#Minimum number of bathrooms= 1

bath_min_list=[]
bath_max_list=[]

#By having a wide range and with a few visual data, we obtain the value that most represents the number of bathrooms in the properties and the IQR
#IQR= The interquartile range (IQR) is the difference between the 75th percentile (0.75 quantile) and the 25th percentile (0.25 quantile) 
#The IQR can be used to detect outliers in the data

unique= data_4["property_type"].unique() # We call as "unique" variable the values ​​of data set 4 attributed to "property_type"

for x in unique:

    Q1 = data_4[data_4["property_type"]==x]["bathrooms"].quantile(0.25)
    Q3 = data_4[data_4["property_type"]==x]["bathrooms"].quantile(0.75)
    IQR = Q3 - Q1
    bath_min = data_4[data_4["property_type"]==x]["bathrooms"].quantile(0.01) #We could have defined it as bath_min = (Q1 - 1.5 * IQR) but we use 
                                                                              #the quantile 0.01 is to avoid having negative values ​​in the minimum bath
    bath_max = Q3 + (IQR*1.5)    #It is a new range of decision that we choose to see the outliers// Upper Bound
                                 #When the scale is taken as 1.5, according to the IQR method, any data that is beyond 2.7σ from the mean (μ), anywhere, 
                                 #will be considered an outlier. And this decision range is the closest to what the Gaussian Distribution tells us, that is, 3σ
    
    bath_min_list.append (data_4[data_4["property_type"]==x]["bathrooms"].quantile(0.01)) #We name the list we had before with the new values
    bath_max_list.append (Q3 + (IQR*1.5)) #The append () method of the FormData interface appends a new value onto an existing key inside a FormData object
    
    print(x)
    print("el numero  maximo de bathrooms es {}, el numero minimo de bathrooms es {} y el IQR {}" .format(bath_max,bath_min,IQR))
    print("-------------------------------------------------------------------")

#We see that the values ​​are not integers, so we use the floor, to raise them to the nearest integer
#We do not use ceil since in department and PH we would be working with non-representative numbers.

for i in bath_max_list:
    floor_max=math.floor(i)
    print(floor_max)


#I am adjusting the new values ​​obtained to the variables we had

dpto_New=data_4[data_4["property_type"]=="Departamento"]
ph_New=data_4[data_4["property_type"]=="PH"]
casa_New=data_4[data_4["property_type"]=="Casa"]


#I define the number of bathrooms in the properties less than the maximum value and greater than the minimum value.

dpto_New= dpto_New[(dpto_New.bathrooms <= 3.0) & (dpto_New.bathrooms >=1.0) ]
ph_New=ph_New[(ph_New.bathrooms <= 3.0) & (ph_New.bathrooms >=1.0) ]
casa_New=casa_New[(casa_New.bathrooms <= 4.0) & (casa_New.bathrooms >=1.0) ]

#We see the relationship between property type and price by making a box plot using our IQR obtained

figure,(ax1,ax2,ax3) = plt.subplots(3,constrained_layout=True,figsize=(10,10)) #define the number of rows and columns of the subplot grid 
                                                                               #The restrained design perfectly fits the graphics within your figure
                                                                               #We give you the size of the figure

ax1.set_title("Relacion tipo de propiedad y bathrooms")
ax2.set_title("Relacion tipo de propiedad y bathrooms")
ax3.set_title("Relacion tipo de propiedad y bathrooms")

sns.boxplot(data=dpto_New,x="bathrooms",y="property_type",ax=ax1) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=ph_New,x="bathrooms",y="property_type",ax=ax2)   #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=casa_New,x="bathrooms",y="property_type",ax=ax3) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it

ax1.grid()
ax2.grid()
ax3.grid()

plt.show()

#To get the output, the left numeric column in ascending order

data_4["bedrooms"].value_counts().sort_index()

#Maximum number of bedrooms=15
#Minimum number of bedrooms= 0

bed_min_list=[]
bed_max_list=[]

#We obtain the maximum and minimum number of bedrooms in the properties and the IQR

unique= data_4["property_type"].unique() # We call as "unique" variable the values ​​of data set 4 attributed to "property_type"


#IQR= The interquartile range (IQR) is the difference between the 75th percentile (0.75 quantile) and the 25th percentile (0.25 quantile) 
#The IQR can be used to detect outliers in the data

for x in unique:

    Q1 = data_4[data_4["property_type"]==x]["bedrooms"].quantile(0.25)
    Q3 = data_4[data_4["property_type"]==x]["bedrooms"].quantile(0.75)
    IQR = Q3 - Q1
    bed_min = data_4[data_4["property_type"]==x]["bedrooms"].quantile(0.01) #We could have defined it as bed_min = (Q1 - 1.5 * IQR) but we use 
                                                                            #the quantile 0.01 is to avoid having negative values ​​in the minimum bed
    bed_max = Q3 + (IQR*1.5)        #It is a new range of decision that we choose to see the outliers// Upper Bound
                                    #When the scale is taken as 1.5, according to the IQR method, any data that is beyond 2.7σ from the mean (μ), anywhere, 
                                    #will be considered an outlier. And this decision range is the closest to what the Gaussian Distribution tells us, that is, 3σ
    
    bed_min_list.append(data_4[data_4["property_type"]==x]["bedrooms"].quantile(0.01))  #We name the list we had before with the new values
    bed_max_list.append(Q3 + (IQR*1.5))  #The append () method of the FormData interface appends a new value onto an existing key inside a FormData object
    
    print(x)
    print("el numero maximo de bedrooms es {}, el numero minimo de bedrooms es {} y el IQR {}" .format(bed_max,bed_min,IQR))
    print("-------------------------------------------------------------------")


#We see that the values ​​are not integers, so we use the floor, to raise them to the nearest integer
#We do not use ceil since in department and PH we would be working with non-representative numbers

for i in bed_max_list:
    floor_max=math.floor(i)
    print(floor_max)

#I am adjusting the new values ​​obtained to the variables we had

dpto_New=data_4[data_4["property_type"]=="Departamento"]
ph_New=data_4[data_4["property_type"]=="PH"]
casa_New=data_4[data_4["property_type"]=="Casa"]

#I define the number of bedrooms in the properties less than the maximum value and greater than the minimum value

dpto_New= dpto_New[(dpto_New.bedrooms <= 3.0) & (dpto_New.bedrooms >=1.0) ]
ph_New=ph_New[(ph_New.bedrooms <= 4.0) & (ph_New.bedrooms >=1.0) ]
casa_New=casa_New[(casa_New.bedrooms <= 5.0) & (casa_New.bedrooms >=1.0) ]

#We see the relationship between property type and price by making a box plot using our IQR obtained

figure,(ax1,ax2,ax3) = plt.subplots(3,constrained_layout=True,figsize=(10,10)) #define the number of rows and columns of the subplot grid 
                                                                               #The restrained design perfectly fits the graphics within your figure
                                                                               #We give you the size of the figure

ax1.set_title("Relacion tipo de propiedad y bedrooms")
ax2.set_title("Relacion tipo de propiedad y bedrooms")
ax3.set_title("Relacion tipo de propiedad y bedrooms")

sns.boxplot(data=dpto_New,x="bedrooms",y="property_type",ax=ax1) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=ph_New,x="bedrooms",y="property_type",ax=ax2)   #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=casa_New,x="bedrooms",y="property_type",ax=ax3) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it

ax1.grid()
ax2.grid()
ax3.grid()

plt.show()



#Maximum number of surface_covered_
#Minimum number of surface_covered 

surface_min_list=[]
surface_max_list=[]


#IQR= The interquartile range (IQR) is the difference between the 75th percentile (0.75 quantile) and the 25th percentile (0.25 quantile) 
#The IQR can be used to detect outliers in the data

unique= data_4["property_type"].unique() #We call as "unique" variable the values ​​of data set 4 attributed to "property_type"

for x in unique:

    Q1 = data_4[data_4["property_type"]==x]["surface_covered"].quantile(0.25)
    Q3 = data_4[data_4["property_type"]==x]["surface_covered"].quantile(0.75)
    IQR = Q3 - Q1
    surface_min = data_4[data_4["property_type"]==x]["surface_covered"].quantile(0.01) #We could have defined it as bed_min = (Q1 - 1.5 * IQR) but we use 
                                                                                       #the quantile 0.01 is to avoid having negative values ​​in the minimum bed
    surface_max = Q3 + (IQR*1.5)    #It is a new range of decision that we choose to see the outliers// Upper Bound
                                    #When the scale is taken as 1.5, according to the IQR method, any data that is beyond 2.7σ from the mean (μ), anywhere, 
                                    #will be considered an outlier. And this decision range is the closest to what the Gaussian Distribution tells us, that is, 3σ
    
    
    surface_min_list.append(data_4[data_4["property_type"]==x]["surface_covered"].quantile(0.01)) #We name the list we had before with the new values
    surface_max_list.append(Q3 + (IQR*1.5))  #The append () method of the FormData interface appends a new value onto an existing key inside a FormData object
    
    print(x)
    print("el maximo de surface covered es {}, el minimo de surface covered es {} y el IQR {}" .format(surface_max,surface_min,IQR))
    print("-------------------------------------------------------------------")


#I am adjusting the new values ​​obtained to the variables we had

dpto_New=data_4[data_4["property_type"]=="Departamento"]
ph_New=data_4[data_4["property_type"]=="PH"]
casa_New=data_4[data_4["property_type"]=="Casa"]


#I define the number of surface_covered in the properties less than the maximum value and greater than the minimum value

dpto_New= dpto_New[(dpto_New.surface_covered <= 137.0) & (dpto_New.surface_covered >=26.0) ]
ph_New=ph_New[(ph_New.surface_covered <= 201.0) & (ph_New.surface_covered >=35.0) ]
casa_New=casa_New[(casa_New.surface_covered <= 420.5) & (casa_New.surface_covered >=56.33) ]


#We see the relationship between property type and price by making a box plot using our IQR obtained

figure,(ax1,ax2,ax3) = plt.subplots(3,constrained_layout=True,figsize=(10,10))  #define the number of rows and columns of the subplot grid 
                                                                                #The restrained design perfectly fits the graphics within your figure
                                                                                #We give you the size of the figure

ax1.set_title("Relacion tipo de propiedad y surface_covered")
ax2.set_title("Relacion tipo de propiedad y surface_covered")
ax3.set_title("Relacion tipo de propiedad y surface_covered")

sns.boxplot(data=dpto_New,x="surface_covered",y="property_type",ax=ax1) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=ph_New,x="surface_covered",y="property_type",ax=ax2)   #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it
sns.boxplot(data=casa_New,x="surface_covered",y="property_type",ax=ax3) #box plot // Dataset for plotting // Returns the Axes object with the plot drawn onto it

ax1.grid()
ax2.grid()
ax3.grid()

plt.show()

# We join the filtered dataset and create a new variable called "data_5"

conc_2=[dpto_New,ph_New,casa_New]
data_5= pd.concat(conc_2)
data_5.head(10)
print(data_5.head(10))

#I use Seaborn's plot function
#Easily display the relationship between data to spot trends and patterns
#This function will create an axis grid so that each numeric variable data is shared between the "Y" axes in a single row and the "X" axes in a single column
#Each column is a variable and each row is an observation

graph_f=sns.pairplot(data_5,hue="property_type") # data_5 = Ordered data frame (long format) where each column is a variable and each row is an observation
                                                 # hue="property_type" = Order of Variable Tone Levels in the Palette
graph_f.fig.set_size_inches(16,8) #These parameters are the (width, height) of the figure in inches
plt.grid()
plt.show()

#we study the correlation between all the variables and see which would be the best to predict the price

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,8)) #A unique identifier for the figure // Width, height in inches
fig.tight_layout(pad=7.0) #tight_layout automatically adjusts the subframe parameters so that the subframe (s) fit within the figure area


sns.heatmap(data_5.corr(), annot = True,vmin=-1, vmax=1, center= 0,cmap= 'coolwarm',linewidths=3, linecolor='black',ax=ax1) # data_5.corr () = 2D data set that can be converted to a ndarray. If provided
                                                                                                                            # a Pandas DataFrame, the index / column info will be used to label the columns and rows
                                                                                                                            # annot = True = If true, write the data value in each cell. If it is a matrix with the same shape as data, use it to annotate the heatmap in place of the data
                                                                                                                            # vmin=-1, vmax=1 = Values ​​to anchor the colormap; otherwise they are inferred from the data and other keyword arguments
                                                                                                                            # cmap= 'coolwarm' = The mapping of data values ​​to color space
                                                                                                                            # linewidths=3 = chart line width
                                                                                                                            # linecolor='black' = Color of the lines that will divide each cell
                                                                                                                            # ax=ax1 = Axes on which to draw the plot


#A correlation heat map is a heat map that displays a 2D correlation matrix between two discrete dimensions, using colored cells to represent data on a generally monochrome scale

matrix = np.triu(data_5.corr()) #Upper triangle of a matrix
                                #Returns a copy of an array with the elements below the k-th diagonal at zero

sns.heatmap(data_5.corr(), annot=True, mask=matrix,ax=ax2) # The data here must be passed with the corr () method to generate a correlation heatmap. Also, corr () itself removes columns that will 
                                                           # not be useful while generating a correlation heatmap and selects those that can be used
                                                           # annot = True = If true, write the data value in each cell. If it is a matrix with the same shape as data, use it to annotate the heatmap in place of the data
                                                           # mask = matrixReplace values where the condition is True
                                                           #ax=ax2 = Axes on which to draw the plot
plt.show()

####### EL MEJOR ATRIBUTO PARA PREDECIR EL PRECIO ES .............










from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from  xgboost import XGBRegressor

#Primero definiremos nuestros datos de entrenamiento y de prueba.
model = []
score = []

x_train, x_test, y_train, y_test = train_test_split(data_5.drop(["property_type"],axis=1),data["property_type"],test_size=0.2,random_state=0)

print("X Train Shape", x_train.shape)
print("Y Train Shape", y_train.shape)
print("X Test Shape", x_test.shape)
print("Y Test Shape", y_test.shape)


#Regresión lineal
linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
linear_model_predict = linear_model.predict(x_test)
print("Score: ",r2_score(linear_model_predict,y_test))
model.append("Multi Linear Regression")
score.append(r2_score(linear_model_predict,y_test))


#Arbol de desiciones
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train,y_train)
tree_reg_predict = tree_reg.predict(x_test)
print("Score: ",r2_score(tree_reg_predict,y_test))
model.append("Decision Tree Regression")
score.append(r2_score(tree_reg_predict,y_test))



#
lasso_model = Lasso()
lasso_model.fit(x_train,y_train)
lasso_model_predict = lasso_model.predict(x_test)
print("Score: ",r2_score(lasso_model_predict,y_test))
model.append("Lasso Regression")
score.append(r2_score(lasso_model_predict,y_test))


#Red elastica.
elasticnet_model = ElasticNet()
elasticnet_model.fit(x_train,y_train)
elasticnet_model_predict = elasticnet_model.predict(x_test)
print("Score: ",r2_score(elasticnet_model_predict,y_test))
model.append("Elastic Net Regression")
score.append(r2_score(elasticnet_model_predict,y_test))


#Random Forest
reg = RandomForestRegressor(n_estimators=100, random_state = 42)
reg.fit(x_train,y_train)
reg_predict = reg.predict(x_test)
print("Score: ",r2_score(reg_predict,y_test))
model.append("Random Forest Regression")
score.append(r2_score(reg_predict,y_test))


#Ada boost
reg_ada = AdaBoostRegressor(random_state=0, n_estimators=5)
reg_ada.fit(x_train,y_train)
reg_ada_predict = reg_ada.predict(x_test)
print("Score: ",r2_score(reg_ada_predict,y_test))
model.append("Ada Boost Regression")
score.append(r2_score(reg_ada_predict,y_test))


#Gradient boost
reg_gb = GradientBoostingRegressor(
    n_estimators = 51
)
reg_gb.fit(x_train,y_train)
reg_gb_predict = reg_gb.predict(x_test)
print("Score: ",r2_score(reg_gb_predict,y_test))
model.append("Gradient Boosting Regression")
score.append(r2_score(reg_gb_predict,y_test))


#XGB
model_params = {}
reg_xgb = XGBRegressor(**model_params)
reg_xgb.fit(x_train,y_train)
reg_xgb_predict = reg_xgb.predict(x_test)
print("Score: ",r2_score(reg_xgb_predict,y_test))
model.append("XGBoost Regression")
score.append(r2_score(reg_xgb_predict,y_test))


plt.subplots(figsize=(15, 5))
sns.barplot(x=score,y=model,palette = sns.cubehelix_palette(len(score)))
plt.xlabel("Score")
plt.ylabel("Regression")
plt.title('Regression Score')
plt.show()


