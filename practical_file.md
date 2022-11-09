<center><span style="font-size:300%; font-weight:bold">Data Mining</span>
<br><span style="font-size:200%;">Mudit Aggarwal, 2020UCD2157</span>
</center>
<br>

<div style="font-size:180%">

S No. | Title | Sign
:---: | --- | ---
1 | Create Employee table in Weka |
2 | Build a Data Warehouse |
3 | Classification model in Weka |
4 | Apply Data pre-processing techinques on a dataset |
5 | Use Weka Knowledge Flow to preprocess data | 
6 | Data visualization/graph using WEKA | 
7 | Implement classification algorithms | 
8 | Implement Association mining algorithms |  

</div>

---

#### 1. Create an Employee table with the help of Data Mining Tool WEKA.

Description: We need to create an Employee table with training dataset, which includes attributes like name, id, salary, exprience, gender and phone number.

The generated arff file
```
@relation employee
@attribute name {x, y, z, a, b}
@attribute id numeric
@attribute salary {low, medium, high}
@attribure exp numeric
@attribute gender {male, female}
@attribute phone numeric

@data
first_name,id,salary,exp,gender,phone
Aldwin,1,high,15,male,372774
Marilin,2,medium,12,female,994465
Marlie,3,high,10,female,883628
Ravi,4,low,3,female,753785
Aindrea,5,medium,7,male,861704
Mona,6,medium,4,female,813501
Uta,7,medium,7,female,140606
Aloysia,8,medium,15,male,329518
Tammy,9,low,1,female,214974
Scotty,10,medium,9,female,070759
Melisandra,11,high,14,male,113473
Raimundo,12,low,8,male,498528
Nolly,13,low,13,male,666163
Jazmin,14,medium,10,female,296011
Cort,15,low,9,female,253710
Nariko,16,medium,11,female,107067
Bobbi,17,high,1,female,172437
Aubine,18,high,4,female,620682
Gregorius,19,low,4,male,697032
Hayes,20,medium,5,female,607779 
```

![](20221109021901.png)  

Data successfully opened and visualized in WEKA

---

#### 2. Build a Data Warehouse (Using open source tools like Pentaho Data Integration tool, MYSQL Server, Microsoft-SSIS, etc) 

Create a database in the SQL server

Server accessed using MYSQL Workbench

_Create database and add Dimension and Fact tables_
```
CREATE DATABASE customer_warehouse;
USE customer_warehouse;

CREATE TABLE Customer
(
id int PRIMARY KEY,
Name varchar(20) 
);

CREATE TABLE Van
(
id int PRIMARY KEY,
plate_number varchar(10)
);

CREATE TABLE Fact
(
transaction_no int PRIMARY KEY,
date DATE,
customer_id int,
van_id int,
FOREIGN KEY (customer_id) REFERENCES Customer(id),
FOREIGN KEY (van_id) REFERENCES Van(id)
);
```

_Insert Data into DImension Table from CSV file_
```
INSERT INTO Customer (id, Customer_name)
VALUES
(1,"Sarge"),
(2,"Shelton"),
(3,"Ettie"),
.
.
.
(99,"Vida"),
(100, "Alisa");

INSERT INTO Van
VALUES
(1,"VL Q3W 3506"),
(2,"KY 172 5405"),
.
.
.
(19,"EV F83 3361"),
(20,"HZ 4GN 1412");

```

Transaction data inserted into Fact table (This data can be inserted through a data source such as a monitoring system)

![](20221109020352.png)  

Final Data Warehouse Uses a Star Schema with a Fact Table and 3 dimenstions namely _Date, Customer and Van_

---

#### 3. Train a classification model in WEKA, explore different testing options


ARFF file for the importing iris data
```
@relation iris
@attribute id numeric
@attribute sepal_length numeric
@attribute sepal_width numeric numeric
@attribute petal_length numeric
@attribute petal_width numeric
@attribute species {Iris-setosa, Iris-versicolor, Iris-virginica}

@data
1,5.1,3.5,1.4,0.2,Iris-setosa
.
.
.
149,6.2,3.4,5.4,2.3,Iris-virginica
150,5.9,3.0,5.1,1.8,Iris-virginica
```


Imported the iris dataset in WEKA, preprocessed to remove attribute id

![](20221109024408.png)  

Training the model using different test options

_Using the data as training data for Naive Bayes classification algorithm_
(As we can see the accuracy on the training data is 100% as expected)
![](20221109024634.png)

_Using 10 fold cross validation (Most appropriate since the data is very small)_
![](20221109024803.png)  

_Using 75% percentage split to have 75% training data and rest is testing data_
![](20221109024855.png)  

Hence we have trained a Naive Bayes model for the iris dataset in WEKA

---

#### 4. Pre-process techniques on Dataset
Real-world databases are highly influenced by noise, missing and inconsistency due to their queue size so the data can be pre-processed to improve the quality of data and missing results and it also improves the efficiency.
There are 3 pre-processing techniques they are:                   
1) Add
2) Remove
3) Normalization                   

Pre-process a given dataset based on Handling Missing Values                   
Replacing missing attribute values by the attribute mean. This method is used for data sets with numerical attributes.


Using a sample database to test pre-processing techniques available in WEKA

CSV file used
(The file contains missing values across several attributes)
```
first_name,salary,height
Willie,679,1.28
Daryl,,1.99
Harmony,,1.08
Corabel,5871,1.43
Gwenni,,1.6
,2465,1.25
,1367,1.09
Alford,2261,1.19
Ilsa,2241,1.63
Gilemette,8458,1.55
Beret,2662,1.72
Randi,998,1.06
Nikolaos,2016,1.57
Ashien,317,1.3
Malcolm,8085,1.86
```

Imported and pre-profiling the data in WEKA

Missing values under the first_name attribute
![](20221109030658.png)  

Instances with missing values removed
![](20221109032102.png)  

Instances are missing in the salary column as well
![](20221109032216.png)  

The missing values are replaced by the mean of the attribute
![](20221109032422.png)  

---

#### 5. Use WEKA Knowledge flow to normalize dataset
(a). Normalize Weather Table data using Knowledge Flow

                   
The knowledge flow provides an alternative way to the explorer as a
graphical front end to WEKA’s algorithm. Knowledge flow is a
working progress. So, some of the functionality from explorer is not
yet available. So, on the other hand, there are things that can be
done in knowledge flow, but not in explorer. Knowledge flow
presents a dataflow interface to WEKA.

                   
(b). Normalize Employee Table data using Knowledge Flow


_(a)_ The dataset used 
```
@relation weather

@attribute outlook {sunny, overcast, rainy}
@attribute temperature real
@attribute humidity real
@attribute windy {TRUE, FALSE}
@attribute play {yes, no}

@data
sunny,85,85,FALSE,no
sunny,80,90,TRUE,no
overcast,83,86,FALSE,yes
rainy,70,96,FALSE,yes
rainy,68,80,FALSE,yes
rainy,65,70,TRUE,no
overcast,64,65,TRUE,yes
sunny,72,95,FALSE,no
sunny,69,70,FALSE,yes
rainy,75,80,FALSE,yes
sunny,75,70,TRUE,yes
overcast,72,90,TRUE,yes
overcast,81,75,FALSE,yes
rainy,71,91,TRUE,no
```

Created a dataflow to normalize the numeric values to be in between 0 and 1

![](20221109040737.png)  

Database before and after
![](20221109040838.png)  

_(b)_ Same action is done on employee table

![](20221109041024.png)  

Database before and after
![](20221109041150.png)  

---

#### 6. Visualization using WEKA
                   
WEKA's visualization allows you to visualize a 2-D plot of the current working
relation. Visualization is very useful in practice; it helps to determine the difficulty of the
learning problem. WEKA can visualize single attributes (1-d) and pairs of attributes
(2-d), and rotate 3-d visualizations (Xgobi-style).

(a) Visualization of Weather Table Dataset 

_(a)_ Weather dataset is visualized using the WEKA knowledgeflow tool

![](20221109041635.png)  

X axis denotes temperature, Y axis denotes humidity and colour represents the dependent varibale play. Red means no and blue means yes

![](20221109041620.png)

---

#### 7. Implement Classification algorithms
1. One R algorithm

```python
#Program to implement One R algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('weather.csv')
print(df.info())

target_col = "play"
nom_attr = ['outlook', 'windy', 'play']

for col in nom_attr:
	if col == target_col:
		continue

	prediction_table = df.groupby([col])[target_col].agg(pd.Series.mode)

	y_predicted = \
		[prediction_table[x][0] if isinstance(prediction_table[x], np.ndarray) 
								else prediction_table[x] for x in df[col]]
	accuracy = \
		sum([1 if x == y else 0 for x, y in 
				zip(y_predicted, list(df[target_col]))])/len(y_predicted)
	print(f"Accuracy with predictor {col} = {100*accuracy:.2f}%")
```

OUTPUT:
```
RangeIndex: 14 entries, 0 to 13
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   outlook      14 non-null     object
 1   temperature  14 non-null     int64 
 2   humidity     14 non-null     int64 
 3   windy        14 non-null     bool  
 4   play         14 non-null     object
dtypes: bool(1), int64(2), object(2)
memory usage: 590.0+ bytes
None
Accuracy with predictor outlook = 71.43%
Accuracy with predictor windy = 64.29%
```

<br>

2. Decision Tree algorithm

```python
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

df = pd.read_csv('weather.csv')

outlook = {'sunny': 0, 'overcast': 1, 'rainy': 2}
df['outlook'] = df['outlook'].map(outlook)

target_col = 'play'

print(df.info())

X, y = df.loc[:, df.columns!=target_col], df.loc[:, [target_col]]

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size=0.3, random_state=1) 

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))	
```

OUTPUT
```
RangeIndex: 14 entries, 0 to 13
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   outlook      14 non-null     int64 
 1   temperature  14 non-null     int64 
 2   humidity     14 non-null     int64 
 3   windy        14 non-null     bool  
 4   play         14 non-null     object
dtypes: bool(1), int64(3), object(1)
memory usage: 590.0+ bytes
None
Accuracy: 0.8
```

---