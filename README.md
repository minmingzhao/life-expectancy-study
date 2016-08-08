Topic: Big Data Case Study in Life Sciences
Concentration: Predict life expectancy of different countries

Problem Statement: This study is to investigate the relationship between the life expectancy at birth of different countries and many other factors such as country area, population, economy condition, climate, happiness index etc. With this study, we could conclude some important factors which should be taken seriously by each government in order to improve the life expectancy for certain countries. 

Data Set: 
Country name and general info: 
https://www.cia.gov/library/publications/the-world-factbook/ 

World Population: 
http://gsociology.icaap.org/data/WorldPopulation.xlsx 
http://www.census.gov/population/international/data/idb/informationGateway.php

Quality of Life:
http://gsociology.icaap.org/data/QualityOfLife_reconstructed.ods 

Indicator WorldBank GDP data
http://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD 

Life expectancy data:
http://www.gapminder.org/gapminder-world/documentation/gd004 

Hardware:
•	Windows 8.1 running Cloudera Quick VM
Software:
Technology/tools	Description
Hive/HDFS	Hadoop HDFS, Cloudera Hive SQL
PySpark w/ MLlib	Pyspark for machine learning
Anaconda Spyder w/ Scikit-learn	Scikit-learn machine learning

Overview of steps:
Step1: Identify independent variables for dependent variable life expectancy. Clean raw data. 
Step2: Import cleaned raw data to Hive and merge tables based on country names
Step3: Use Pyspark MLLib to do regression or decision tree to find relationship between life expectancy and all different variables. Use Scikit-learn to do regression or decision tree 
Step4: Visualize the results using python matplot lib.   

Summary: A good linear regression model is generated to predict life expectancy of each country. Some factors are listed and discussed as they are important to improve life expectancy. 
Cons/Issues: The factors of a country are various and some are not necessarily considered in this report and some may be not completely independent to each other. 
YouTube URLs here:
15mins: https://youtu.be/KJLHTcrihTY 
2mins: https://youtu.be/wqPIqF-s1_Q 
