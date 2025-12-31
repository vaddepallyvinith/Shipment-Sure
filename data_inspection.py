import pandas as pd

file_path = 'FeaturesImprovedDataSet.csv' 
try:
    df = pd.read_csv(file_path)
    print("Dataset successfully loaded into a Pandas DataFrame.")
    print(" ")
    print("The raw data set initially has 18 features")
    print(" ")
    print("Features and Data Types using pandas data frame")
    print(" ")
    #I used df.info() for finding the features and data types of data set
    df.info()

except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found.")
    print("Please make sure the file name and path are correct.")
