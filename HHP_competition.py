import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

# this function takes the drugcount dataframe as input and output a tuple of 3 data frames: DrugCount_Y1,DrugCount_Y2,DrugCount_Y3
def process_DrugCount(drugcount):
    dc_map = {'1' : 1, '2':2, '3':3, '4':4, '5':5, '6':6, '7+' : 7}
    drugcount['DrugCount'] = drugcount.DrugCount.map(dc_map)
    drugcount['DrugCount'] = drugcount.DrugCount.astype(int)
    dc = drugcount.groupby(drugcount.Year, as_index=False)
    DrugCount_Y1 = dc.get_group('Y1')
    DrugCount_Y2 = dc.get_group('Y2')
    DrugCount_Y3 = dc.get_group('Y3')

    return (DrugCount_Y1,DrugCount_Y2,DrugCount_Y3)

# converts strings from "1- 2 month" to "1_2"
def replaceMonth(string):
    converted_string = ""
    map_of_format = {'0- 1 month' : "0_1", "1- 2 months": "1_2", "2- 3 months": "2_3", "3- 4 months": '3_4', "4- 5 months": "4_5", "5- 6 months": "5_6", "6- 7 months": "6_7", \
                   "7- 8 months" : "7_8", "8- 9 months": "8_9", "9-10 months": "9_10", "10-11 months": "10_11", "11-12 months": "11_12"}
    converted_string = string.map(map_of_format)

    return converted_string

def dummy_coding(df, variable,  dummy_na=False, drop_first = False, prefix=None):
    if prefix==None:
        prefix = variable
    outputdata = pd.get_dummies(df, columns=[variable], prefix= prefix, dummy_na=dummy_na, drop_first=drop_first)
    return outputdata
    
# this function processes a yearly drug count data
def process_yearly_DrugCount(aframe):
    processed_frame = None
    #print aframe.columns
    #print aframe
    aframe.drop("Year", axis = 1, inplace = True)
    after_month_conversion = aframe[['DSFS']].apply(replaceMonth)
    #print 'Count: ',aframe['DSFS'].unique()
    df = dummy_coding(after_month_conversion, "DSFS", dummy_na=True, drop_first=True) # you set drop_na=True only when are certain this variable contains missing values

    #gd = pd.get_dummies(after_month_conversion)
    joined =  pd.concat([aframe, df], axis = 1)
    joined.drop("DSFS", axis = 1, inplace = True)
    joined_grouped = joined.groupby("MemberID", as_index = False)
    joined_grouped_agg = joined_grouped.agg(np.sum)

    # Rename DrugCount to Total_DrugCount
    processed_frame = joined_grouped_agg.rename(columns={'DrugCount': 'Total_DrugCount'})
    return processed_frame

# run linear regression. You don't need to change the function
def linear_regression(train_X, test_X, train_y, test_y):
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_y)
    print 'Coefficients: \n', regr.coef_
    pred_y = regr.predict(test_X) # your predicted y values
    # The root mean square error
    mse = np.mean( (pred_y - test_y) ** 2)
    import math
    rmse = math.sqrt(mse)
    print ("RMSE: %.2f" % rmse)
    from sklearn.metrics import r2_score
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

# for a real-valued variable, replace missing with median
def process_missing_numeric(df, variable):
    # below is the code I used in the lecture ("exploratory_analysis.py") for dealing with missing values of the variable "age".
    # You need to change the code below slightly
    df[variable] = np.where(df[variable].isnull(),1,0)
    medianVar = df[variable].median()
    df[variable].fillna(medianVar, inplace= True)

# This function prints the ratio of missing values for each variable. You don't need to change the function
def print_missing_variables(df):
    for variable in df.columns.tolist():
        percent = float(sum(df[variable].isnull()))/len(df.index)
        print variable+":", percent

def main():
    pd.options.mode.chained_assignment = None # remove the warning messages regarding chained assignment. 
    daysinhospital = pd.read_csv('/Users/aakanksha/Downloads/INFS772_Ass3/HHP_release3/DaysInHospital_Y2.csv') 
    drugcount = pd.read_csv('/Users/aakanksha/Downloads/INFS772_Ass3/HHP_release3/DrugCount.csv') 
    #print drugcount.columns
    li = map(process_yearly_DrugCount, process_DrugCount(drugcount))
    DrugCount_Y1_New = li[0]
    Master_Assn3 = None
    
    # your code here to create Master_Assn3 by merging daysinhospital and DrugCount_Y1_New
    Master_Assn3 = pd.merge(daysinhospital, DrugCount_Y1_New, on='MemberID', how='left')
    # Master_Assn3 = DrugCount_Y1_New + daysinhospital use left join
    
    process_missing_numeric(Master_Assn3, 'Total_DrugCount')
    # Your code here for deal with missing values of the dummy variables. Please don't overthink this. 
    Master_Assn3 = Master_Assn3.fillna(0)
    # Your dode here to drop the column 'MemberID'. You need to drop the column in place
    Master_Assn3.drop("MemberID", axis = 1, inplace = True)
    print Master_Assn3.shape
    print Master_Assn3.head(3)
    '''output:
    ClaimsTruncated  DaysInHospital  Total_DrugCount  DSFS_0_1  DSFS_10_11  \
    0                0               0              3.0       0.0         0.0   
    1                0               0              1.0       1.0         0.0   
    2                1               1             23.0       1.0         1.0   
    
       DSFS_11_12  DSFS_1_2  DSFS_2_3  DSFS_3_4  DSFS_4_5  DSFS_5_6  DSFS_6_7  \
    0         0.0       0.0       0.0       1.0       0.0       0.0       0.0   
    1         0.0       0.0       0.0       0.0       0.0       0.0       0.0   
    2         0.0       0.0       1.0       1.0       1.0       1.0       1.0   
    
       DSFS_7_8  DSFS_8_9  DSFS_9_10  DrugCount_missing  
    0       0.0       0.0        0.0                  0  
    1       0.0       0.0        0.0                  0  
    2       1.0       1.0        1.0                  0  
    '''
    dependent_var = 'DaysInHospital'
    # The next two lines of code creat a list of independent variable names.
    independent_var = Master_Assn3.columns.tolist()
    independent_var.remove(dependent_var)
    # next we split the data into training vs. test. 
    train_X, test_X, train_y, test_y= train_test_split(Master_Assn3[independent_var], Master_Assn3[dependent_var], test_size=0.3, random_state=123)
    print train_X.shape, test_X.shape, train_y.shape, test_y.shape
    linear_regression(train_X, test_X, train_y, test_y)
    '''outputs:
    (76038, 16)
    (53226, 15) (22812, 15) (53226L,) (22812L,)
    ['ClaimsTruncated', 'Total_DrugCount', 'DSFS_0_1', 'DSFS_10_11', 'DSFS_11_12', 'DSFS_1_2', 'DSFS_2_3', 'DSFS_3_4', 'DSFS_4_5', 'DSFS_5_6', 'DSFS_6_7', 'DSFS_7_8', 'DSFS_8_9', 'DSFS_9_10', 'DrugCount_missing']
    ['ClaimsTruncated', 'Total_DrugCount', 'DSFS_0_1', 'DSFS_10_11', 'DSFS_11_12', 'DSFS_1_2', 'DSFS_2_3', 'DSFS_3_4', 'DSFS_4_5', 'DSFS_5_6', 'DSFS_6_7', 'DSFS_7_8', 'DSFS_8_9', 'DSFS_9_10', 'DrugCount_missing']
    Coefficients: 
    [ 0.9318124   0.01375816 -0.06578017 -0.01265249 -0.02528982  0.02518371
      0.02925542 -0.01464571 -0.02716822  0.00613997  0.00668293 -0.05660316
     -0.03220839  0.0174422  -0.18013557]
    RMSE: 1.60
    R2 value: -23.80 # don't worry about the negative value. It simply means the model is bad
    '''

if __name__ == '__main__':
    main()



