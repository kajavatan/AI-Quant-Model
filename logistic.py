print("Logistic Regression")
import numpy as np
import pandas as pd
#df = pd.read_excel('FundamentalDataWithRatios1.8.xlsx', sheet_name='DATA-TRANPOSED
df = pd.read_csv('finalData.csv')
#print(df.head())
#df.to_csv('finalData.csv')

inputData = df.iloc[:, :-1].copy()
#print(inputData.head())
targetData = df.iloc[:, -1:].copy()
#print(targetData.head())

inputTrain = inputData.iloc[:-2, 2:].copy() 
targetTrain = targetData.iloc[:-2].copy()
inputTest = inputData.iloc[-2:, 2:]
targetTest = targetData.iloc[-2:].copy()
#print(inputTrain)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import linear_model
import scipy.stats as stat

feature_name = inputTrain.columns.values

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.pinv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values
        
reg = LogisticRegression_with_p_values()

reg.fit(inputTrain, targetTrain)

# Same as above.
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()


p_values = reg.p_values
# We take the result of the newly added method 'p_values' and store it in a variable 'p_values'.
p_values = np.append(np.nan, np.array(p_values))
# We add the value 'NaN' in the beginning of the variable with p-values.
summary_table['p_values'] = p_values
# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable.
print(summary_table)

summary_table.to_csv('output.csv')

#import pickle
#pickle.dump(reg, open('pd_model.sav', 'wb'))
# Here we export our model to a 'SAV' file with file name 'pd_model.sav'.
# inputTest = inputTest.to_frame().T 

y_hat_test = reg.model.predict(inputTest)
print(y_hat_test)

y_hat_test_proba = reg.model.predict_proba(inputTest)
y_hat_test_proba = y_hat_test_proba[: ][: , 1]
# Here we take all the arrays in the array, and from each array, we take all rows, and only the element with index 1,
# that is, the second element.
# In other words, we take only the probabilities for being 1.
print(y_hat_test_proba)
