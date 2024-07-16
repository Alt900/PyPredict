import statsmodels
import statsmodels.api as api
from . import np

from .API_Interface import data

class Standard_Measurements():
    def __init__(self,ticker,variable):
        self.ticker=ticker
        self.variable=variable

    def Central_Tendacy(self):
        return [
            data[self.ticker][self.variable].mean(),
            data[self.ticker][self.variable].median(),
            data[self.ticker][self.variable].mode()[0],]
                
    def standard_deviation(self):
        return data[self.ticker][self.variable].std()
    
class Ordinary_least_squares():
    def __init__(self,x,y,pval_threshold):
        self.pval_threshold=pval_threshold
        self.y=y#endog
        self.x=api.add_constant(x)#exog
        print("Running ordinary least squares model...")
        self.model=api.OLS(y,exog=x).fit()
        self.summary=self.model.summary()

        self.heteroskedasticity_tests={
            "whites": statsmodels.stats.diagnostic.het_white(self.model.resid, self.x),
            "breusch-pagan": statsmodels.stats.diagnostic.het_breuschpagan(self.model.resid, self.x),
            "goldfeld-quandt": statsmodels.stats.diagnostic.het_goldfeldquandt(self.y,self.x,)
        }

    def null_hypothesis(self):#if accepted, data shares no relation only by chance, conversely if rejected
        pvalue=self.model.pvalues[0]
        if pvalue>self.pval_threshold:
            print(f"Null hypothesis accepted, p-value is {pvalue-self.pval_threshold} greater than the threshold.")
            print("Data has no correlation")
            self.null_hypothesis=False#if the data shares relation this will be useful for forming alternative hypothesis H1
        else:
            print(f"Null hypothesis rejected, p-value is {self.pval_threshold-pvalue} less than the threshold.")
            self.null_hypothesis=True

    def diagnose_heteroskedasticity(self,test:str):
        return self.heteroskedasticity_tests[test.lower()]