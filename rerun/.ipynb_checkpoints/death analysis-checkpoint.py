import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('/Users/lilimatic/stopah.csv')

#treat values as NA
for x in df.columns:
        if len(df[df[x] == -2147483648].index) >0:
            ix = df[df[x] == -2147483648].index
            imp = df[[x]].loc[~df.index.isin(ix)].mean()[0]
            df.loc[ix,x] = np.nan
            
#Gender

#All gender types
df['Gender'].unique()

#Number of male 
print('Male n='+str(len(df[['Gender']][df['Gender']==0])) + ' and percentage (%):' +str(round(len(df[['Gender']][df['Gender']==0])*100/len(df),2)))

#Number of female
print('Female n='+str(len(df[['Gender']][df['Gender']==1])) + ' and percentage (%):' +str(round(len(df[['Gender']][df['Gender']==1])*100/len(df),2)))


#Analysis of Death Cases             
                      
date_format = "%d/%m/%Y"

df['Initial.Admission.date'] = pd.to_datetime(df['Initial.Admission.date'], format='%d-%b-%y')
df['Date_of_death']          = pd.to_datetime(df['Date_of_death'], format='%d/%m/%y')
df['diff']                   = df['Date_of_death'] - df['Initial.Admission.date']
df['diff']                   = df['diff'].dt.days

df['diff'].hist()
df[['diff']].describe()

#Age 

df[['Age.at.randomisation..calc.']].describe()