import pandas as pd
import numpy as np
from datetime import datetime

#Analysis of Death Cases 

df = pd.read_csv('/Users/lilimatic/stopah.csv')

#treat values as NA
for x in df.columns:
        if len(df[df[x] == -2147483648].index) >0:
            ix = df[df[x] == -2147483648].index
            imp = df[[x]].loc[~df.index.isin(ix)].mean()[0]
            df.loc[ix,x] = np.nan
            
date_format = "%d/%m/%Y"

df['Initial.Admission.date'] = pd.to_datetime(df['Initial.Admission.date'], format='%d-%b-%y')
df['Date_of_death']          = pd.to_datetime(df['Date_of_death'], format='%d/%m/%y')
df['diff']                   = df['Date_of_death'] - df['Initial.Admission.date']
df['diff']                   = df['diff'].dt.days

df['diff'].hist()
df[['diff']].describe()