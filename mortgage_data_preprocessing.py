import pandas as pd
import numpy as np
import datetime as dt

#Since our time is measured in months, we define elapsed_months to build newer variables
def elapsed_months(start_date,end_date):
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    return num_months

#We need to import some files to add Macro Variables to our df
#30 Yr. Mortgage rates for every month since 2000s
int_rates = pd.read_excel('MacroVariables/FannieRates.xlsx')
#Monthly unemployment rate on a MSA level
unemp_msa_level = pd.read_excel('MacroVariables/UnempMSA.xlsx')
#Housing Price Index at MSA level
hpi_msa_level = pd.read_excel('MacroVariables/HPI_AT_3zip.xlsx',skiprows=4)

#Now we load the df we made with the data from Fannie Mae
loans = pd.read_csv('Combined_Data_Raw.csv')

#Change if you want to subsample the loans df. In my case I will, just for convienence sake
loans = loans.sample(frac=0.1,random_state=100)

#We need to transform these original variables into datetimes
loans['ORIG_DTE'] = pd.to_datetime(loans['ORIG_DTE'])
loans['LAST_DTE'] = pd.to_datetime(loans['LAST_DTE'])
loans = loans[(loans['ORIG_DTE']>start_date)]
loans = loans[(loans['LAST_DTE']<end_date)]

#Now we'll only leave the variables we are interested to work with
loans = loans[[
    'ORIG_RT',
    'ORIG_DTE',
    'PURPOSE',
    'LAST_DTE',
    'LAST_STAT',
    'OLTV',
    'DTI',
    'CSCORE_B',
    'ZIP_3',
    'MSA',
    'Fin_UPB',
    'SATO',
    'ORIG_AMT',
    'ORIG_TRM']]

#First let's rename the CSCORE_B column as FICO which is it's most common name
loans = loans.rename(columns={'CSCORE_B': 'FICO'})

#We'll also drop the U=Undefined cases for the Purpose column
loans = loans[loans['PURPOSE'] != 'U']

#And transform the Purpose variable into dummies
loans = pd.concat([loans,pd.get_dummies(loans['PURPOSE'],drop_first=True)],axis=1)

loans.drop('PURPOSE',axis=1,inplace=True)

#We also create the Loan Age variable to account for the age of the loan
loans['LoanAge'] = loans.apply(lambda x: elapsed_months(x.ORIG_DTE, x.LAST_DTE), axis=1)

#Now we add the Macro variables
#First the 30Yr rate
int_rates = int_rates.rename(columns={'Date': 'LAST_DTE'})
loans = pd.merge(loans,int_rates,how='left',on='LAST_DTE')
#Now we create the most important variable for predicting prepayment:incentive
loans['Incentive'] = loans['ORIG_RT']-loans['Yield']
loans.drop('Yield',axis=1,inplace=True)


#Now we add month, quarter and year variables to link the df with the Macro csv's
loans['Month'] = loans['LAST_DTE'].dt.month
loans['Quarter'] = loans['LAST_DTE'].dt.quarter
loans['Year'] = loans['LAST_DTE'].dt.year

#We'll also need these variables for the origination date
loans['OrigMonth'] = loans['ORIG_DTE'].dt.month
loans['OrigQuarter'] = loans['ORIG_DTE'].dt.quarter
loans['OrigYear'] = loans['ORIG_DTE'].dt.year

#Now we'll add the HPI Factor
hpi_msa_level.drop('Index Type',axis=1,inplace=True)
hpi_msa_level = hpi_msa_level.rename(columns={'Three-Digit ZIP Code': 'ZIP_3'})
#We now get the HPI Factor at the Last Date of the Mortgage on Zip level
loans = pd.merge(loans,hpi_msa_level,how='left',on=['Year','Quarter','ZIP_3'])
loans = loans.rename(columns={'Index (NSA)':'CurrentHPI'})

#We now get the HPI Factor at the Origination of the Mortgage on Zip level
loans = pd.merge(loans,hpi_msa_level.rename(columns={'Quarter': 'OrigQuarter','Year':'OrigYear'}),how='left',on=['OrigYear','OrigQuarter','ZIP_3'])
loans = loans.rename(columns={'Index (NSA)':'OrigHPI'})

#With these two new variables we create the HPI Factor
loans['HPIFactor'] = loans['CurrentHPI']/loans['OrigHPI']

#We now get the Unemployment Rate at the Last Date of the Mortgage on MSA level
loans = pd.merge(loans,unemp_msa_level,how='left',on=['Year','Month','MSA'])
loans = loans.rename(columns={'Unemployment Rate':'MSAUnempRate'})

#At last! We now only transform the objective variable into a dummy one
loans['Prep'] = np.where(loans['LAST_STAT']=='P',1, 0)

#To wrap it up, we delete the variables we no longer need
loans.drop(['ORIG_RT','ORIG_DTE','LAST_DTE','LAST_STAT','Fin_UPB','Quarter','Year','OrigMonth','OrigQuarter','OrigYear','CurrentHPI','OrigHPI'],axis=1,inplace=True)


#Given that we only have numeric data I'll just save the df to a .csv
#For more on that, check https://stackoverflow.com/questions/17098654/how-to-store-a-dataframe-using-pandas
loans.to_csv('loans_clean.csv')
