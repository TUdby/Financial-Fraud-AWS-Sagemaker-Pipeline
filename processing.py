from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import PCA
import os
import numpy as np
import pandas as pd
import math

scaler = StandardScaler()

# These functions will be passed a single column to be processed by its own process

# this function takes a numerical column, imputes the mean into empty areas, and standard scales it
def scale_nums(series):
    series = series.fillna(series.mean())
    return pd.Series(np.transpose(scaler.fit_transform(pd.DataFrame(series)))[0])

# scale_cats takes a categorical dummy column and divides by the sqrt of the probability of the class
def scale_cats(series):
    series /= math.sqrt((sum(series) / len(series)))
    return series


if __name__=='__main__':
    print('Beginning Processing Job')
    
    # Get input data paths
    print('Getting and Formatting Input Data')
    identity_path = os.path.join('/opt/ml/processing/identity', 'train_identity.csv')
    transaction_path = os.path.join('opt/ml/processing/transaction', 'train_transaction.csv')
    
    # get tables and join together
    identity = pd.read_csv(identity_path)
    train_transaction = pd.read_csv(transaction_path)
    data = train_transaction.join(identity, on='TransactionID', lsuffix='', rsuffix='-identity')

    # seperate label from data
    label = data['isFraud']
    data = data.drop(['isFraud', 'TransactionID'], axis=1)

    # drop columns with too much nan vals
    data.dropna(thresh=data.shape[0]*0.15, axis=1, inplace=True)
    
    # get list of numerical vs categorical columns
    numericals = data.select_dtypes(include='number').columns
    categoricals = list(set(data.columns)-set(numericals))

    # we now want to move to FAMD (explained in the notebook) and will do so in parallel.
    # We get as many processes as is helpful. due to the large size of our 
    # dataset this should be bounded by the amount of cores
    num_processes = min(data.shape[1], cpu_count())
    
    print('Beginning FAMD Processing')
    # this first pool deals with numerical columns. We put them in a list and
    # map them to their own processes to be standard scaled. Then they are
    # reassembled and placed back into our dataframe
    with Pool(num_processes) as pool:

        seq = [data[col_name] for col_name in numericals]
        results = pool.map(scale_nums, seq)

        data[numericals] = pd.concat(results, axis=1)
    
    print('Finished Scaling Numerical Columns')
    
    # before moving to the categorical processing, we need to dummify our categories
    data = pd.get_dummies(data)
    
    print('Dummified Categoricals Columns')
    
    # this second pool takes the categorical dummy columns in a list and maps them to
    # processes to be scaled by their probability. They are then reassembled and 
    # placed back into our dataframe.
    with Pool(num_processes) as pool:

        cats = list(set(data.columns)-set(numericals))
        seq = [data[col_name] for col_name in cats]
        
        results = pool.map(scale_cats, seq)
        
        data[cats] = pd.concat(results, axis=1)
        
    print('Finished Scaling Categorical Columns')
    
    # Nan values in categorical columns are treated as their own class
    # by pd.get_dummys, and for the numericals we imputed the mean into
    # nan values. However, for any numerical column that was completely 
    # nan values, nothing was done and so it must be dropped before we do PCA
    data = data.drop(data.columns[data.isna().any()], axis=1)

    # now we can perform PCA
    pca = PCA(n_components = min(32, data.shape[1]))
    data = pca.fit_transform(data.values)
    
    print('Finished PCA Reduction')
    
    # append the labels to the data 
    data = pd.concat(
        [
            label,
            pd.DataFrame(data)
        ],
        axis=1
    )
    
    output_path = os.path.join(
        '/opt/ml/processing/output',
        'pca-reduced.csv'
    )
    
    # write processed data
    data.to_csv(output_path, index=False)
    
    print('Finished Processing')