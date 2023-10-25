import pandas as pd
from time import time, sleep, gmtime, strftime
import boto3, argparse, os, subprocess, sys

subprocess.call([sys.executable, '-m', 'pip', 'install', 'sagemaker'])
import sagemaker
from sagemaker import Session
from sagemaker.feature_store.feature_group import FeatureGroup

if __name__=='__main__':
    print('Beginning Feature Store Job')
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str)
    parser.add_argument('--bucket', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--role', type=str)
    parser.add_argument('--feature-group-name', type=str)
    parser.add_argument('--max-processes', type=int, default=3)
    parser.add_argument('--max-workers', type=int, default=4)
    
    args, _ = parser.parse_known_args()
    print('Received Arguments {}'.format(args))
    region = args.region
    bucket = args.bucket
    prefix = args.prefix
    execution_role = args.role
    fg_name = args.feature_group_name
    max_processes = args.max_processes
    max_workers = args.max_workers
    
    
    #get data and place eventtime and recordid columns for feature store
    data =  pd.read_csv('/opt/ml/processing/input/pca-reduced.csv')

    cur_time = int(round(time()))
    data['EventTime'] = pd.Series([cur_time] * len(data), dtype='float64')
    data['RecordID'] = pd.Series([i for i in range(len(data))], dtype='int')
    print('Data retrieved')

    # setup feature store session
    boto_session = boto3.Session(region_name = region)
    sagemaker_client = boto_session.client('sagemaker', region)
    fs_runtime = boto_session.client('sagemaker-featurestore-runtime', region)

    featurestore_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=fs_runtime
    )


    # create feature group with feature definitions supplied by the df
    fg = FeatureGroup(
        name = fg_name,
        sagemaker_session=featurestore_session
    )

    fg.load_feature_definitions(data_frame=data)

    # create feature group
    s3_uri = 's3://{}/bank_fraud/{}/'.format(bucket, prefix)
    
    print('Creating Feature Group')
    fg.create(
        s3_uri= s3_uri,
        record_identifier_name='RecordID',
        event_time_feature_name='EventTime',
        role_arn=execution_role,
        enable_online_store=False
    )

    while fg.describe().get('FeatureGroupStatus') != 'Created':
        print('.', end='')
        sleep(5)
    
    print('feature group created, ingesting data')

    # ingest data into feature group
    fg.ingest(
        data_frame=data, 
        max_processes=max_processes,
        max_workers=max_workers, 
        wait=True
    )
    print('Data Ingested')
    print('Waiting ten minutes for offline store to be filled')
    sleep(600)
    # saving the feature group name is good practice
    f = open('/opt/ml/processing/output/feature_group_name.txt', 'w+')
    f.write(fg_name)
    
    print('Feature Group Job Complete')