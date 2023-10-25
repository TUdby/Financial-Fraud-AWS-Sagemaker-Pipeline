import sys, subprocess, argparse, boto3
from sklearn.model_selection import train_test_split

if 'sagemaker' not in sys.modules:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'sagemaker'])
    
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

if __name__=='__main__':
    print('Starting Athena Job')
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str)
    parser.add_argument('--bucket', type=str)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    region = args.region
    bucket = args.bucket
    

    # get feature group name file
    f = open('/opt/ml/processing/input/feature_group_name.txt')
    fg_name = f.read()
    
    
    # get feature group
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name='sagemaker')
    featurestore_client = boto_session.client(service_name='sagemaker-featurestore-runtime')
    
    session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_client
    )
    
    fg = FeatureGroup(
        name=fg_name,
        sagemaker_session=session
    )
    print('Retrieved feature group')
    
    # get athena query
    aquery = fg.athena_query()
    table = aquery.table_name


    # define columns 
    columns = '"'+'", "'.join([ feature['FeatureName'] for feature in fg.describe()['FeatureDefinitions'] ][:-2])+'"'
    
    # query dataset
    aquery.run(
        query_string = f'SELECT {columns} FROM "{table}"',
        output_location=f's3://{bucket}/bank_fraud/query_result'
    )

    aquery.wait()
    print('queried data')
    # get dataset, split, and write out
    data = aquery.as_dataframe()

    train, val = train_test_split(data, test_size=0.1)

    train.drop('isFraud', axis=1).to_csv(
        '/opt/ml/processing/rcf-data/rcf-data.csv',
        index=False,
        header=False
    )
    
    train.to_csv(
        '/opt/ml/processing/xgb-data/xgb-data.csv',
        index=False,
        header=False
    )

    # I leave the header on the val data
    val.to_csv(
        '/opt/ml/processing/val-data/val-data.csv',
        index=False,
        header=False
    )
    print('finished athena job')
