import pandas as pd
from google.cloud import storage
import joblib
from TaxiFareModel.utils import simple_time_tracker

STORAGE_LOCATION = 'models/TaxiFare/model.joblib'
BUCKET_NAME = 'wagon-data-batch769-caiotizo'
AWS_BUCKET_PATH = f"gs://{BUCKET_NAME}/data/train_1k.csv"
LOCAL_PATH = "your_localpath"

DIST_ARGS = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")


@simple_time_tracker
def get_data(nrows=1000, local=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    if local:
        path = LOCAL_PATH
    else:
        path = AWS_BUCKET_PATH
    df = pd.read_csv(path, nrows=nrows)
    return df


def clean_df(df, test=False):
    """ Cleaning Data based on Kaggle test sample
    - remove high fare amount data points
    - keep samples where coordinate wihtin test range
    """
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == "__main__":
    params = dict(nrows=1000,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  )
    df = get_data(**params)
