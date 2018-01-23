from datetime import datetime
from tensorflow.python.lib.io import file_io
import pandas as pd
import os

_PROJECT_ROOT = os.pardir

_CSV_CLOUD_PATH = 'gs://keras-titanic-models/data/predictions/'
_CSV_LOCAL_PATH = 'data/predictions/'


def predictions_to_csv(predictions, passenger_ids, cloud_train):
    # Get current time. Will be prepended to filename.
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Generate csv filename.
    filename = date_str + '_titanic_predictions'

    # Generate kaggle submission file.
    df_submit = pd.DataFrame({'PassengerId': passenger_ids,
                              'Survived': predictions})

    if not cloud_train:
        # Save predictions to local storage.
        df_submit.to_csv(os.path.join(_PROJECT_ROOT, _CSV_LOCAL_PATH, filename), index=False)
    else:
        # Save predictions to the cloud storage bucket's jobs directory.
        with file_io.FileIO(_CSV_CLOUD_PATH + filename, mode='w+') as input_f:
            df_submit.to_csv(input_f, index=False)
