import pandas as pd
import numpy as np
from datetime import datetime


# Returns combined keras + xgboost average, but could be modified for weighted average in the future.
def average_results(keras_pred, xg_pred):
    results = np.mean(np.array([keras_pred, xg_pred]), axis=0)
    return results


def predictions_to_csv(predictions, passenger_df):
    # Round predictions into 0/1 ints.
    predictions = np.round(predictions).astype('int')

    # Combine PassengerIds with their prediction.
    passenger_predictions = [([passenger_df.index[prediction], predictions[prediction][0]]) for prediction in
                             range(len(predictions))]

    # Prepend date to csv and output.
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(passenger_predictions, columns=['PassengerId', 'Survived'])
    df.to_csv(('%s_kaggle_titanic.csv' % date_str), index=0)
