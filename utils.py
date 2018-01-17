import pandas as pd
from datetime import datetime


def predictions_to_csv(predictions, df_predict):
    # Get date for csv filename generation.
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate kaggle submission file.
    df_submit = pd.DataFrame({'PassengerId': df_predict.index.values,
                              'Survived': predictions})
    df_submit.to_csv('csv/%s_kaggle_titanic.csv' % date_str, index=False)
