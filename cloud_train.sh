#! /bin/bash

DATE=`date +"%Y%m%d%H%M%S"`
FILENAME="titanic_$DATE"
export BUCKET_NAME=keras-titanic-models
export JOB_NAME=$FILENAME
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1


gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --region $REGION \
    --runtime-version 1.4 \
    --module-name trainer.main \
    --config=config.yaml \
    --package-path ./trainer \
    -- \
    --train-file gs://$BUCKET_NAME/data \
    --dropout-one 0.2 \
    --epochs-one 100 \
    --units-one 8 \
    --rms-one 0.001 \
    --batchsize-one 1
