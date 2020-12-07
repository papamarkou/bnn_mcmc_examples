#!/bin/bash

gcloud compute instances create \
  instance01 \
  --machine-type=c2-standard-4 \
  --zone=us-central1-b \
  --image-project=ubuntu-os-cloud \
  --image-family=ubuntu-2004-lts \
  --boot-disk-size=200GB \
  --no-restart-on-failure \
  --maintenance-policy=terminate \
  --metadata-from-file startup-script=gdev.sh
