#!/bin/sh
while [ true ]
do
    gcloud compute --project "adaptation-based-programming" scp --zone "us-central1-c" pdx:~/pdx/results/TreasureHunter/decompose/explore ./cloud_results --recurse
    sleep 1000
done
