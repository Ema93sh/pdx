echo "Running treasure hunt"

if [ -z "$1" ]
  then
    echo "Requires arg load path"
    exit 0
fi

if [ -z "$2" ]
  then
    echo "Requires arg sleep time (0 for wait every step)"
    exit 0
fi


python3 -u main.py --env TreasureHunter \
                   --decompose \
                   --log-interval 1  \
                   --load-path $1 \
                   --result-path "./results/TreasureHunter/decompose/mse_summaries" \
                   --sleep $2 \
                   --render \
                   --test


python3 plot_mse_summaries.py --path "./results/TreasureHunter/decompose/mse_summaries"

echo "Done"
