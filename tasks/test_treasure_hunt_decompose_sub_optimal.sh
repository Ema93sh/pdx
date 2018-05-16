echo "Running treasure hunt sub optimal"

python3 -u main.py --env TreasureHunter \
                   --decompose \
                   --log-interval 1  \
                   --load-path "./results/TreasureHunter/decompose/sub_optimal/TreasureHunter_decompose_.torch" \
                   --result-path "./results/TreasureHunter/decompose/sub_optimal_mse_summaries" \
                   --test


python3 plot_mse_summaries.py --path "./results/TreasureHunter/decompose/sub_optimal_mse_summaries"

echo "Done"
