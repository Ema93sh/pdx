echo "Running with high Exploration"
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 500 \
             --target-update-frequency 400 \
             --update-frequency 6 \
             --replay-capacity 20000 \
             --train-episodes 2000 \
             --save-steps 500 \
             --lr 0.001 \
             --gamma 0.99 \
             --result-path "./results/TreasureHunter/decompose/high_explore" \
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --load-path ./results/TreasureHunter/linear/decompose/TreasureHunter_decompose_.torch \
             --minimum-epsilon 1.0 \
             --save
