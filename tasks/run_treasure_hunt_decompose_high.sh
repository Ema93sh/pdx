echo "Running with high Exploration"
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 500 \
             --target-update-frequency 50 \
             --update-frequency 2 \
             --replay-capacity 20000 \
             --train-episodes 700 \
             --save-steps 200 \
             --lr 0.00029 \
             --gamma 0.99 \
             --result-path "./results/TreasureHunter/decompose/high_explore" \
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --load-path ./results/TreasureHunter/decompose/optimal/TreasureHunter_decompose_.torch \
             --minimum-epsilon 1.0 
