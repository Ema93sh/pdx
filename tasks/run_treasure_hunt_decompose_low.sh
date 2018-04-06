echo "Running with low Exploration"
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 500 \
             --target-update-frequency 400 \
             --update-frequency 6 \
             --replay-capacity 20000 \
             --train-episodes 2000 \
             --save-steps 500 \
             --result-path "./results/TreasureHunter/decompose/low_explore" \
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --lr 0.001 \
             --gamma 0.99 \
             --load-path ./results/TreasureHunter/linear/decompose/TreasureHunter_decompose_.torch \
             --starting-episilon 0.3 \
             --minimum-epsilon 0.3 \
             --save