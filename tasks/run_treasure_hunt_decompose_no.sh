echo "Running with no Exploration"
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
             --starting-episilon 0 \
             --minimum-epsilon 0 \
             --result-path "./results/TreasureHunter/decompose/no_explore" \
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --load-path ./results/TreasureHunter/decompose/optimal/TreasureHunter_decompose_.torch 