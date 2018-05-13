mkdir -p ./results/TreasureHunter/decompose/average/run$number

echo "Running with Replay $number" >> "./results/TreasureHunter/decompose/average/run$number/run.log"

python3 -u main.py --env TreasureHunter \
               --decompose \
               --log-interval 10  \
               --decay-rate 500 \
               --target-update-frequency 50 \
               --update-frequency 2 \
               --replay-capacity 50000 \
               --train-episodes 5000 \
               --save-steps 200 \
               --lr 0.00029 \
               --gamma 0.99 \
               --result-path "./results/TreasureHunter/decompose/average/run$number" \
               --scenarios-path "./scenarios/TreasureHunter_easy.json" \
               --starting-episilon 1 \
               --minimum-epsilon 0.1
