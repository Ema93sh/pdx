echo "Training Treasure Hunter "
python main.py --env TreasureHunter \
               --decompose \
               --log-interval 10  \
               --decay-rate 500 \
               --target-update-frequency 300 \
               --update-frequency 6 \
               --replay-capacity 200000 \
               --train-episodes 5000 \
               --save-steps 1000 \
               --restart-epsilon-steps 0 \
               --gamma 0.99 \
               --lr 0.0009 \
               --result-path "./results/TreasureHunter/decompose/optimal" \
               --scenarios-path "./scenarios/TreasureHunter.json" \
               --save

echo "Done"
