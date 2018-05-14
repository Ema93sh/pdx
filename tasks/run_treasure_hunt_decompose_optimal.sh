mkdir -p ./results/TreasureHunter/decompose/optimal

echo "Running with Replay $number" >> "./results/TreasureHunter/decompose/optimal/run.log"

python3 -u main.py --env TreasureHunter \
                   --decompose \
                   --log-interval 1  \
                   --decay-rate 600 \
                   --target-update-frequency 60 \
                   --update-frequency 2 \
                   --replay-capacity 50000 \
                   --train-episodes 5000 \
                   --save-steps 2000 \
                   --lr 0.00029 \
                   --gamma 0.99 \
                   --result-path "./results/TreasureHunter/decompose/optimal" \
                   --starting-episilon 1 \
                   --save \
                   --minimum-epsilon 0.1 >> "./results/TreasureHunter/decompose/optimal/run.log"
echo "Done"
done
