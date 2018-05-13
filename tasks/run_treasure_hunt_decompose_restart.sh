
echo "Running with exploration restart"
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 100 \
             --target-update-frequency 50 \
             --update-frequency 2 \
             --replay-capacity 20000 \
             --train-episodes 700 \
             --save-steps 200 \
             --restart-epsilon-steps 2000 \
             --lr 0.00029 \
             --gamma 0.99 \
             --starting-episilon 0.5 \
             --minimum-epsilon 0.1 \
             --result-path "./results/TreasureHunter/decompose/restart_explore" \
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --load-path ./results/TreasureHunter/decompose/optimal/TreasureHunter_decompose_.torch 

echo "Done"