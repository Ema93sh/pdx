# echo "Training Treasure Hunter "
# python main.py --env TreasureHunter \
#              --decompose \
#              --log-interval 10  \
#              --decay-rate 500 \
#              --target-update-frequency 400 \
#              --update-frequency 6 \
#              --replay-capacity 200000 \
#              --train-episodes 5000 \
#              --save-steps 1000 \
#              --restart-epsilon-steps 0 \
#              --gamma 0.99 \
#              --save \
#              --lr 0.0009
#
#              # --scenarios-path "./scenarios/TreasureHunter.json" \
#
# echo "Done"

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
             --load-path ./results/TreasureHunter\ /linear/decompose/TreasureHunter_decompose_.torch \
             --minimum-epsilon 1.0

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
             --load-path ./results/TreasureHunter\ /linear/decompose/TreasureHunter_decompose_.torch \
             --starting-episilon 0.3 \
             --minimum-epsilon 0.3

echo "Running with no Exploration"
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 500 \
             --target-update-frequency 400 \
             --update-frequency 6 \
             --replay-capacity 20000 \
             --train-episodes 2000 \
             --save-steps 500 \
             --result-path "./results/TreasureHunter/decompose/no_explore" \
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --load-path ./results/TreasureHunter\ /linear/decompose/TreasureHunter_decompose_.torch \
             --lr 0.001 \
             --gamma 0.99 \
             --starting-episilon 0 \
             --minimum-epsilon 0

echo "Running with exploration restart"
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 100 \
             --target-update-frequency 400 \
             --update-frequency 6 \
             --replay-capacity 20000 \
             --train-episodes 2000 \
             --save-steps 500 \
             --restart-epsilon-steps 2000 \
             --result-path "./results/TreasureHunter/decompose/restart_explore" \
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --load-path ./results/TreasureHunter\ /linear/decompose/TreasureHunter_decompose_.torch \
             --lr 0.001 \
             --gamma 0.99 \
             --starting-episilon 0.5 \
             --minimum-epsilon 0.1

echo "Done"
