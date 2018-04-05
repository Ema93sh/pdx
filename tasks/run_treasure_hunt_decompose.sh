echo "Running 2D without Restart...."
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 500 \
             --target-update-frequency 400 \
             --update-frequency 6 \
             --replay-capacity 200000 \
             --train-episodes 5000 \
             --save-steps 500 \
             --restart-epsilon-steps 0 \
             --gamma 0.99 \
             --save \
             --lr 0.0009

             # --scenarios-path "./scenarios/TreasureHunter.json" \

echo "Done"

# echo "Running 2D with Restart..."
#
# python main.py --env TreasureHunter \
#              --decompose \
#              --log-interval 1  \
#              --decay-rate 2000 \
#              --update-steps 50 \
#              --replay-capacity 20000 \
#              --train-episodes 5000 \
#              --save-steps 1000 \
#              --restart-epsilon-steps 1000 \
#              --result-path "./results/TreasureHunter/decompose/restart" \
#              --scenarios-path "./scenarios/TreasureHunter.json" \
#              --save
# echo "Done"
