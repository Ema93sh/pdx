echo "Running 2D without Restart...."
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 1  \
             --decay-rate 500 \
             --update-steps 100 \
             --replay-capacity 20000 \
             --train-episodes 5000 \
             --save-steps 1000 \
             --restart-epsilon-steps 0 \
             --result-path "./results/TreasureHunter/decompose/no_restart" \
             --save

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
