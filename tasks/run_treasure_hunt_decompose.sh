# echo "Running 2D without Restart...."
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

echo "Running 2D with Restart..."


# Training and saving a normal model
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 500 \
             --target-update-frequency 400 \
             --update-frequency 6 \
             --replay-capacity 20000 \
             --train-episodes 4000 \
             --save-steps 1000 \
             --restart-epsilon-steps 1000 \
             --result-path "./results/TreasureHunter/decompose/restart/normal_q" \
             --save

# Evaluating with Exploration
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 500 \
             --target-update-frequency 400 \
             --update-frequency 6 \
             --replay-capacity 20000 \
             --train-episodes 2000 \
             --save-steps 1000 \
             --restart-epsilon-steps 1000 \
             --result-path "./results/TreasureHunter/decompose/restart/post_learn_high_expo" \
             --scenarios-path "./scenarios/TreasureHunter.json" \
             --post_explore_init_episodes 0 \
             --post_train_explore

# Evaluating with Normal Q-Learning with low exploration
python main.py --env TreasureHunter \
             --decompose \
             --log-interval 10  \
             --decay-rate 500 \
             --target-update-frequency 400 \
             --update-frequency 6 \
             --replay-capacity 20000 \
             --train-episodes 2000 \
             --save-steps 1000 \
             --restart-epsilon-steps 1000 \
             --result-path "./results/TreasureHunter/decompose/restart/post_learn_low_expo" \
             --scenarios-path "./scenarios/TreasureHunter.json"
             --init_expo_rate 0.1

echo "Done"