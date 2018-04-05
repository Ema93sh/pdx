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
# python main.py --env TreasureHunter \
#              --decompose \
#              --log-interval 10  \
#              --decay-rate 500 \
#              --target-update-frequency 400 \
#              --update-frequency 6 \
#              --replay-capacity 20000 \
#              --train-episodes 4000 \
#              --save-steps 1000 \
#              --restart-epsilon-steps 1000 \
#              --result-path "./results/TreasureHunter/decompose/restart/normal_q" \
#              --save
#              --replay-capacity 200000 \
#              --train-episodes 5000 \
#              --save-steps 500 \
#              --restart-epsilon-steps 0 \
#              --gamma 0.99 \
#              --save \
#              --lr 0.0009

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
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --post_explore_init_episode 0 \
             --post_train_explore \
             --lr 0.001 \
             --gamma 0.99 \
             --load-path "./results/TreasureHunter/linear/decompose/normal_q/TreasureHunter_decompose_.torch"

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
             --scenarios-path "./scenarios/TreasureHunter_easy.json" \
             --init_expo_rate 0.1 \
             --lr 0.001 \
             --gamma 0.99 \
             --load-path "./results/TreasureHunter/linear/decompose/normal_q/TreasureHunter_decompose_.torch"

echo "Done"