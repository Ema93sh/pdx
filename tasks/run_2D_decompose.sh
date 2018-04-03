echo "Running 2D without Restart...."
python main.py --env FruitCollection2D \
              --decompose \
              --log-interval 1  \
              --decay-rate 2000 \
              --update-steps 50 \
              --replay-capacity 20000 \
              --train-episodes 5000 \
              --save-steps 1000 \
              --restart-epsilon-steps 0 \
              --result-path "./results/FruitCollection2D/decompose/no_restart" \
              --scenarios-path "./scenarios/FruitCollection2D.json" \
              --save

echo "Done"

echo "Running 2D with Restart..."

python main.py --env FruitCollection2D \
              --decompose \
              --log-interval 1  \
              --decay-rate 2000 \
              --update-steps 50 \
              --replay-capacity 20000 \
              --train-episodes 5000 \
              --save-steps 1000 \
              --restart-epsilon-steps 1000 \
              --result-path "./results/FruitCollection2D/decompose/restart" \
              --scenarios-path "./scenarios/FruitCollection2D.json" \
              --save
echo "Done"
