echo "Running 1D without Restart...."
python main.py --env FruitCollection1D \
              --log-interval 30  \
              --decay-rate 10 \
              --update-steps 50 \
              --replay-capacity 4000 \
              --train-episodes 5000 \
              --save-steps 1000 \
              --restart-epsilon-steps 0 \
              --result-path "./results/FruitCollection1D/nondecompose/no_restart" \
              --scenarios-path "./scenarios/FruitCollection1D.json" \
              --save

echo "Done"

echo "Running 1D with Restart..."

python main.py --env FruitCollection1D \
              --log-interval 30  \
              --decay-rate 10 \
              --update-steps 50 \
              --replay-capacity 4000 \
              --train-episodes 5000 \
              --save-steps 1000 \
              --restart-epsilon-steps 1000 \
              --result-path "./results/FruitCollection1D/nondecompose/restart" \
              --scenarios-path "./scenarios/FruitCollection1D.json" \
              --save
echo "Done"
