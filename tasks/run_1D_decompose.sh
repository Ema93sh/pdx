echo "Running 1D without Restart...."
python main.py --env FruitCollection1D \
               --decompose \
               --log-interval 30  \
               --decay-rate 100 \
               --update-steps 50 \
               --replay-capacity 4000 \
               --train-episodes 5000 \
               --save-steps 1000 \
               --restart-epsilon-steps 0 \
               --result-path "./results/FruitCollection1D/decompose/no_restart" \
               --scenarios-path "./scenarios/FruitCollection1D.json" \
               --save

echo "Done"

echo "Running 1D with Restart..."

python main.py --env FruitCollection1D \
               --decompose \
               --log-interval 30  \
               --decay-rate 100 \
               --update-steps 50 \
               --replay-capacity 4000 \
               --train-episodes 5000 \
               --save-steps 1000 \
               --restart-epsilon-steps 500 \
               --result-path "./results/FruitCollection1D/decompose/restart" \
               --scenarios-path "./scenarios/FruitCollection1D.json" \
               --save
echo "Done"
