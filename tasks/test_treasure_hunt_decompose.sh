echo "Running treasure hunt"

if [ -z "$1" ]
  then
    echo "Requires arg load path"
    exit 0
fi

if [ -z "$2" ]
  then
    echo "Requires sleep time"
    exit 0
fi

python3 -u main.py --env TreasureHunter \
                   --decompose \
                   --log-interval 1  \
                   --load-path $1 \
                   --sleep $2 \
                   --test \
                   --render
echo "Done"
