start1=$(date +%Y-%m-%d" "%H:%M:%S)
a=$(date +%s)
python train.py \
        --train_data_path data/train_270.txt \
        --test_data_path data/valid_30.txt \
        2>&1 | tee train.log
end=$(date +%Y-%m-%d" "%H:%M:%S)
b=$(date +%s)
c=$((($b-$a)))
echo $c
echo $start1
echo $end
