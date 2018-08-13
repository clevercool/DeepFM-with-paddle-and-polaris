start1=$(date +%Y-%m-%d" "%H:%M:%S)
a=$(date +%s)
python infer.py --model_gz_path models/model-pass-9-batch-0.tar.gz --data_path data/test.txt  --prediction_output_path ./p1.txt
end=$(date +%Y-%m-%d" "%H:%M:%S)
b=$(date +%s)
c=$((($b-$a)))
echo $c
echo $start1
echo $end
