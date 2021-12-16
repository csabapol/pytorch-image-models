#!/bin/bash
array=(levit_128 pit_b_224 deit_base_distilled_patch16_224 cait_m36_384 eca_botnext26ts_256 eca_halonext26ts halo2botnet50ts_256 halonet26t sehalonet33ts sebotnet33ts_256 botnet26t_256 lambda_resnet26rpt_256 lambda_resnet26t lamhalobotnet50ts_256)
for j in "${array[@]}"
do
	echo "python train.py data --model ${j} --num-classes 2 --pretrained  --epochs $1  --no-aug --batch-size $2 --output /home/rohrerc/model_zoo/transformers --experiment ${j} -j $3" >> train_missing_commands.txt
done
