#!/bin/bash
array=(eca_botnext26ts_256 eca_halonext26ts halo2botnet50ts_256 halonet26t sehalonet33ts sebotnet33ts_256 botnet26t_256 lambda_resnet26rpt_256 lambda_resnet26t lamhalobotnet50ts_256 vit_base_patch16_224 efficientnet_b0 resnet18 swin_base_patch4_window7_224 twins_pcpvt_base twins_svt_base levit_128 pit_b_224 deit_base_distilled_patch16_224 vit_tiny_patch16_224 cait_m36_384 coat_mini convit_base tnt_s_patch16_224)
for j in "${array[@]}"
do
	echo "python train.py data --model ${j} --num-classes 2 --pretrained  --epochs $1  --no-aug --batch-size $2 --output /home/rohrerc/model_zoo/transformers --experiment ${j} -j $3 --checkpoint-hist 1" >> train_commands.txt
done
