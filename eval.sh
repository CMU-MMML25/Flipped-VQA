torchrun --nproc_per_node 1 eval.py --model 7B --max_seq_len 128 --batch_size 8 --epochs 5 \
--warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star --blr 9e-2 --weight_decay 0.16 \
--output_dir ./checkpoint/star --accum_iter 1 --vaq --qav --resume ./checkpoint/star.pth --device cuda