python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data  --algorithm deepCORAL --seed 0 --lr 3e-05 --weight_decay 0 --coral_penalty_weight 10.0 --n_epochs 30 --log_dir ./logs/t23_iwildcam_deepCORAL_30_epoch_seed_0 &&
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data  --algorithm deepCORAL --seed 1 --lr 3e-05 --weight_decay 0 --coral_penalty_weight 10.0 --n_epochs 30 --log_dir ./logs/t23_iwildcam_deepCORAL_30_epoch_seed_1 &&
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data  --algorithm deepCORAL --seed 2 --lr 3e-05 --weight_decay 0 --coral_penalty_weight 10.0 --n_epochs 30 --log_dir ./logs/t23_iwildcam_deepCORAL_30_epoch_seed_2

