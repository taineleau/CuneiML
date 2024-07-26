python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data  --algorithm deepCORAL --seed 0 --lr 3e-05 --weight_decay 0 --coral_penalty_weight 0.1 --n_epochs 30 --log_dir ./logs/t23_iwildcam_deepCORAL_30_epoch_coral_penalty_weight_0.1 &&
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data  --algorithm deepCORAL --seed 0 --lr 3e-05 --weight_decay 0 --coral_penalty_weight 1.0 --n_epochs 30 --log_dir ./logs/t23_iwildcam_deepCORAL_30_epoch_coral_penalty_weight_1.0 &&
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data  --algorithm deepCORAL --seed 0 --lr 3e-05 --weight_decay 0 --coral_penalty_weight 10.0 --n_epochs 30 --log_dir ./logs/t23_iwildcam_deepCORAL_30_epoch_coral_penalty_weight_10.0

