python examples/run_expt.py --dataset iwildcam --algorithm ERM --root_dir data --lr 1e-5 --weight_decay 0 --n_epochs 30 --log_dir ./logs/t23_front_ERM_lr_1e-5_weight_decay_0_n_epochs_30 &&
python examples/run_expt.py --dataset iwildcam --algorithm ERM --root_dir data --lr 1e-5 --weight_decay 0.01 --n_epochs 30 --log_dir ./logs/t23_front_ERM_lr_1e-5_weight_decay_0.01_n_epochs_30 &&
python examples/run_expt.py --dataset iwildcam --algorithm ERM --root_dir data --lr 1e-5 --weight_decay 0.001 --n_epochs 30 --log_dir ./logs/t23_front_ERM_lr_1e-5_weight_decay_0.001_n_epochs_30 


