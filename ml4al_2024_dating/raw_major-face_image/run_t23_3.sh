python examples/run_expt.py --dataset iwildcam --algorithm IRM --root_dir data --lr 1e-5 --weight_decay 0 --irm_lambda 1 --n_epochs 30 --seed 2 --log_dir ./logs/t23_front_IRM_lr_3e-5_weight_decay_0_seed_2_lr_1e-5_epochs_30 &&
python examples/run_expt.py --dataset iwildcam --algorithm IRM --root_dir data --lr 2e-5 --weight_decay 0 --irm_lambda 1 --n_epochs 30 --seed 2 --log_dir ./logs/t23_front_IRM_lr_3e-5_weight_decay_0_seed_2_lr_2e-5_epochs_30 &&
python examples/run_expt.py --dataset iwildcam --algorithm IRM --root_dir data --lr 3e-5 --weight_decay 0 --irm_lambda 1 --n_epochs 30 --seed 2 --log_dir ./logs/t23_front_IRM_lr_3e-5_weight_decay_0_seed_2_lr_3e-5_epochs_30 

