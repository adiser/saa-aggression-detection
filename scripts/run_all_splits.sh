python main.py aggression RGB data/cctv_train_split_0.txt data/cctv_test_split_0.txt --arch BNInception --lr 0.001 --lr_steps 10 15 --epochs 100 --gd 20 -b 15 -j 4 --pretrained_on_kinetics 1 --snapshot_pref cctv_0  
python main.py aggression RGB data/cctv_train_split_1.txt data/cctv_test_split_1.txt --arch BNInception --lr 0.001 --lr_steps 10 15 --epochs 100 --gd 20 -b 15 -j 4 --pretrained_on_kinetics 1 --snapshot_pref cctv_1
python main.py aggression RGB data/cctv_train_split_2.txt data/cctv_test_split_2.txt --arch BNInception --lr 0.001 --lr_steps 10 15 --epochs 100 --gd 20 -b 15 -j 4 --pretrained_on_kinetics 1 --snapshot_pref cctv_2
python main.py aggression RGB data/cctv_train_split_3.txt data/cctv_test_split_3.txt --arch BNInception --lr 0.001 --lr_steps 10 15 --epochs 100 --gd 20 -b 15 -j 4 --pretrained_on_kinetics 1 --snapshot_pref cctv_3 
python main.py aggression RGB data/cctv_train_split_4.txt data/cctv_test_split_4.txt --arch BNInception --lr 0.001 --lr_steps 10 15 --epochs 100 --gd 20 -b 15 -j 4 --pretrained_on_kinetics 1 --snapshot_pref cctv_4

