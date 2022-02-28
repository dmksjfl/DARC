#!/bin/bash

for ((i=1;i<6;i+=1))
do 
	python main.py --env Ant-v2 --save-model --policy DARC --dir './logs/DARC/Ant/r'$i --seed $i --qweight 0.25
	python main.py --env HalfCheetah-v2 --save-model --policy DARC --dir './logs/DARC/HalfCheetah/r'$i --seed $i --qweight 0.1
	python main.py --env Walker2d-v2 --save-model --policy DARC --dir './logs/DARC/Walker2d/r'$i --seed $i --qweight 0.1
	python main.py --env Hopper-v2 --save-model --policy DARC --dir './logs/DARC/Hopper/r'$i --seed $i --qweight 0.15
	python main.py --env BipedalWalker-v3 --save-model --policy DARC --dir './logs/DARC/BipedalWalker/r'$i --seed $i --qweight 0.4
	python main.py --env AntMuJoCoEnv-v0 --save-model --policy DARC --dir './logs/DARC/AntPybullet/r'$i --seed $i --qweight 0.2
	python main.py --env Walker2DMuJoCoEnv-v0 --save-model --policy DARC --dir './logs/DARC/Walker2dPybullet/r'$i --seed $i --qweight 0.15
	python main.py --env Humanoid-v2 --policy DARC --hidden-sizes 256,256 --batch-size 256 --actor-lr 3e-4 --critic-lr 3e-4 --save-model --steps 3000000 --dir './logs/DARC/Humanoid/r'$i --seed $i --qweight 0.05
done
