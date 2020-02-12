py run_trainer.py -e 1 -b 32 -lr 0.01 -v 0.2 -inc 1 -out 2 -wf 64 -pad True -m BasicUNet -dp /datavol/brain_data/atlas/ --dataset atlas --op adam -s 0 -d True -g -gid 0 -freq 5 -ld /home/gnahum/stroke_detection/sd/results/runs/BasicUNet_b32_lr1e_2_in1_out2 ; 
py run_trainer.py -e 1 -b 32 -lr 0.01 -v 0.2 -inc 1 -out 2 -wf 64 -pad True -m 3DUNet -dp /datavol/brain_data/atlas/ --dataset atlas --op adam -s 0 -d True -g -gid 0 -freq 5 -ld /home/gnahum/stroke_detection/sd/results/runs/3DUNet_b32_lr1e_2_in1_out2;

