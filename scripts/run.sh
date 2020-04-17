python run_trainer.py -e 1 -b 1 -lr 0.01 -v 0.2 -inc 1 -out 2 -wf 64 -pad True -m BasicUNet -dp /datavol/brain_data/atlas/ --dataset atlas -op sgd -s 0 -d True -g -freq 5000; 
python run_trainer.py -e 1 -b 1 -lr 0.01 -v 0.2 -inc 1 -out 2 -wf 64 -pad True -m BasicUNet -dp /datavol/brain_data/atlas/ --dataset atlas -op adam -s 0 -d True -g -freq 5000; 
python run_trainer.py -e 1 -b 1 -lr 0.01 -v 0.2 -inc 1 -out 2 -wf 64 -pad True -m 3DUNet -dp /datavol/brain_data/atlas/ --dataset atlas -op sgd -s 0 -d True -g -freq 5000; 
