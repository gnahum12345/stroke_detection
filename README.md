# Stroke MRI Segmentation 

This repo contains all my code for Stroke MRI Segmentation. This work was done at UCLA under Professor Eran Halperin and Daniel Tward. 

## File Structure 

```
+---data/
|   +---.ipynb_checkpoints/
|   \---README.md
+---dependencies/
|   \---envs/
|       +---base.txt
|       +---base_check.txt
|       +---dunet.txt
|       +---main_env.yml
|       +---pt_tumor_seg.txt
|       \---tf_tumor_seg.txt
+---infra/
|   +---__init__.py
|   +---dataset.py
|   +---dl_trainer.py
|   +---global_utils.py
|   +---logger.py
|   +---losses.py
|   \---plot.py
+---models/
|   +---__init__.py
|   +---convs.py
|   \---unet.py
+---scripts/
|   +---notebooks/
|   |   +---.png
|   |   +---Foreground-Background.png
|   |   +---Package Test.ipynb
|   |   +---Training Network-2D.ipynb
|   |   +---Training Network.ipynb
|   |   +---Viewing Data.ipynb
|   |   +---scan85.npy
|   |   +---scan86.npy
|   |   +---stroke_prediction.png
|   |   \---stroke_prediction_unet.png
|   +---__init__.py
|   +---data_shape_atlas.py
|   +---move_directories.py
|   +---net3d.sh
|   +---new_mask.npy
|   +---new_mask2204.npy
|   +---notebook2script.py
|   +---run.py
|   +---run.sh
|   +---run_trainer.py
|   +---testing_dataset.py
|   +---testing_loop.py
|   +---testing_model.py
|   \---unzip_files.py
\---README.md
```
### Folders:
- `data`: A simple README where all the data has been found. 
- `dependencies`: The environment and all other dependencies 
- `infra`: Infrastructure to train, plot and test data and models. This contains: 
   - `losses.py`: A python file with all the different loss functions (`dice_general` works for all shapes)
   - `plot.py` : A way to render different types of MRI. 
   - `logger.py`: A way to use SummaryWriter and visualize results in tensorboard. 
   - `global_utils.py`: common functions used in plot and losses. 
   - `dl_trainer.py`: A class that will train any neural network. 
   - `datasets.py`: A file that contains all the dataset classes (`AtlasDataset`)
- `models`: A class that contains all the models used. 
   - `unet.py`: All the unets that are re-implemented. 
   - `conv.py`: A start to deformable convolutions. 
- `scripts`: A folder that contains all the scripts used. 
   - `run_trainer.py` is the main script that will train a given model. 
   - `notebooks`: A folder with all the jupyter notebooks. 
       - `View Data.ipynb`: A notebook that allows to visualize the datasets. 
       - `Train Network.ipynb`: A notebook that allows to train a 3D Unet. 
       - `Train Network 2d.ipynb` : A replica of Train Network notebook but to train for 2D. 
       - `Package Test.ipynb`: A notebook to test the package. 
       
       
### UNets Tested: 
| Model | Dice Score | Accuracy | Memory Consumption |
|------------------|------------|----------|--------------------|
| 2D U-Net | ~60 ± 20% | ~95% | Low |
| 3D U-Net | ~80 ± 20%  | ~97% | High |
| V Net  | ~80 ± 20% | ~97% | High |
| Deformable U-Net | ~55 ± 20% | ~97% | Medium |
| Attention U-Net | ~59 ± 20% | ~97% | Medium |


### Things to note: 
- The Deformable U-Net was done in `known_methods`. 
- The Attention Gate was also done in `known_methods`. 
- The `known_methods` is currently not displayed in this repository. 
- Deformable U-Net and Attention U-Net were run from pre-existing code and limited number of epochs. I assume that if we re-implemented it and ran for longer time, it will perform better. 

### Steps forward: 
- Combine Deformable U-Net with Attention and use Cosegementaion algorithms for cross validation. 

### How to Continue: 

#### Steps: 
1. clone this repo. 
2. look at  `View Data.ipynb` to see data. 
3. look at  `run_trainer.py` to see how models are trained. 
4. look at `train network.ipynb` to see how models are trained interactively. 
5. Run new models by modifying `run_trainer.py` to accept those models. 

## Good luck! 

**Contact Info**

If you have any question please feel free to contact me: 

- Email: [gnahum12345@berkeley.edu](mailto:gnahum12345@berkeley.edu)

- LinkedIn: [Gabriel Nahum](https://www.linkedin.com/in/gabrielnahum/)

- Github: [@gnahum12345](https://www.github.com/gnahum12345/)


