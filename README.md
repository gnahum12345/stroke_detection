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
   - 