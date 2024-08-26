# Estimating the position of a non-cooperative spacecraft

The goal of this project is to estimate the pose of a non-cooperative spacecraft. To reach this objective, we created a full pipeline from image generation to pose estimation via neural network based image analysis. You can find more details in the project report: [report/report.pdf](report/report.pdf).

## Pipeline

### Dataset generation

#### Tango spacecraft
 - We created a Python script searching for the most axis aligned images of SPEED+ dataset: [model/search_ref.py](model/search_ref.py)
 - Running this script, we extracted image references: [model/references](model/references)
 - Using these images, we modeled Tango spacecraft in Blender: [model/tango.blend](model/tango.blend)
 - We defined 11 key-points, the 8 corners and the 3 antennas: [model/key_points.json](model/key_points.json)
 - We exported side-views of the mesh: [model/sides](model/sides)

#### Rendering process
 - We created a Blender file for the dataset generation: [dataset.blend](dataset.blend)
 - We wrote a Python script using Blender API to render images: [dataset.py](dataset.py)
 - We created another Python script to gather all the labels: [concat.py](concat.py)

### Image analysis
 - We implemented a ResNet-50 neural network in a Python Notebook: [ResNet-50.ipynb](ResNet-50.ipynb)

### Pose estimation
 - We wrote a Python Notebook using PnP algorithm for pose estimation: [PnP.ipynb](PnP.ipynb)



## Commands
We created a Makefile to simplify the commands: [Makefile](Makefile)
 - Download SPEED+ dataset: ```make speedplusv2```
 - Extract image references from SPEED+: ```make model/references```
 - Generate a dataset of 100 images: ```make dataset NB_IMAGES=100```
