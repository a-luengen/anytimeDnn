# Description of anytimeDnn:

Code for the Master-Thesis "Development of an interruptible DNN-Architecture for image classification with realtime constriants". 
("Entwicklung einer unterbrechbaren DNN-Architektur zur Bildklassifikation mit Echtzeitbedingungen")

# Goals of this Thesis:

- evaluate Layer-Skipping as a technique to dynamically adjust inferencing speed on ResNet-alike DNN architectures (namely ResNet and DenseNet variants)
- evaluate Multi-scaled Dense Networks in comparison to Layer-Skipping 


## Setup for development:
 
 1. `conda create -n capp python=3.8.5`

 2. `conda activate capp`

 3. `pip install -r requirements.txt`

 4. (optional) Ensure tests are running: `python -m unittest`