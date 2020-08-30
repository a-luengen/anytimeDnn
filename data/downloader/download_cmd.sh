#!/bin/bash

python3 ./downloader.py \
    -data_root /home/alex/Documents/pythonProjects/anytimeDnn/data/imagenet_images/unsplitted \
    -use_class_list True \
    -class_list n03590841 n13132338 n03000134 n02280649 n07720875 \
    -images_per_class 1000

#n12454705 n11786131 n02007284 n12291959 n11848479