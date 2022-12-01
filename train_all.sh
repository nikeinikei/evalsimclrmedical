#!/bin/bash

python pretrain.py --epochs 0 --pretrained           model_pretrained_imagenet.pt &&

python pretrain.py --no_dry_run --epochs 250 --pretrained           model_pretrained_imagenet_med.pt &&
python pretrain.py --no_dry_run --epochs 250                        model_pretrained_med.pt &&

python pretrain.py --no_dry_run --epochs 250 --pretrained --micle   model_micle_pretrained_imagenet_med.pt &&
python pretrain.py --no_dry_run --epochs 250              --micle   model_micle_pretrained_med.pt &&

python train_supervised.py --no_dry_run --epochs 250 model_pretrained_imagenet.pt               model_pretrained_imagenet_supervised.pt &&
python train_supervised.py --no_dry_run --epochs 250 model_pretrained_imagenet_med.pt           model_pretrained_imagenet_med_supervised.pt &&
python train_supervised.py --no_dry_run --epochs 250 model_pretrained_med.pt                    model_pretrained_med_supervised.pt &&
python train_supervised.py --no_dry_run --epochs 250 model_micle_pretrained_imagenet_med.pt     model_micle_pretrained_imagenet_med_supervised.pt &&
python train_supervised.py --no_dry_run --epochs 250 model_micle_pretrained_med.pt              model_micle_pretrained_med_supervised.pt