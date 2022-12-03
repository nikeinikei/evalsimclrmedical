#!/bin/bash

python evaluate_accuracy.py --label "ImageNet -> CheXpert (SimCLR)" model_pretrained_imagenet_med_supervised.pt &&
python evaluate_accuracy.py --label "ImageNet -> CheXpert (MICLe)" model_micle_pretrained_imagenet_med_supervised.pt &&
python evaluate_accuracy.py --label "ImageNet" model_pretrained_imagenet_supervised.pt &&
python evaluate_accuracy.py --label "CheXpert (SimCLR)" model_pretrained_med_supervised.pt &&
python evaluate_accuracy.py --label "CheXpert (MICLe)" model_micle_pretrained_med_supervised.pt