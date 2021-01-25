#!/bin/sh
pip install -r requirements_gpu.txt

cd center/models/DCNv2
rm -rf build
rm -rf DCNv2.egg-info
sh make.sh
python testcuda.py
cd ../../../

gdown --id 1lubOiOXsh9A4D5FUCKubaR1K8GSfnnVq
mv model_cmnd_best.pth weights