#!/bin/sh
#pip install -r requirements_cpu.txt
#
#npm install localtunnel

cd center/models/DCNv2
make.sh
python testcuda.py
cd ../../../

#gdown --id 1lubOiOXsh9A4D5FUCKubaR1K8GSfnnVq
#mv detect_cmnd_weight.pth weights