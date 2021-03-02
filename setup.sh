pip install -r requirements_gpu.txt

cd center/models/DCNv2
rm -rf build
rm -rf DCNv2.egg-info
rm -rf _ext*
sh make.sh
python testcuda.py

cd line_detection_module/models/networks/DCNv2
rm -rf build
rm -rf DCNv2.egg-info
_ext*
sh make.sh
python testcuda.py
cd ../../../

gdown --id 1DWyQhOpLlVYu4r35IijaWHbjS9A7cxTF
mv detect_cmnd_weight.pth weights

gdown --id 1ztUf3lzPCHl0ND73MMgYvsuusxYMys-x
mv line_detect_weight.pth weights

gdown --id 1FbVV6H9Y41v_fz0MT4X5PriTxZlNm2KL
mv seq2seqocr_best.pth weights