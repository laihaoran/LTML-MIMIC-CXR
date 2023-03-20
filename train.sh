## train ANR_LLA mdethod
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config ./configs/mimic/LTML_resnet50_ANR_LLA.py

## test ANR_LLA mdethod
CUDA_VISIBLE_DEVICES=0 python tools/test.py --config './work_dirs/LTML_MIMIC_CXR_resnet50_ANR_LLA/LTML_resnet50_ANR_LLA.py'  --checkpoint './work_dirs/LTML_MIMIC_CXR_resnet50_ANR_LLA/latest.pth'


## train ANR_LLM mdethod
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config ./configs/mimic/LTML_resnet50_ANR_LLM.py

## test ANR_LLM mdethod
CUDA_VISIBLE_DEVICES=0 python tools/test.py --config './work_dirs/LTML_MIMIC_CXR_resnet50_ANR_LLA/LTML_resnet50_ANR_LLM.py'  --checkpoint './work_dirs/LTML_MIMIC_CXR_resnet50_ANR_LLM/latest.pth'