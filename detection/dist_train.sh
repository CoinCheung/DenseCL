
## densecl

CKPT=../checkpoint_0199.pth.tar
CONFIG=configs/pascal_voc_R_50_C4_24k_moco.yaml
# CONFIG=configs/pascal_voc_R_101_C4_24k_moco.yaml



rm ./output_denseCL_200ep.pkl
python convert-pretrain-to-detectron2.py ../checkpoint_0199.pth.tar ./output_denseCL_200ep.pkl
python train_net.py --config-file $CONFIG --num-gpus 8 MODEL.WEIGHTS ./output_denseCL_200ep.pkl

## original supervised
# python train_net.py --config-file configs/pascal_voc_R_50_C4_24k.yaml --num-gpus 8
