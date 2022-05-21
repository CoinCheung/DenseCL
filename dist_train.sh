
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


########
# for pretrain
########

## denseCL r50/r101
DATAPATH=/path/to/imagenet
ARCH=resnet50/resnet101
URL='tcp://localhost:10001'
WORD_SIZE=1
RANK=0
EPOCHS=200
LR=0.03 # 8 gpus
python main_densecl.py -a $ARCH --lr $LR --batch-size 256 --epochs $EPOCHS --world-size $WORD_SIZE --rank $RANK --dist-url $URL --multiprocessing-distributed --use-mixed-precision --mlp --moco-t 0.2 --aug-plus --cos $DATAPATH


## regionCL-D r50/r101
DATAPATH=/path/to/imagenet
ARCH=resnet50/resnet101
URL='tcp://localhost:10001'
WORD_SIZE=1
RANK=0
EPOCHS=200
LR=0.03 # 8 gpus
python main_densecl.py -a $ARCH --lr $LR --batch-size 256 --epochs $EPOCHS --world-size $WORD_SIZE --rank $RANK --dist-url $URL --multiprocessing-distributed --use-mixed-precision --mlp --moco-t 0.2 --aug-plus --cos $DATAPATH --cutmix


########
# linear eval
########
URL='tcp://localhost:10001'
WORD_SIZE=1
RANK=0
DATAPATH=/path/to/imagenet
PRETRAINED=./checkpoint_0199.pth.tar
ARCH=resnet50/resnet101
python main_lincls.py -a $ARCH --lr 30.0 --batch-size 256 --pretrained $PRETRAINED --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK $DATAPATH

