
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# for pretrain
# EPOCHS=200
# python main_moco.py -a resnet50 --lr 0.03 --batch-size 256 --epochs $EPOCHS --world-size 1 --rank 0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --use-mixed-precision --mlp --moco-t 0.2 --aug-plus --cos $1

PRETRAINED=res/r50_org_v2/checkpoint_0199.pth.tar
python main_lincls.py -a resnet50 --lr 30.0 --batch-size 256 --pretrained $PRETRAINED --dist-url 'tcp://127.0.0.1:10002' --multiprocessing-distributed --world-size 1 --rank 0 /data/zzy/.datasets/imagenet/

