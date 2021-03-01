## DenseCL: Dense Contrastive Learning for Self-Supervised Visual Pre-Training


<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

This is an unofficial PyTorch implementation of the [DenseCL paper](https://arxiv.org/abs/2011.09157), with the help and suggestions from @WXinlong and @DerrickWang005.


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on that code. Check the modifications by:
```
diff main_densecl.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

This implementation only supports **resnet-50**, since we need to modify model architecture and I only modified resnet-50.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
sh dist_train.sh [your imagenet-folder with train and val folders]
```
Since the paper says they use default mocov2 hyper-parameters, the above script uses same hyper-parameters as mocov2.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

Linear classification results on ImageNet using this repo with 8 NVIDIA V100 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">MoCo v1<br/>top-1 acc.</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<th valign="bottom">DenseCL<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">53 hours</td>
<td align="center">60.8&plusmn;0.2</td>
<td align="center">67.5&plusmn;0.1</td>
<td align="center"> 63.8 &plusmn;0.1</td>
</tr>
</tbody></table>

Here we run 5 trials (of pre-training and linear classification) and report mean&plusmn;std: the 5 results of MoCo v1 are {60.6, 60.6, 60.7, 60.9, 61.1}, of MoCo v2 are {67.7, 67.6, 67.4, 67.6, 67.3}, and of DenseCL are (...).

Please be aware that though denseCL cannot match mocov2 in the filed of classification, it is superior to mocov2 in terms of object detection. See [./detection](detection) for details.


### Models

Our pre-trained ResNet-50 models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">mlp</th>
<th valign="bottom">aug+</th>
<th valign="bottom">cos</th>
<th valign="bottom">top-1 acc.</th>
<th valign="bottom">model</th>
<th valign="bottom">md5</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://arxiv.org/abs/1911.05722">MoCo v1</a></td>
<td align="center">200</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">60.6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>b251726a</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">67.7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>59fd9945</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">800</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">71.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>a04e12f8</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2011.09157">DenseCL</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center"> 63.8 </td>
<td align="center"><a href="https://github.com/CoinCheung/denseCL/releases/download/v0.0.1/checkpoint_0199.pth.tar">download</a></td>
<td align="center"><tt>7cfc894c</tt></td>
</tr>
</tbody></table>


### Transferring to Object Detection

See [./detection](detection).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### See Also
* [moco.tensorflow](https://github.com/ppwwyyxx/moco.tensorflow): A TensorFlow re-implementation.
* [Colab notebook](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb): CIFAR demo on Colab GPU.
