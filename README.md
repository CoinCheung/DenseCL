## DenseCL: Dense Contrastive Learning for Self-Supervised Visual Pre-Training


<p align="center">
  <img src="./denseCL.png" width="600">
</p>

This is an unofficial PyTorch implementation of the [DenseCL paper](https://arxiv.org/abs/2011.09157), with the help and suggestions from @WXinlong and @DerrickWang005.

Currently, [regionCL-D](https://arxiv.org/abs/2111.12309) is added, and pretrained checkpoints are uploaded.


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on that code. Check the modifications by:
```
diff main_densecl.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```


### Unsupervised Training & Linear Classification

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

This implementation only supports **ResNet50/ResNet101**, since we need to modify computing graph architecture and I only modified ResNet50/ResNet101.

To do unsupervised pre-training and linear-evaluation of a ResNet50/ResNet101 model on ImageNet in an 8-gpu machine, please refer to [dist_train.sh](./dist_train.sh) for relevant starting script.

Since the paper says they use default mocov2 hyper-parameters, the above script uses same hyper-parameters as mocov2.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.


### Models

Our pre-trained denseCL/RegionCL-D models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">mlp</th>
<th valign="bottom">aug+</th>
<th valign="bottom">cos</th>
<th valign="bottom">IM<br/>top1</th>
<th valign="bottom">VOC<br/>AP50</th>
<th valign="bottom">model</th>
<th valign="bottom">md5</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCov2 R50</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">67.7</td>
<td align="center">82.4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>59fd9945</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2011.09157">DenseCL R50</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">63.8</td>
<td align="center">82.7</td>
<td align="center"><a href="https://github.com/CoinCheung/DenseCL/releases/download/v0.0.1/densecl_r50_checkpoint_0199.pth.tar">download</a></td>
<td align="center"><tt>7cfc894c</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2011.09157">DenseCL R101</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">65.4</td>
<td align="center">83.5</td>
<td align="center"><a href="https://github.com/CoinCheung/DenseCL/releases/download/v0.0.1/densecl_r101_checkpoint_0199.pth.tar">download</a></td>
<td align="center"><tt>006675e5</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2111.12309">RegionCL-D R50</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">67.5</td>
<td align="center">83.3</td>
<td align="center"><a href="https://github.com/CoinCheung/DenseCL/releases/download/v0.0.1/regioncl_r50_checkpoint_0199.pth.tar">download</a></td>
<td align="center"><tt>8afad30e</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2111.12309">RegionCL-D R101</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">67.5</td>
<td align="center">84.3</td>
<td align="center"><a href="https://github.com/CoinCheung/DenseCL/releases/download/v0.0.1/regioncl_r101_checkpoint_0199.pth.tar">download</a></td>
<td align="center"><tt>a1489ad4</tt></td>
</tr>
</tbody></table>

Here **IM** is imagenet-1k dataset. We freeze pretrained weights and only fine tune the last classifier layer.

Please be aware that though DenseCL cannot match mocov2 in the filed of classification, it is superior to mocov2 in terms of object detection. More results of detection can be found [here](./detection).


### Transferring to Object Detection

For details, see [./detection](./detection).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.



regioncl, r50:  
    Acc@1 67.518 Acc@5 88.256  
    Acc@1 67.534 Acc@5 88.212  
regioncl, r101:  
    Acc@1 67.504 Acc@5 88.212  
    Acc@1 67.470 Acc@5 88.104
    
