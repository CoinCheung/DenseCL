
## DenseCL: Transferring to Detection

The `train_net.py` script reproduces the object detection experiments on Pascal VOC and COCO.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
    ```
    $ git clone https://github.com/facebookresearch/detectron2.git
    $ cd detectron2
    $ git checkout 3e71a2711bec
    $ python -m pip install -e .
    ```
    This requires cuda10.2 to work.

2. Convert a pre-trained model to detectron2's format:
   ```
   python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

3. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.
     ```
        $ mkdir -p datasets && cd datasets
        $ ln -s VOC2007 .
        $ ln -s VOC2012 .
     ```

4. Run training:
   ```
   # r50 
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   # r101 
   python train_net.py --config-file configs/pascal_voc_R_101_C4_24k_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```
    
    Or you can see [dist_train.sh](./dist_train.sh) for the training scripts.

### Results

Below are the results on Pascal VOC 2007 test, fine-tuned on 2007+2012 trainval for 24k iterations using Faster R-CNN with a R50/R101-C4 backbone:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pretrain</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP75</th>
<!-- TABLE BODY -->
<tr><td align="left">ImageNet-1M, R50, supervised</td>
<td align="center">81.3</td>
<td align="center">53.5</td>
<td align="center">58.8</td>
</tr>
<tr><td align="left">ImageNet-1M, R50, MoCo v1, 200ep</td>
<td align="center">81.5</td>
<td align="center">55.9</td>
<td align="center">62.6</td>
</tr>
</tr>
<tr><td align="left">ImageNet-1M, R50, MoCo v2, 200ep</td>
<td align="center">82.4</td>
<td align="center">57.0</td>
<td align="center">63.6</td>
</tr>
</tr>
<tr><td align="left">ImageNet-1M, R50, MoCo v2, 800ep</td>
<td align="center">82.5</td>
<td align="center">57.4</td>
<td align="center">64.0</td>
</tr>
<tr><td align="left">ImageNet-1M, R50, DenseCL, 200ep</td>
<td align="center">82.7</td>
<td align="center">58.5</td>
<td align="center">65.6</td>
</tr>
<tr><td align="left">ImageNet-1M, R101, DenseCL, 200ep</td>
<td align="center">83.57</td>
<td align="center">61.02</td>
<td align="center">68.20</td>
</tr>
<tr><td align="left">ImageNet-1M, R50, RegionCL-D, 200ep</td>
<td align="center">83.32</td>
<td align="center">58.72</td>
<td align="center">65.57</td>
</tr>
<tr><td align="left">ImageNet-1M, R101, RegionCL-D, 200ep</td>
<td align="center">84.30</td>
<td align="center">61.59</td>
<td align="center">68.17</td>
</tr>
</tbody></table>

***Note:*** These results are means of 5 trials. Variation on Pascal VOC is large: the std of AP50, AP, AP75 is expected to be 0.2, 0.2, 0.4 in most cases. We recommend to run 5 trials and compute means.


denseCL, r50:  
    82.64/58.32/64.60  
    82.64/58.41/64.89

denseCL, r101:  
    83.57/61.02/68.20  
    83.52/60.89/67.32

regionCL-D, r50:  
    83.24/58.84/65.98  
    83.40/58.60/65.16

regionCL-D, r101:  
    84.22/61.48/68.14  
    84.39/61.70/68.21


