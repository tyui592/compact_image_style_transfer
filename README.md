Compact Image Style Transfer
==

**Unofficial PyTorch implementation of Compact Image Style Transfer: Channel Pruning on a Single Training of Network**

**Reference**
* [Uncorrelated feature encoding for faster image style transfer](https://linkinghub.elsevier.com/retrieve/pii/S0893608021000873)
* [Compact Image Style Transfer: Channel Pruning on a Single Training of Network](https://www.mdpi.com/1424-8220/22/21/8427)


## Intro.

> This method automatically finds a number of consistently inactive convolution channels during the network training phase by using two new losses, i.e., channel loss and xor loss. The former maximizes the number of inactive channels and the latter fixes the positions of these inactive channels to be the same for the image.


## Usage
scripts.sh


## Result

### $l_0$-norm per image
* Channel response (white: nonzero response, black: zero response) of encoded feature map per image.
| baseline | w/ uncorrelation loss | w/ channel & xor loss |
| --- | --- | --- |
| ![img1](./imgs/for_readme/l0_norm_baseline.png) | ![img2](./imgs/for_readme/l0_norm_with_uncorrelation.png) |![img3](./imgs/for_readme/l0_norm_with_ch_xor.png) |

*The result of pretrained vgg encoder(baseline) has no zero response, and if uncorrelation loss is used, zero response comes out, but the position changes depending on the image. However, using channel loss and xor loss results in consistent zero response for the same channel.*


### Pruning Result
For channel pruning, channels were sorted in order of the magnitude of the l2-norm for the test data, and prunings were performed sequentially starting with the smallest value.

**baseline**
| Sorted Channel Magnitude(l2-norm) | Pruning Results | 
| --- | --- |
| ![img4](./imgs/for_readme/l2_norm_sorted_values_baseline.png) | ![img5](./imgs/for_readme/prunings_baseline.png) |

**w/ channel & xor losses**
| Sorted Channel Magnitude(l2-norm) | Pruning Results | 
| --- | --- |
| ![img6](./imgs/for_readme/l2_norm_sorted_values_ch_xor.png) | ![img7](./imgs/for_readme/prunings_ch_xor.png) |