# CTC-NET
The codes for the work "An effective CNN and Transformer complementary network for medical image segmentation"(https://www.sciencedirect.com/science/article/abs/pii/S0031320322007075). 

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data
- The datasets and the original version of codes we used are provided by TransUnet's authors. Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).

## 3. Environment
- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test
- Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the batch size can be reduced to 12 or 6 to save memory.
- - Train
```bash
sh train.sh or python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```
- Test
```bash
sh test.sh or python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```
## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)

## Citation
```bibtex
@article{YUAN2023109228,
title = {An effective CNN and Transformer complementary network for medical image segmentation},
journal = {Pattern Recognition},
volume = {136},
pages = {109228},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.109228},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322007075},
author = {Feiniu Yuan and Zhengxiao Zhang and Zhijun Fang},
keywords = {Transformer, Medical image segmentation, Feature complementary module, Cross-domain fusion, Convolutional Neural Network},
abstract = {The Transformer network was originally proposed for natural language processing. Due to its powerful representation ability for long-range dependency, it has been extended for vision tasks in recent years. To fully utilize the advantages of Transformers and Convolutional Neural Networks (CNNs), we propose a CNN and Transformer Complementary Network (CTCNet) for medical image segmentation. We first design two encoders by Swin Transformers and Residual CNNs to produce complementary features in Transformer and CNN domains, respectively. Then we cross-wisely concatenate these complementary features to propose a Cross-domain Fusion Block (CFB) for effectively blending them. In addition, we compute the correlation between features from the CNN and Transformer domains, and apply channel attention to the self-attention features by Transformers for capturing dual attention information. We incorporate cross-domain fusion, feature correlation and dual attention together to propose a Feature Complementary Module (FCM) for improving the representation ability of features. Finally, we design a Swin Transformer decoder to further improve the representation ability of long-range dependencies, and propose to use skip connections between the Transformer decoded features and the complementary features for extracting spatial details, contextual semantics and long-range information. Skip connections are performed in different levels for enhancing multi-scale invariance. Experimental results show that our CTCNet significantly surpasses the state-of-the-art image segmentation models based on CNNs, Transformers, and even Transformer and CNN combined models designed for medical image segmentation. It achieves superior performance on different medical applications, including multi-organ segmentation and cardiac segmentation.}
}
```
