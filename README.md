<div align="center">
<h2>MICCAI 2025, Early Accepted<br>"MadCLIP: Few-shot Medical Anomaly Detection with CLIP"</h2>

<p>
  <a href="https://scholar.google.com/citations?user=7XWvNE4AAAAJ&hl=en">Mahshid Shiri</a>, 
  <a href="https://scholar.google.com/citations?user=VmjUxckAAAAJ&hl=en">Cigdem Beyan</a>,   
  <a href="https://scholar.google.com/citations?user=yV3_PTkAAAAJ&hl=en">Vittorio Murino</a> 
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2503.11609-b31b1b.svg)](https://arxiv.org/abs/2506.23810)

</div>

>**Abstract.** *An innovative few-shot anomaly detection approach is presented, leveraging the pre-trained CLIP model for medical data, and adapting it for both image-level anomaly classification (AC) and pixel-level anomaly segmentation (AS). A dual-branch design is proposed to separately capture normal and abnormal features through learnable adapters in the CLIP vision encoder. To improve semantic alignment, learnable text prompts are employed to link visual features. Furthermore, SigLIP loss is applied to effectively handle the many-to-one relationship between images and unpaired text prompts, showcasing its adaptation in the medical field for the first time. Our approach is validated on multiple modalities, demonstrating superior performance over existing methods for AC and AS, in both same-dataset and cross-dataset evaluations. Unlike prior work, it does not rely on synthetic data or memory banks, and an ablation study confirms the contribution of each component.*
> <center><img src="images/diagram.png "width="80%"></center>

### Install Dependencies
Create a virtual environment and install requirements via:
```
conda create -n madclip python=3.9
conda activate madclip
pip install -r requirements.txt
```
**ℹ️ Note**: This project uses Python `3.9.23`. Also, we used a Single NVIDIA GTX 4090 for the experiments.

### Pretrained model
- CLIP: Download [CLIP ckpt](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt) and put it under `CLIP/ckpt` folder

- MadCLIP: Download [MadCLIP ckpt](https://univr-my.sharepoint.com/:u:/g/personal/mahshid_shiri_univr_it/EW83zI3pwGhAs0TpQxq7_-YBG4n9M0JV0pvgp1BMrWYY3A) and put it under `ckpt` folder. The checkpoints are for 16 samples.


### Get the Datasets
The repository requires datasets to be formatted as per the MVFA guidelines. 
Please follow the instructions [here](https://github.com/MediaBrain-SJTU/MVFA-AD/blob/main/README.md#medical-anomaly-detection-benchmark) to do so.

### Project Structure
```
MadCLIP
├─ ckpt
│  ├─ Liver.pth
│  └─ Brain.pth
│  ├─ Histopathology.pth
│  └─ Retina_RESC.tar.pth
│  ├─ Retina_OCT2017.pth
│  └─ Chest.pth
├─ CLIP
│  ├─ ckpt
│  │  └─ ViT-L-14-336px.pt
│  ├─ model_configs
│  │  └─ ViT-L-14-336.json
│  ├─ adapter_shared.py
│  ├─ clip.py
│  ├─ model.py
│  ├─ modified_resnet.py
│  ├─ openai.py
│  ├─ tokenizer.py
│  └─ transformer.py
│  └─ bpe_simple_vocab_16e6.txt.gz
├─ data
│  ├─ Brain_AD
│  │  ├─ valid
│  │  └─ test
│  ├─ ...
│  └─ Retina_RESC_AD
│     ├─ valid
│     └─ test
├─ dataset
│  ├─ fewshot_seed
│  │  ├─ Brain
│  │  ├─ ...
│  │  └─ Retina_RESC
│  ├─ medical_few.py
├─ Prompt
│  ├─ CoOp.py
│  ├─ promptChooser.py
├─ loss.py
├─ key.json
├─ readme.md
├─ train.py
├─ test.py
└─ utils.py
└─ requirements.txt

```
### Experiment Tracking
This project uses [Weights & Biases (WandB)](https://wandb.ai/home) for experiment tracking. To enable WandB logging,go to your WandB settings and copy your API key and paste it in key.json file. 

### Training
The entrypoint of this repository is `train.py`. for quick start you can simply run:

`python train.py --obj $target-object --shot $few-shot-number`

Here is a snapshot of `--help`:
```
  --exp_name                 Experiment name.
  --model_name               Vision backbone to use. Choices: 'ViT-B-16-plus-240', 'ViT-L-14-336'. Default: 'ViT-L-14-336'.
  --pretrain                 Pretraining source. Choices: 'laion400m', 'openai'. Default: 'openai'.
  --obj                      Target dataset. Should be included in CLASS_INDEX. Default: 'Liver'.
  --data_path                Path to dataset root. Default: './data/'.
  --batch_size               Training batch size. Default: 16.
  --save_model               Whether to save model checkpoints.
  --save_path                Path to save models. Default: './ckpt/'.
  --img_size                 Input image size. Default: 240.
  --epoch                    Number of training epochs. Default: 60.
  --learning_rate            Learning rate. Default: 0.001.
  --features_list            List of vision transformer layers to apply adapter on. Default: [6, 12, 18, 24].
  --seed                     Random seed. Default: 111.
  --shot                     Number of training samples per class (few-shot setting). Default: 16.
  --iterate                  few shot samples indices for reproducibility.
  --text_mood                Prompt learning strategy. Choices: 'fix', 'learnable_all', 'learnable_abnormal'. Default: 'learnable_all'.
  --contrast_mood            Whether to apply dual optimization objective or not. Choices: 'yes', 'no'. Default: 'yes'.
  --dec_type                 Aggregation method of adapter outputs. Choices: 'mean', 'max', 'both'. Default: 'mean'.
  --loss_type                Loss type. Choices: 'sigmoid', 'softmax'. Default: 'sigmoid'.
  --visionA VISIONA          Adapter architecture type. Default: 'MFCFC'.
```
### Testing
You can run the following:

`python test.py --obj $target-object --shot $few-shot-number`

### Acknowledgements
This repo builds upon [CoOp / CoCoOp](https://github.com/KaiyangZhou/CoOp/tree/main) and [MVFA](https://github.com/MediaBrain-SJTU/MVFA-AD/tree/main), so huge thanks to the authors!

We acknowledge the financial support  of the PNRR project FAIR - Future AI Research (PE00000013), under the NRRP MUR program funded by the NextGenerationEU.

### Citation  
Please cite this work as follows if you find it useful!
```bibtex
@article{shiri2025madclip,
  title={MadCLIP: Few-shot Medical Anomaly Detection with CLIP},
  author={Shiri, Mahshid and Beyan, Cigdem and Murino, Vittorio},
  journal={arXiv preprint arXiv:2506.23810},
  year={2025}
}
```
