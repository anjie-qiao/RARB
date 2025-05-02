# RARB： Advancing Retrosynthesis with Retrieval-Augmented Graph Generation

<a href="https://ojs.aaai.org/index.php/AAAI/article/view/34203"><img src="https://img.shields.io/badge/AAAI-2025-brown.svg" height=22.5></a>

> We introduce a retrieval-augmented molecular graph generation framework. Our framework comprises three key components: a retrieval component that identifies similar molecules for the given product, an integration component that learns valuable clues from these molecules about which part of the product should remain unchanged, and a base generative model that is prompted by these clues to generate the corresponding reactants. We explore various design choices for critical and under-explored aspects of this framework and instantiate it as the Retrieval-Augmented RetroBridge(RARB). RARB demonstrates state-of-the-art performance on standard benchmarks, achieving a 14.8% relative improvement in top-1 accuracy over its base generative model, highlighting the effectiveness of retrieval augmentation. Additionally, RARB excels in handling out-of-distribution molecules, and its advantages remain significant even with smaller models or fewer denoising steps. These strengths make RARB highly valuable for real-world retrosynthesis applications, where extrapolation to novel molecules and high-throughput prediction are essential.

<img src="overview.jpg">

## Appendix
  You can find the appendix of our paper in this repository: rarb_appendix.pdf.

## Environment Setup
|Software|Version|
|-----|-----|
|Python|3.9|
|CUDA|11.6|
```shell
conda create --name rarb python=3.9 rdkit=2023.09.5 -c conda-forge -y
conda activate rarb
pip install -r requirements.txt
```

## Data
  You can download all the data used in our work from: https://drive.google.com/file/d/13FP-RBetjKZ1T-6gzD_PosLMIcaNLfF7/view?usp=sharing
## Training

 `python train.py --config configs/retrobridge.yaml --model RetroBridge`

## Sampling
Sampling with RARB:
```shell
python samplet.py \
       --config configs/retrobridge.yaml \
       --checkpoint checkpoints/RARB.ckpt \
       --samples samples \
       --model RetroBridge \
       --mode test \
       --n_samples 100 \
       --n_steps 500\
       --sampling_seed 1
```

## Uni-Rxn molecule encoder
we use the already trained Uni-rxn model to encode reactants for retrobridge and retrieval learning
the code come from the official git repository : https://github.com/qiangbo1222/Uni-RXN-official
we modified the following files for our tasks:
1. Uni-RXN-official/generation/build_retro_emb.py: ADDED to generate reacants embedding for retrieval learning
2. Uni-RXN-official/generation/build_retrieval_index.py: ADDED to retrieve top-k reactants embedding for products
3. Uni-RXN-official/generation/prepare_rxn_for_feat.py: MODIFIED to test our data
4. Uni-RXN-official/generation/featurize.py: MODIFIED to test our data

generate embedding commands:
```shell
python generation/build_retro_emb.py  \
--retrieval_file   /data/uspto50k/raw/uspto50k_train.csv  \
--model_dir  ckpt/uni_rxn_base.ckpt
```

retrieve top-k reactants commands:
```shell
python generation/build_retrieval_index.py  \
--input_file  /data/uspto50k/raw/uspto50k_train.csv  \
--retrieval_file  /data/uspto50k/raw/uspto50k_train.csv  \
--embedding_file tencoded_react_tensor.pt \
--retrieval_type  morgan \
--model_dir  ckpt/uni_rxn_base.ckpt
```
## USPTO-50k-cluseter dataset
```shell
cluster_splitting.ipynb
```

## Citation
```shell
@article{rarb2025, 
    title={Advancing Retrosynthesis with Retrieval-Augmented Graph Generation},
    author={Qiao, Anjie and Wang, Zhen and Rao, Jiahua and Yang, Yuedong and Wei, Zhewei},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    year={2025}, 
    month={Apr.}, 
    volume={39}, 
    pages={20004-20013} 
    DOI={10.1609/aaai.v39i19.34203}, 
}
```


