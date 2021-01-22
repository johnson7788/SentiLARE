# SentiLARE: Sentiment-Aware Language Representation Learning with Linguistic Knowledge

## Introduction

SentiLARE是一种通过语言知识增强的感知情感的预训练语言模型。 
您可以阅读我们的[paper](https://www.aclweb.org/anthology/2020.emnlp-main.567/)了解更多详细信息。 该项目是我们工作的PyTorch实施。

## Dependencies

* Python 3
* NumPy
* Scikit-learn
* PyTorch >= 1.3.0
* PyTorch-Transformers (Huggingface) 1.2.0
* TensorboardX
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) 0.2.6 (Optional, used for linguistic knowledge acquisition during pre-training and fine-tuning)
* NLTK (Optional, used for linguistic knowledge acquisition during pre-training and fine-tuning)

## 微调快速入门 

### 下游任务数据集 

我们的实验包含sentence-level的情感分类(例如SST / MR / IMDB / Yelp-2 / Yelp-5)和aspect-level的情感分析(例如Lap14 / Res14 / Res16)。 
您可以下载预处理的数据集 ([Google Drive](https://drive.google.com/drive/folders/1v84riTNxCMJi3HWhJdDNyBryCtTTfNjy?usp=sharing) / [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/f6baaff5c398463388b2/)) 下游任务。 数据格式的详细说明附在数据集上。 

### 微调 Fine-tuning

要快速进行微调实验，您可以直接下载checkpoint  ([Google Drive](https://drive.google.com/drive/folders/1v84riTNxCMJi3HWhJdDNyBryCtTTfNjy?usp=sharing) / [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/f6baaff5c398463388b2/)) 我们预训练的模型, 我们显示了在SST上微调SentiLARE的样本，如下所示： 

```shell
cd finetune
CUDA_VISIBLE_DEVICES=0,1,2 python run_sent_sentilr_roberta.py \
          --data_dir data/sent/sst \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name sst \
          --do_train \
          --do_eval \
          --max_seq_length 256 \
          --per_gpu_train_batch_size 4 \
          --learning_rate 2e-5 \
          --num_train_epochs 3 \
          --output_dir sent_finetune/sst \
          --logging_steps 100 \
          --save_steps 100 \
          --warmup_steps 100 \
          --eval_all_checkpoints \
          --overwrite_output_dir
```

注意，`data_dir` 设置为预处理的SST数据集的目录，而`model_name_or_path`设置为预处理的模型checkpoint的目录。 
`output_dir`是保存微调checkpoint的目录。 您可以参考微调代码以获取其他超参数的描述。


有关在其他数据集上微调SentiLARE的更多详细信息，请参见 [`finetune/README.MD`](https://github.com/thu-coai/SentiLARE/tree/master/finetune).

### POS Tagging and Polarity Acquisition for Downstream Tasks

在预处理期间，我们使用NLTK tokenize原始数据集，使用Stanford Log-Linear Tagger 标注句子，
并使用Sentence-BERT获得情感倾向。 我们会很快发布了原始数据集和预处理脚本(即将发布)，因此您可以按照我们的流程来获取自己数据集的语言知识。


## Pre-training
如果您想自己进行预训练，而不是直接使用我们提供的checkpoint，
则本部分可以帮助您预处理预训练数据集并运行预训练脚本。 

### Dataset

We use Yelp Dataset Challenge 2019 as our pre-training dataset. According to the [Term of Use](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf) of Yelp dataset, you should download [Yelp dataset](https://www.yelp.com/dataset) on your own.

### POS Tagging and Polarity Acquisition for Pre-training Dataset

类似于微调，我们还对预训练数据集进行词性标注和情感倾向获取。
预处理脚本将很快发布。 请注意，由于预训练数据集非常大，因此预处理过程可能会花费很长时间，
因为我们需要使用Sentence-BERT来获取预训练数据集中所有句子的表示向量。 

### Pre-training

预训练代码将很快发布。 

## Citation

```
@inproceedings{ke-etal-2020-sentilare,
    title = "{S}enti{LARE}: Sentiment-Aware Language Representation Learning with Linguistic Knowledge",
    author = "Ke, Pei  and Ji, Haozhe  and Liu, Siyang  and Zhu, Xiaoyan  and Huang, Minlie",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "6975--6988",
}
```

**Please kindly cite our paper if this paper and the codes are helpful.**

## Thanks

Many thanks to the GitHub repositories of [Transformers](https://github.com/huggingface/transformers) and [BERT-PT](https://github.com/howardhsu/BERT-for-RRC-ABSA). Part of our codes are modified based on their codes.
