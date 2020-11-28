# AuGPT
![tests](https://github.com/ufal/augpt/workflows/tests/badge.svg)
## Getting started
Start with creating a python 3.7 venv and installing requirements.txt. Python 3.8 is **not supported** by ConvLab-2. Also,
newer version of transformers is unfortunatelly not supported by ConvLab-2, therefore you need to install the legacy
transformers version. 

Get started by cloning this repository:
```bash
git clone https://github.com/ufal/augpt.git
```

## Downloading datasets
First, start by installing required packages. If you intend to train on these datasets, you can install all required packages
by running:
```bash
pip install -r requirements.txt
```
Otherwise, install the required packages by running:
```bash
pip install -r requirements-minimal.txt
```

To download datasets, run `scripts/download_{dataset}.py`, where dataset is the name of the dataset you need.
Supported datasets:
- `taskmaster`: The Taskmaster corpus [1] comprising over 55,000 spoken and written task-oriented dialogs in over a dozen domains.
- `schemaguided`: The Schema-Guided Dialogue [2] dataset consisting of over 20k annotated multi-domain, task-oriented conversations between a human and a virtual assistant.  
- `multiwoz`: The MultiWOZ 2.0 dataset [3] - a large-scale multi-domain wizard-of-oz dataset for task-oriented dialogue modelling.
- `convlab_multiwoz`: The MultiWOZ 2.1 dataset [4] - a cleaner version of MultiWOZ 2.0 with span information.

In this work, we name the union of `taskmaster` and `schemaguided` as `bigdata`.


## Interact and generate
To interact with the model or use it in your own pipline, you need to ensure all the required packages are present by running:
```bash
pip install -r requirements-minimal.txt
```

To run the model in interactive mode, you can use `interact.py` utility. Alternatively, to use the model
in your code, you can modify the following code:
```python
import pipelines  # Required here, modifies the transformers package to support AuGPT pipeline.
import transformers

# Loads the pipeline with MultiWOZ 2.1 model
pipeline = transformers.pipeline('augpt-conversational', 'jkulhanek/aug-mw-21')

# Either AuGPTConversation or Conversation can be used
conversation = pipelines.AuGPTConversation('Hi, I need a hotel')

conversation = conversation(pipeline)
print(conversation.generated_responses[-1])
```

To generate the predictions, use `generate.py` script.
```bash
./generate.py --model jkulhanek/aug-mw-21 --dataset multiwoz-2.1-test --file predictions.txt
```

## Training and evaluation
The following scripts creates a virtual environment and installs required packages for training and ConvLab-2 evaluation.
```bash
python -m venv ~/envs/dstc
source ~/envs/dstc/bin/activate
pip install -r requirements.txt
cd ~/source 
git clone git@github.com:ufal/ConvLab-2.git
cd ConvLab-2
git reset --hard 8b4464c57de0fbc497ce3532532c30ae461906e9
pip install -e . --no-deps
python -m spacy download en_core_web_sm
```

### Training bigdata model
The bigdata pre-trained model can be trained using the following arguments:
```bash
./train.py --epochs 8 --restrict-domains --train-dataset schemaguided-train+taskmaster-train --dev-dataset schemaguided-dev+taskmaster-dev --validation-steps 10000 --logging-steps 1000 --warmup-steps 5000 --evaluation-dialogs 0 --fp16
```

The pre-trained model can be downloaded from the Hugging Face model repository as `jkulhanek/augpt-bigdata`.

### Fine-tuning on MultiWOZ
The pretrained model can be finetuned on MultiWOZ 2.x dataset as follows:
```bash
./train_multiwoz.py --train-dataset multiwoz-2.1-train --dev-dataset multiwoz-2.1-val --model jkulhanek/augpt-bigdata --backtranslations latest --response-loss unlikelihood --epochs 10 --fp16 --clean-samples
```
For MultiWOZ 2.0, substitute the correct dataset version.

### Distributed training 
To start the training on single CPU node (for testing), run the training with the following arguments:
```bash
./train.py --no-cuda --gradient-accumulation-steps 4
```

> **_NOTE:_**  For optimal performance at least **four** GPUs are required for training.

To run the training with single GPU:
```bash
./train.py --gradient-accumulation-steps 4
```

To run on single node with multiple GPUs, run the following command:
```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py
```
In this case the expected number of GPUs is four, you may need to adjust *learning_rate* and/or *gradient-accumulation-steps* accordingly.

To run the training on multiple nodes with multiple GPUs, you can use pytorch launch utility <https://pytorch.org/docs/stable/distributed.html#launch-utility>.
Alternatively, consult your job scheduling system. You may need to set the environment variables: `LOCAL_RANK`, `RANK`, `WORLD_SIZE`, `MASTER_PORT`, `MASTER_ADDR`.
In this case, `RANK` is global number of current process across the world and `LOCAL_RANK` is the number of each process running on single node. 
Every node is required to have as many GPUs as there are processes running on single machine.

## Evaluation
All packages required for the training must also be installed for evaluation.

### ConvLab-2 evaluation
To evaluate your trained model using ConvLab-2 evaluation, run the following script:
```bash
./evaluate_convlab.py --model {model}
```

### MultiWOZ 2.x evaluation
To evaluate your trained model using MultiWOZ evaluation, run the following:
```bash
./evaluate_multiwoz.py --model {model} --dataset multiwoz-2.1-test
```

If you have your predictions generated by running `generate.py` script, you can 
evaluate them by running:
```bash
./evaluate_multiwoz.py --file predictions.txt --dataset multiwoz-2.1-test
```
For MultiWOZ 2.0, substitute the correct dataset version.

## References
[1]: Byrne, B.; Krishnamoorthi, K.; Sankar, C.; Neelakantan, A.; Duckworth, D.; Yavuz, S.; Goodrich, B.; Dubey, A.; Kim, K.-Y.; and Cedilnik, A. 2019. Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset.

[2]: Rastogi, A.; Zang, X.; Sunkara, S.; Gupta, R.; and Khaitan, P. 2019. Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset.arXiv preprint arXiv:1909.05855.

[3]: Budzianowski, P.; Wen, T.-H.; Tseng, B.-H.; Casanueva, I.; Ultes, S.; Ramadan, O.; and Gašić, M. 2018. Multiwoz - a large-scale multi-domain wizard-of-oz dataset for task-oriented dialogue modelling. arXiv preprint arXiv:1810.00278.

[4]: Eric, M.; Goel, R.; Paul, S.; Kumar, A.; Sethi, A.; Ku, P.;Goyal, A. K.; Agarwal, S.; Gao, S.; and Hakkani-Tur, D. 2019. MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines.arXiv preprint arXiv:1907.01669.
