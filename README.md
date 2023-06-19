# On the Functional Similarity of Robust and Non-Robust Neural Representations

This repository contains the code to reproduce the results presented in the ICML'23 paper "On the Functional Similarity of Robust and Non-Robust Neural Representations".

## Installation

### Conda
```console
conda env create -f environment.yml
```

### Pip
```console
pip install -r requirements.txt
```

## Setup

There's a config file which tells the script where it can find or download the datasets to. Please edit `config/default.env`:
```bash
[dataset_root]
pytorch = '/data/pytorch' # path to pytroch datasets such as cifar10
celeba = '/data/celeba' # path to celeba dataset
```

## Train

### Train a neural net

If you want to skip this step, just use the pretrained neural networks uploaded under *archive/* folder. Otherwise train a new network by

```console
python src/bin/train.py
```
#### Settings

* **-h, --help** Get help about parameters
* **-m, --model** Model to train. Please choose from: *lenet, tiny10, resnet_w1, resnet_w2, resnet_w3, resnet_w4, resnet_w5, resnet18, resnet34, resnet50, resnet101, resnet152, preactresnet18 wide_resnet_28_10, wide_resnet_34_10, vgg11, vgg13, vgg16, vgg19, rvit, engstromrn50, imagenetrn50*. Default: lenet
* **-d, --dataset** Dataset to learn on. Please choose from: *cifar10, cifar100, mnist, svhn, fashion, celeba*. Default: mnist
* **-e, --epochs**  Number of epochs to train. Default: 300
* **-lr, --lr**     Learning rate. Default: 1e-1
* **-o, --out-dir** Folder to save networks to. Default: snapshots/
* **-b, --batch-size**     Batch size. Default: 128
* **-s, --save-frequency** How often to save model in iterations. Default: 10000
* **-i, --init-noise**     Initialisation seed. Default: 0
* **-g, --gradient-noise** Image batch order seed. Default: 0
* **-wd, --weight-decay**  Weight decay to use. Deault: 1e-4
* **-opt, --optimizer**    Name of the optimizer. Please choose from: *adam, sgd*. Default: sgd
* **--robust**             Boolean for triggering TRADES adversarial training. Default: False
* **--madry**             Boolean for triggering Madry PGD adversarial training. Default: False
* **--epsilon**            Epsilon for linf adversarial training (float or float-convertible expression). Default: 8/255
* **--step-size**          Attack optimization step size for adversarial training (float or float-convertible expression). Default: 2/255
* **--beta**               Beta parameter for TRADES adversarial training. Default: 10.
* **--attack-ratio**       Ratio of adversarial samples (alpha in the paper) in adversarial training. Default: 1.

There's a default schedule in the learning, the learning rate is divided by 10 at 1/3 of the training, and with another 10 at the 2/3 of the training.

Your models are going to be saved under *snapshots/* by default.

## Stitching

### Standard stitching
```console
python src/bin/find_transform.py path/to/model1.pt /path/to/model2.pt layer1 layer2 dataset [options]
```
where layer1 corresponds to a layer of model1, and layer2 to model2. Example:
```console
python src/bin/find_transform.py snapshots/resnet18/CIFAR10/in0-gn0/model.pt snapshots/resnet18/CIFAR10/robust/Sehwag2021Proxy_R18.pt layer1.0.add layer1.0.add cifar10 -e 30 -b 128 --madry --attack-ratio 0.5
```

#### Settings

* **-h, --help** Get help about parameters
* **--run-name** The name of the subfolder to save to. If not given, it defaults to the current date-time.
* **-e, --epochs** Number of epochs to train. Default: 30
* **-lr, --lr** Learning rate. Default: 1e-3
* **-o, --out-dir** Folder to save networks to. Default: snapshots/
* **-b, --batch-size** Batch size. Default: 128
* **-s, --save-frequency** How often to save the transformation matrix in iterations.
                       This number is multiplied by the number of epochs. Default: 10
* **--seed** Seed of the run. Default: 0
* **-wd, --weight-decay** Weight decay to use. Deault: 1e-4
* **--optimizer** Name of the optimizer. Please choose from: adam, sgd. Default: adam
* **--debug** Either to run in debug mode or not. Default: False
* **--flatten** Either to flatten layers around transformation.
* **--l1** l1 regularization used on transformation matrix. **Not used in the paper.** Default: 0
* **--cka-reg** CKA regularisation used on transformation matrix. Default: 0
* **-i, --init** Initialisation of transformation matrix. Options:
   * random: random initialisation. Default.
   * perm: random permutation
   * eye: identity matrix
   * ps_inv: pseudo inverse initialisation
   * ones-zeros: weight matrix is all 1, bias is all 0.
* **-m, --mask** Any mask applied on transformation. **Not used in the paper.** Options:
   * identity: All values are 1 in mask. Default.
   * semi-match: Based on correlation choose the best pairs.
   * abs-semi-match: Semi-match between absolute correlations.
   * random-permuation: A random permutation matrix.
* **--target-type** The loss to apply at logits. **Not used in the paper.** Options:
   * hard: Use true labels. Default.
   * soft_1: Use soft crossentropy loss to model1.
   * soft_2: Use soft crossentropy loss to model2.
   * soft_12: Use soft crossentropy loss to the mean of model1 and model2.
   * soft_1_plus_2: Use soft crossentropy loss to the sum of model1 and model2.
* **--temperature** The temperature to use if target type is a soft label. Default: 1. **Not used in the paper.**
* **--robust**             Boolean for triggering TRADES adversarial training. Default: False
* **--madry**              Boolean for triggering Madry PGD adversarial training. Default: False
* **--epsilon**            Epsilon for linf adversarial training (float or float-convertible expression). Default: 8/255
* **--step-size**          Attack optimization step size for adversarial training (float or float-convertible expression). Default: 2/255
* **--beta**               Beta parameter for TRADES adversarial training. Default: 10.
* **--attack-ratio**       Ratio of adversarial samples (alpha in the paper) in adversarial training. Default: 1.
* **--no-pinv**            Disable several calculations (CKA, pseudoinverse-based direct matching, etc.). **Might be necessary** as pinv can crash the program. Generally recommended. Default: False.
* **--no-bn**              Freeze batch norm layers in the donor networks. Leads to significantly slower convergence. Not recommended.


There's a default schedule in the learning, the learning rate is divided by 10 at 1/3 of the training, and with another 10 at the 2/3 of the training.

You will find the results of your runs under *results/* folder by default.

### Cross-task stitching
```console
python src/bin/find_cross_dataset_transform.py path/to/model1.pt /path/to/model2.pt layer1 layer2 dataset eval_dataset [options]
```
where `layer1` corresponds to a layer of `model1`, and `layer2` to `model2`. `dataset` will be used for training and evaluating the stitcher and `eval_dataset` will only be used for evaluation. Ideally, the two datasets should represent the two tasks on which the models were trained and `eval_dataset` should be the dataset od the two datasets that was not used for training. Example:
```console
python src/bin/find_cross_dataset_transform.py snapshots/resnet18/CIFAR10/in0-gn0/model.pt snapshots/resnet18/SVHN/in0-gn0/model.pt layer1.0.add layer1.0.add cifar10 svhn -e 30 -b 64
```

* **-h, --help** Get help about parameters
* **--run-name** The name of the subfolder to save to. If not given, it defaults to the current date-time.
* **-e, --epochs** Number of epochs to train. Default: 30
* **-lr, --lr** Learning rate. Default: 1e-3
* **-o, --out-dir** Folder to save networks to. Default: snapshots/
* **-b, --batch-size** Batch size. Default: 128
* **-s, --save-frequency** How often to save the transformation matrix in iterations.
                       This number is multiplied by the number of epochs. Default: 10
* **--seed** Seed of the run. Default: 0
* **-wd, --weight-decay** Weight decay to use. Deault: 1e-4
* **--optimizer** Name of the optimizer. Please choose from: adam, sgd. Default: adam
* **--debug** Either to run in debug mode or not. Default: False
* **--flatten** Either to flatten layers around transformation.
* **--l1** l1 regularization used on transformation matrix. **Not used in the paper.** Default: 0
* **--cka-reg** CKA regularisation used on transformation matrix. Default: 0
* **-i, --init** Initialisation of transformation matrix. Options:
   * random: random initialisation. Default.
   * perm: random permutation
   * eye: identity matrix
   * ps_inv: pseudo inverse initialisation
   * ones-zeros: weight matrix is all 1, bias is all 0.
* **-m, --mask** Any mask applied on transformation. **Not used in the paper.** Options:
   * identity: All values are 1 in mask. Default.
   * semi-match: Based on correlation choose the best pairs.
   * abs-semi-match: Semi-match between absolute correlations.
   * random-permuation: A random permutation matrix.
* **--target-type** The loss to apply at logits. **Not used in the paper.** Options:
   * hard: Use true labels. Default.
   * soft_1: Use soft crossentropy loss to model1.
   * soft_2: Use soft crossentropy loss to model2.
   * soft_12: Use soft crossentropy loss to the mean of model1 and model2.
   * soft_1_plus_2: Use soft crossentropy loss to the sum of model1 and model2.
* **--madry**              Boolean for triggering Madry PGD adversarial training. Default: False
* **--epsilon**            Epsilon for linf adversarial training (float or float-convertible expression). Default: 8/255
* **--step-size**          Attack optimization step size for adversarial training (float or float-convertible expression). Default: 2/255
* **--attack-ratio**       Ratio of adversarial samples (alpha in the paper) in adversarial training. Default: 1.

### Results

#### Pdf folder

There is a human readable format of the results, in a pdf version, under
the pdf/ folder.

#### Matrix folder

In the matrix folder, you'll find the saved transformation matrix (all parameters of the stitching layer).

#### Frank model folder
In the frank_model folder you'll find the entire saved stitched model. This should be used for further experiments with a finished stitcher.

## Layer information

If you're not aware of the available layer names to a given model, you can check our cheat sheet:

```console
python src/bin/layer_info.py model_name
```
Example:
```bash
python src/bin/layer_info.py resnet_w3
```

## Evaluation

#### Accuracy
Run the `eval_net.py` script as follows:
```console
python src/bin/eval_net.py path-to-model
```
Example:
```
python src/bin/eval_net.py snapshots/resnet18/CIFAR10/in0-gn0/model.pt
```

#### Robust accuracy
Run the `eval_net_robustness.py` script as follows:
```console
python src/bin/eval_net_robustness.py path-to-model
```
Example:
```
python src/bin/eval_net_robustness.py snapshots/resnet18/CIFAR10/in0-gn0/model.pt
```