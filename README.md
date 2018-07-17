# LeakGAN
The code of research paper [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624).

This paper has been accepted at the Thirty-Second AAAI Conference on Artificial Intelligence ([AAAI-18](https://aaai.org/Conferences/AAAI-18/)).
## Requirements
* **Tensorflow r1.2.1**
* Python 3.4+
* CUDA 7.5+ (For GPU)

## Introduction
Automatically generating coherent and semantically meaningful text has many applications in machine translation, dialogue systems, image captioning, etc. Recently, by combining with policy gradient, Generative Adversarial Nets (GAN) that use a discriminative model to guide the training of the generative model as a reinforcement learning policy has shown promising results in text generation. However, the scalar guiding signal is only available after the entire text has been generated and lacks intermediate information about text structure during the generative process. As such, it limits its success when the length of the generated text samples is long (more than 20 words). In this project, we propose a new framework, called LeakGAN, to address the problem for long text generation. We allow the discriminative net to leak its own high-level extracted features to the generative net to further help the guidance. The generator incorporates such informative signals into all generation steps through an additional Manager module, which takes the extracted features of current generated words and outputs a latent vector to guide the Worker module for next-word generation. Our extensive experiments on synthetic data and various real-world tasks with Turing test demonstrate that LeakGAN is highly effective in long text generation and also improves the performance in short text generation scenarios. More importantly, without any supervision, LeakGAN would be able to implicitly learn sentence structures only through the interaction between Manager and Worker.

![](https://github.com/CR-Gjx/LeakGAN/blob/master/figures/leakgan.png)

As the illustration of LeakGAN. We specifically introduce a hierarchical generator G, which consists of a high-level MANAGER module and a low-level WORKER module. The MANAGER is a long short term memory network (LSTM) and serves as a mediator. In each step, it receives generator D’s high-level feature representation, e.g., the feature map of the CNN, and uses it to form the guiding goal for the WORKER module in that timestep. As the information from D is internally-maintained and in an adversarial game it is not supposed to provide G with such information. We thus call it a leakage of information from D.

Next, given the goal embedding produced by the MANAGER, the WORKER firstly encodes current generated words with another LSTM, then combines the output of the LSTM and the goal embedding to take a final action at current state. As such, the guiding signals from D are not only available to G at the end in terms of the scalar reward signals, but also available in terms of a goal embedding vector during the generation process to guide G how to get improved.

## Reference
```bash
@article{guo2017long,
  title={Long Text Generation via Adversarial Training with Leaked Information},
  author={Guo, Jiaxian and Lu, Sidi and Cai, Han and Zhang, Weinan and Yu, Yong and Wang, Jun},
  journal={arXiv preprint arXiv:1709.08624},
  year={2017}
}
```


You can get the code and run the experiments in follow folders.
## Folder

Synthetic Data: synthetic data experiment

Image COCO: a real text example for our model using dataset Image COCO (http://cocodataset.org/#download)

Note: this code is based on the [previous work by LantaoYu](https://github.com/LantaoYu/SeqGAN). Many thanks to [LantaoYu](https://github.com/LantaoYu).


## File

LeakGANModel.py : The generator model of LeakGAN including Manager and Worker.

Discriminator.py: The discriminator model of LeakGAN including Feature Extractor and classification.

data_loader.py: Data helpy function for this experiment.

Main.py: The Main function of this experiment.

convert.py: The convert one-hot number to real word.

eval_bleu.py: Evaluation the BLEU scores (2-5) between test datatset and generated data.

## Details
We provide example codes to repeat the synthetic data experiments with oracle evaluation mechanisms in length 20.
To run the experiment with default parameters for length 20:
```
$ python Main.py
```

In our code, we
The experiment has two stages. In the first stage, use the positive data provided by the oracle and Maximum Likelihood Estimation to perform supervise learning. In the second stage, use adversarial training to improve the generator.

When you running the code, the pre-train model will be store in folder ``ckpts``, if you want to restore the pre-trained discriminator model, you can run:
```
$ python Main.py --resD=True --model=leakgan_preD
```

if you want to restore all pre-traine model or unsupervised model (store model every 30 epoch named ``leakgan-31`` or other number), you can run:
```
$ python Main.py --restore=True --model=leakgan_pre
```

After running the experiments, you can run the ``convert.py`` to obtain the real sentence in folder ``speech``.You also can run the ``eval_bleu.py`` to acquire the BLEU score in your command line.
The generated examples store in folder ``save`` every 30 epochs, and the file named ``coco_31.txt`` or other numbers.

