# Adversarial-Inspired Backdoor Defense via Bridging Backdoor and Adversarial Attacks

Pytorch implement of our paper [**Adversarial-Inspired Backdoor Defense via Bridging Backdoor and Adversarial Attacks - link**] accepted by AAAI2025. In this paper, we focus on bridging backdoor and adversarial attacks and observe two intriguing phenomena when applying adversarial attacks on an infected model implanted with backdoors: 1) the sample is harder to be turned into an adversarial example when the trigger is presented; 2) the adversarial examples generated from backdoor samples are highly likely to be predicted as its true labels. Inspired by these observations, we proposed a novel backdoor defense method, dubbed Adversarial-Inspired Backdoor Defense (AIBD), to isolate the backdoor samples by leveraging a progressive top-q scheme and break the correlation between backdoor samples and their target labels using adversarial labels. The Algorithm pipeline can refer to Algorithm 1 of this paper and the related legend of the observed phenomenon is as follows:

<div style="text-align: center;">
    <img src="redemeImg/legend of the observed phenomenon.png"/>
</div>

## Requirements

- numpy==1.22.4
- torchattacks==3.5.1
- torch==1.11.0
- torchvision==0.12.0
- tqdm==4.61.2

## Run

1. You can run the **AIBD** directly with `defense.py`. We run the script with RTX 3060.

2. You can use the command line configuration to run the **AIBD** to defend against different attack methods. The detailed configuration parameters are in config.py：

   ​	python defense.py --trigger_list ['badnets']

## Citation

## Help

When you have any question/idea about the code/paper. Please comment in Github or send us Email. We will reply as soon as possible.

