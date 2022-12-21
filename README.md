# Wiggling-Weights-to-Improve-the-Robustness-of-Classifiers

This is the official implementation of
[Wiggling-Weights-to-Improve-the-Robustness-of-Classifiers](https://arxiv.org/pdf/2111.09779.pdf) 



**Abstract** 
*Robustness against unwanted perturbations is an important aspect of deploying neural network classifiers in the real world. Common natural perturbations include noise, saturation, occlusion, viewpoint changes, and blur deformations. All of them can be modelled by the newly proposed transform-augmented convolutional networks. While many approaches for robustness train the network by providing augmented data to the network, we aim to integrate perturbations in the network architecture to achieve improved and more general robustness. To demonstrate that wiggling the weights consistently improves classification, we choose a standard network and modify it to a transform-augmented network. On perturbed CIFAR-10 images, the modified network delivers a better performance than the original network. For the much smaller STL-10 dataset, in addition to delivering better general robustness, wiggling even improves the classification of unperturbed, clean images substantially. We conclude that wiggled transform-augmented networks acquire good robustness even for perturbations not seen during training.*
<img src="https://github.com/sadafgulshad1/Wiggling-Weights-to-Improve-the-Robustness-of-Classifiers/blob/main/Architecture.png"  />

Rotation scaling transform and local elastic transform:

<img src="https://github.com/sadafgulshad1/Wiggling-Weights-to-Improve-the-Robustness-of-Classifiers/blob/main/Transforms.gif"  />

## Experiments
To reproduce the experiments on STL10 follow the instructions below

For training in the scripts folder run 
```
bash train.sh
```

For testing in the scripts folder run 
```
bash test.sh
```

## Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.

## BibTeX
If you found this work useful in your research, please consider citing
```
@article{gulshad2021wiggling,
  title={Wiggling Weights to Improve the Robustness of Classifiers},
  author={Gulshad, Sadaf and Sosnovik, Ivan and Smeulders, Arnold},
  journal={arXiv e-prints},
  pages={arXiv--2111},
  year={2021}
}
```
or 

```
@article{gulshad2021built,
  title={Built-in Elastic Transformations for Improved Robustness},
  author={Gulshad, Sadaf and Sosnovik, Ivan and Smeulders, Arnold},
  journal={arXiv e-prints},
  pages={arXiv--2107},
  year={2021}
}
```
