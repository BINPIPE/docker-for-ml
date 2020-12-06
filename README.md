# MACHINE LEARNING WITH DOCKER    |[![BINPIPE](https://img.shields.io/badge/YouTube-red.svg)](https://www.youtube.com/channel/UCPTgt4Wo0MAnuzNEEZlk90A)
Demonstrations about Using Docker for Machine Learning

## Machine learning development environment: Bare necessities

Let’s start with what the four basic ingredients you need for a machine learning development environment:

1.  **Compute:**  High-performance CPUs and GPUs to train models.
2.  **Storage:** For large training datasets and metadata you generated during training.
3.  **Frameworks and libraries:**  To provide APIs and execution environment for training.
4.  **Source control:** For collaboration, backup, and automation.

As a machine learning researcher, developer, or data scientist, you can set up an environment with these four ingredients on a single  [Amazon Elastic Compute Cloud (Amazon EC2)](https://aws.amazon.com/ec2/)  instance or a workstation at home.

![Basic ingredients for a machine learning development environment](https://d2908q01vomqb2.cloudfront.net/ca3512f4dfa95a03169c5a670a4c91a19b3077b4/2020/03/11/prasanna_f1_2020_03_11.png)

So, what’s wrong with this setup?

Nothing really, most development setups have looked like this for decades—no clusters, no shared file systems.

Except for a small community of researchers in High-Performance Computing (HPC) who develop code and run them on supercomputers, the rest of us rely on our own dedicated machines for development.

As it turns out, machine learning has more in common with HPC than it does with traditional software development. Like HPC workloads, machine learning workloads can benefit from faster execution and quicker experimentation when running on a large cluster. To take advantage of a cluster for machine learning training, you’ll need to make sure your development environment is portable and training is reproducible on a cluster.

## Why you need portable training environments

At some point in your machine learning development process, you’ll hit one of these two walls:

1.  You’re experimenting and you have too many variations of your training scripts to run, and you’re bottlenecked by your single machine.
2.  You’re running training on a large model with a large dataset, and it’s not feasible to run on your single machine and get results in a reasonable amount of time.

These are two common reasons why you may want to run machine learning training on a cluster. These are also reasons why scientists use supercomputers such as  [Summit supercomputer](https://en.wikipedia.org/wiki/Summit_(supercomputer))  to run their scientific experiments. To address the first wall, you can run every model independently and asynchronously on a cluster of computers. To address the second wall, you can distribute a single model on a cluster and train it faster.

Both these solutions require that you be able to  **successfully and consistently reproduce your development training setup on a cluster**. And that’s challenging because the cluster could be running different operating systems and kernel versions; different GPUs, drivers and runtimes; and different software dependencies than your development machine.

Another reason why you need portable machine learning environments is for collaborative development. Sharing your training scripts with your collaborator through version control is easy. Guaranteeing reproducibility without sharing your full execution environment with code, dependencies, and configurations is harder, as we’ll see in the next section.

## Machine learning, open source, and specialized hardware

A challenge with machine learning development environments is that they rely on complex and continuously evolving open source machine learning frameworks and toolkits, and complex and continuously evolving hardware ecosystems. Both are positive qualities that we desire, but they pose short-term challenges.

![diagram showing Your code has more dependencies than you think.](https://d2908q01vomqb2.cloudfront.net/ca3512f4dfa95a03169c5a670a4c91a19b3077b4/2020/03/11/prasanna_f2_2020_03_11_150.png)

How many times have you run machine learning training and asked yourselves these questions:

-   Is my code taking advantage of all available resources on CPUs and GPUs?
-   Do I have the right hardware libraries? Are they the right versions?
-   Why does my training code work fine on my machine, but crashes on my colleague’s, when the environments are more or less identical?
-   I updated my drivers today and training is now slower/errors out. Why?

If you examine your machine learning software stack, you will notice that you spend most of your time in the magenta box called  **My code**  in the accompanying figure. This includes your training scripts, your utility and helper routines, your collaborators’ code, community contributions, and so on. As if that were not complex enough, you also would notice that your dependencies include:

-   the machine learning framework API that is evolving rapidly;
-   the machine learning framework dependencies, many of which are independent projects;
-   CPU-specific libraries for accelerated math routines;
-   GPU-specific libraries for accelerated math and inter-GPU communication routines; and
-   GPU driver that needs to be aligned with the GPU compiler used to compile above GPU libraries.

Due to the high complexity of an open source machine learning software stack, when you move your code to a collaborator’s machine or a cluster environment, you introduce multiple points of failure. In the figure below, notice that even if you control for changes to your training code and the machine learning framework, there are lower-level changes that you may not account for, resulting in failed experiments.

Ultimately, this costs you the most precious commodity of all— your time.

![Migrating training code isn't the same as migrating your entire execution environment. Dependencies potentially introduce multiple points of failure when moving from development environment to training infrastructure](https://d2908q01vomqb2.cloudfront.net/ca3512f4dfa95a03169c5a670a4c91a19b3077b4/2020/03/11/prasanna_f3_2020_03_11.png)

**Why not virtual Python environments?**

You could argue that virtual environment approaches such as  [conda](https://docs.conda.io/en/latest/)  and  [virtualenv](https://virtualenv.pypa.io/en/latest/)  address these issues. They do, but only partially. Several non-Python dependencies are not managed by these solutions. Due to the complexity of a typical machine learning stack, a large part of framework dependencies, such as hardware libraries, are outside the scope of virtual environments.

## Enter containers for machine learning development

Machine learning software is part of a fragmented ecosystem with multiple projects and contributors. That can be a good thing, as everyone one benefits from everyone’s contributions, and developers always have plenty of options. The downside is dealing with problems such as consistency, portability, and dependency management. This is where container technologies come in. In this article, I won’t discuss the general benefits of containers, but I will share how machine learning benefits from them.

Containers can fully encapsulate not just your training code, but the entire dependency stack down to the hardware libraries. What you get is a machine learning development environment that is consistent and portable. With containers, both collaboration and scaling on a cluster becomes much easier. If you develop code and run training in a container environment, you can conveniently share not just your training scripts, but your entire development environment by pushing your container image into a container registry, and having a collaborator or a cluster management service pull the container image and run it to reproduce your results.

![Containers allow you to encapsulate all your dependencies into a single package that you can push to a registry and make available for collaborators and orchestrators on a training cluster](https://d2908q01vomqb2.cloudfront.net/ca3512f4dfa95a03169c5a670a4c91a19b3077b4/2020/03/11/prasanna_f4_2020_03_11.png)

## What you should and shouldn’t include in your machine learning development container

There isn’t a right answer and how your team operates is up to you, but there are a couple of options for what to include:

1.  **Only the machine learning frameworks and dependencies:**  This is the cleanest approach. Every collaborator gets the same copy of the same execution environment. They can clone their training scripts into the container at runtime or mount a volume that contains the training code.
2.  **Machine learning frameworks, dependencies, and training code:**  This approach is preferred when scaling workloads on a cluster. You get a single executable unit of machine learning software that can be scaled on a cluster. Depending on how you structure your training code, you could allow your scripts to execute variations of training to run hyperparameter search experiments.

Sharing your development container is also easy. You can share it as a:

1.  **Container image:**  This is the easiest option. This allows every collaborator or a cluster management service, such as Kubernetes, to pull a container image, instantiate it, and execute training immediately.
2.  **Dockerfile:**  This is a lightweight option.  [Dockerfiles](https://docs.docker.com/engine/reference/builder/)  contain instructions on what dependencies to download, build, and compile to create a container image. Dockerfiles can be versioned along with your training code.

---
## ML terminologies:

**Machine learning** is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

**Batch inference** or offline inference, is the process of generating predictions on a batch of observations. The batch jobs are typically generated on some recurring schedule (e.g. hourly, daily). These predictions are then stored in a database and can be made available to developers or end users. Batch inference may sometimes take advantage of big data technologies such as Spark to generate predictions. This allows data scientists and machine learning engineers to take advantage of scalable compute resources to generate many predictions at once.
```
+---------------+   +--------------+   +--------------+   +--------------+
| Observation|1 |   |Observation|2 |   |Observation|3 |   |Observation|N |
+----------+----+   +------+-------+   +--------+-----+   +--------+-----+
           |               |                    |                  |
           |               |                    v                  |
           +>--------------v------------+-------+--<---------------+
                                        |
                                        |
                                        |
                               +--------+---------+
                               |      MODEL       |
                               |                  |
                               +---------+--------+
                                         |
                                         |
       ++-----------------+--------------v----+----------------+------+
        |                 |                   |                |
        |                 |                   |                |
        |                 |                   |                |
  +-----v-------+  +------v------+  +---------v---+  +---------v---+
  |Prediction|1 |  |Prediction|2 |  |Prediction|3 |  |Prediction|N |
  +-------------+  +-------------+  +-------------+  +-------------+

```


**Online Inference** is the process of generating machine learning predictions in real time upon request. It is also known as real time inference or dynamic inference. Typically, these predictions are generated on a single observation of data at runtime. Predictions generated using online inference may be generated at any time of the day.

```
 +-------------+
 | OBSERVATION |
 +------+------+
        |
        |
        |
+-------v------+          +--------+
|              +--------> |        |
|  REST API    |          | MODEL  |
|              | <--------+        |
+-------+------+          +--------+
        |
        |
        v
  +-----+------+
  |PREDICTION  |
  +------------+

```


## Activation Function

An activation, or activation function,  for a  **neural network**  is defined as the mapping of the input to the output via a non-linear transform function at each “node”, which is simply a locus of computation within the net. Each layer in a neural net consists of many nodes, and the number of nodes in a layer is known as its width.

[deeplearning4j.org](http://deeplearning4j.org/glossary.html)

## Artificial Neural Network

An  **artificial neuron network**  (ANN) is a computational model based on the structure and functions of  **biological neural networks**. Information that flows through the network affects the structure of the ANN because a neural network changes — or learns, in a sense — based on that input and output.

[techopedia.com](https://www.google.ca/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjDuq7bjJ7OAhUr4IMKHdNZBmQQFgglMAE&url=https%3A%2F%2Fwww.techopedia.com%2Fdefinition%2F5967%2Fartificial-neural-network-ann&usg=AFQjCNHjBtPlMh7tuZYEE0xJDAqkUtrkAQ)

## Back propagation

**Backpropagation**  is an algorithm to efficiently calculate the gradients in a Neural Network, or more generally, a feedforward computational graph. It boils down to applying the chain rule of  **differentiation**  starting from the network output and propagating the gradients backward.

[wildml.com](http://www.wildml.com/deep-learning-glossary/)

## Bayesian Optimization

**Bayesian Optimization**  is an optimization algorithm that is used when three conditions are met:

1.  you have a black box that evaluates the objective function
2.  you do not have a black box (or any box) that evaluates the gradient of your objective function
3.  your objective function is expensive to evaluate.

[cs.ubc.ca](http://www.cs.ubc.ca/~mgelbart/glossary.html#bayesopt)

## Classification

**Classification**  is concerned with building models that separate data into distinct classes. Well-known classification schemes include decision trees and support vector machines. As this type of algorithm requires explicit class labelling, classification is a form of  **supervised learning**.

[kdnugget.com](http://www.kdnuggets.com/2016/05/machine-learning-key-terms-explained.html)

## Clustering

**Clustering**  is used for analyzing data which does not include pre-labeled classes, or even a class attribute at all. Data instances are grouped together using the concept of “maximizing the intraclass similarity and minimizing the interclass similarity,” as concisely described by Han, Kamber & Pei. k-means clustering is perhaps the most well-known example of a clustering algorithm. As clustering does not require the pre-labeling of instance classes, it is a form of  **unsupervised learning**, meaning that it learns by observation as opposed to learning by example.

[kdnugget.com](http://www.kdnuggets.com/2016/05/machine-learning-key-terms-explained.html)

## Convolution

For mathematical purposes, a convolution is the integral measuring how much two functions  **overlap**  as one passes over the other. Think of a convolution as a way of  **mixing two functions by multiplying them**: a fancy form of multiplication. From the Latin  **convolvere**, “to convolve” means to roll together.

[deeplearning4j.org](http://deeplearning4j.org/glossary.html)

## Convolutional neural network

**Convolutional networks**  are a deep neural network that is currently the state-of-the-art in image processing. They are setting new records in accuracy every year on widely accepted benchmark contests like ImageNet.

[deeplearning4j.org](http://deeplearning4j.org/glossary.html)

## Cross Validation

**Cross-validation**  is a deterministic method for model building, achieved by leaving out one of k segments, or folds, of a dataset, training on all k-1 segments, and using the remaining kth segment for testing; this process is then repeated k times, with the individual prediction error results being combined and averaged in a single, integrated model. This provides variability, with the goal of producing the most accurate predictive models possible.

[kdnugget.com](http://www.kdnuggets.com/2016/05/machine-learning-key-terms-explained.html)

## Feature Engineering

**Feature engineering**  is the art of extracting useful patterns from data that will make it easier for Machine Learning models to distinguish between classes. For example, you might take the number of greenish vs. bluish pixels as an indicator of whether a land or water animal is in some picture. This feature is helpful for a machine learning model because it limits the number of classes that need to be considered for a  **good classification**.

[wildml.com](http://www.wildml.com/deep-learning-glossary/)

## Gradient Descent

**Gradient descent**  is a first-order iterative optimization algorithm. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point.

[wikipedia.com](https://www.google.ca/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwjgoa7Ki57OAhVK7IMKHZxMCq8QFggkMAI&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FGradient_descent&usg=AFQjCNEB7szBsRwTf-gol1jdLEMPb9r-UA)

## Deep Belief Network

A  **deep-belief network**  is a stack of restricted  **Boltzmann**  machines, which are themselves a feed-forward autoencoder that learns to reconstruct input layer by layer, greedily. Pioneered by Geoff Hinton and crew. Because a DBN is deep, it learns a hierarchical representation of input. Because DBNs learn to reconstruct that data, they can be useful in  **unsupervised learning.**

[deeplearning4j.org](http://deeplearning4j.org/glossary.html)

## Decision Trees

**Decision trees** are top-down, recursive, divide-and-conquer classifiers. Decision trees are generally composed of 2 main tasks:

-   **Tree induction** is the task of taking a set of pre-classified instances as input, deciding which attributes are best to split on, splitting the dataset, and recursing on the resulting split datasets until all training instances are categorized.
-   **Tree pruning** is the process of removing the unnecessary structure from a decision tree in order to make it more efficient, more easily-readable for humans, and more accurate as well. This increased accuracy is due to pruning’s ability to reduce overfitting.
-   [kdnugget.com](http://www.kdnuggets.com/2016/05/machine-learning-key-terms-explained.html)

## Inference

**Inference**  refers inferring unknown properties of the real world. This can be confusing both because learning and inference refer to figuring out unknown things. An example from human vision: when you see something, your brain receives a bunch of pixels and you need to infer what it is that you are seeing (e.g., a tree).

[cs.ubc.ca](http://www.cs.ubc.ca/~mgelbart/glossary.html#bayesopt)

## Layer

A  **layer**  is the highest-level building block in a **(Deep) Neural Network**. A layer is a container that usually receives weighted input, transforms it and returns the result as output to the next layer. A layer usually contains one type of function like ReLU, pooling, convolution etc. so that it can be easily compared to other parts of the network. The first and last layers in a network are called  **input and output layers**, respectively, and all layers in between are called hidden layers.

[leaf deep learning glossary](https://github.com/autumnai/leaf/blob/master/doc/src/deep-learning-glossary.md)

## Learning

**Learning**  refers to learning unknown parameters in your model. Your model might be wrong, so these parameters might have nothing to do with the real world — but they can still be useful in modeling your system

[cs.ubc.ca](http://www.cs.ubc.ca/~mgelbart/glossary.html#bayesopt)

## LSTM Long Short-Term Memory

**Long Short-Term Memory networks**  were invented to prevent the vanishing gradient problem in Recurrent Neural Networks by using a memory gating mechanism. Using LSTM units to calculate the hidden state in an RNN we help to the network to efficiently propagate gradients and learn long-range dependencies.

[wildml.com](http://www.wildml.com/deep-learning-glossary/)

## Overfitting

This is when you learn something **too specific**  to your data set and therefore fail to generalize well to unseen data. This causes it to do badly for unseen data. Another example is if you are a student and you memorize the answers to all 1000 questions in your textbook without understanding the material.

[cs.ubc.ca](http://www.cs.ubc.ca/~mgelbart/glossary.html#bayesopt)

## Pooling

**Pooling, max pooling and average pooling**  are terms that refer to downsampling or subsampling within a convolutional network. Downsampling is a way of reducing the amount of data flowing through the network, and therefore decreasing the computational cost of the network. Average pooling takes the average of several values. Max pooling takes the greatest of several values. Max pooling is currently the preferred type of downsampling layer in convolutional networks.

[deeplearning4j.org](http://deeplearning4j.org/glossary.html)

## Normalization

**Normalization**  refers to rescaling real valued numeric attributes into the  **range 0 and 1.**

It is useful to scale the input attributes for a model that relies on the magnitude of values, such as distance measures used in k-nearest neighbours and in the preparation of coefficients in regression.

[machinelearningmastery.com](http://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/)

## Regression

**Regression**  is very closely related to classification. While classification is concerned with the prediction of discrete classes, regression is applied when the “class” to be predicted is made up of continuous numerical values. Linear regression is an example of a regression technique.

[kdnugget.com](http://www.kdnuggets.com/2016/05/machine-learning-key-terms-explained.html)

## Recurrent Neural Network

A  **RNN**  models sequential interactions through a hidden state, or memory. It can take up to N inputs and produce up to N outputs. At each time step, an  **RNN**  calculates a new hidden state (“memory”) based on the current input and the previous hidden state. The “recurrent” stems from the facts that at each step the same parameters are used and the network performs the same calculations based on different inputs.

[wildml.com](http://www.wildml.com/deep-learning-glossary/)

## Recursive Neural Network

**Recursive neural networks** learn data with structural hierarchies, such as text arranged grammatically, much like recurrent neural networks learn data structured by its occurrence in time. Their chief use is in natural-language processing, and they are associated with  **Richard Socher** of Stanford’s NLP lab.

[deeplearning4j.org](http://deeplearning4j.org/glossary.html)

## Word2Vec

**Tomas Mikolov’s neural networks**, known as Word2vec, have become widely used because they help produce state-of-the-art  **word embeddings.**  Word2vec is a two-layer neural net that processes text. Its input is a text corpus and its output is a set of vectors: feature vectors for words in that corpus.

[deeplearning4j.org](http://deeplearning4j.org/glossary.html)

## Vector

A  **vector**  is a data structure with at least two components, as opposed to a scalar, which has just one. For example, a vector can represent velocity, an idea that combines speed and direction: wind velocity = (50mph, 35 degrees North East). A scalar, on the other hand, can represent something with one value like temperature or height: 50 degrees Celsius, 180 centimeters.

[deeplearning4j.org](http://deeplearning4j.org/glossary.html)


// Sources: Excerpts from blog post by Prasanjit Singh (binpipe.org), Shashank Prasanna (AWS Blog) & Luigi Patruno (ML in Production Blog) + ML terminologies from multiple sources attributed above.
___
:ledger: Maintainer: **[Prasanjit Singh](https://www.linkedin.com/in/prasanjit-singh)** | **www.binpipe.org**   [![BINPIPE](https://img.shields.io/badge/YouTube-red.svg)](https://www.youtube.com/channel/UCPTgt4Wo0MAnuzNEEZlk90A)
___ 
