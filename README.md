# Simultaneous Preference and Metric Learning from Paired Comparisons
Demo code for "Simultaneous Preference and Metric Learning from Paired Comparisons" ([NeurIPS 2020](https://arxiv.org/abs/2009.02302)) by Austin Xu, Mark A. Davenport.

## Paper Abstract
A popular model of preference in the context of recommendation systems is the so-called ideal point model. In this model, a user is represented as a vector **u** together with a collection of items **x<sub>1</sub>** , . . . , **x<sub>N</sub>** in a common low-dimensional space. The vector **u** represents the user’s “ideal point,” or the ideal combination of features that represents a hypothesized most preferred item. The underlying assumption in this model is that a smaller distance between **u** and an item **x<sub>j</sub>** indicates a stronger preference for **x<sub>j</sub>**. In the vast majority of the existing work on learning ideal point models, the underlying distance has been assumed to be Euclidean. However, this eliminates any possibility of interactions between features and a user’s underlying preferences. In this paper, we consider the problem of learning an ideal point representation of a user’s preferences when the distance metric is an unknown Mahalanobis metric. Specifically, we present a novel approach to estimate the user’s ideal point **u** and the Mahalanobis metric from paired comparisons of the form “item **x<sub>i</sub>** is preferred to item **x<sub>j</sub>**.” This can be viewed as a special case of a more general metric learning problem where the location of some points are unknown a priori. We conduct extensive experiments on synthetic and real-world datasets to exhibit the effectiveness of our algorithm

## Requirements
* [CVX](http://cvxr.com/cvx/download/)

## Code
* `generate_params.m`: Generates ideal point, metric, embedding of items, paired comparisons
* `learn_Md.m`: Single-step estimation (requires CVX)
* `alt_Mu.m`: Alternating estimation (requires CVX)
* `main.m`: Example of how to use `generate_params.m`, `learn_Md.m`, and `alt_Mu.m`
