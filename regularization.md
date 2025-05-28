---
layout: page
permalink: /regularization/
show_excerpts: true
---

<p align='justify'>
    As scientists and engineers when we build a mathemetical model 
    to describe a process or system, we often start with the simplest
    explanation or model. This line of thinking is pervasive throughout
    science and engineering and goes by many names (Ex : Occam's razor).
    Simple models are not only more intuitive and interpretable, but are also
    more likely to generalize. This guiding principle has led to several 
    strategies of training models that learn from data and collectively fall 
    under the name of <em>regularization</em>.
</p>

<p align="center">
    <img src="/assets/occams_razor.pdf" alt="Occam's Razor">
</p>

### 1. Parameter Norm Penalties

Say we have a dataset with 2 variables ($x_1$ and $x_2$) and a 
dependent variable $y$ that linearly depends on the two independent variables as $ y = 3*x_1 + 2*x_2 + N(0,0.5) $.
The black dots in the below plot are the observations while the red
surface depicts the model without the gaussian noise.

<iframe id="igraph" scrolling="no" 
style="border:none; margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/linear_surface.html"
height="400" width="400"></iframe>

Lets define a linear model as $ \hat{y} = w_1 * x_1 + w_2 * x_2 $.
Let's also define a loss function that will tell us how good our model
parameters $w_1$ and $w_2$ are. For this lets use the mean squared error
(MSE) loss function.
 
$$
J = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

```python
def loss_fn(w, dataset):
    return np.mean((dataset[:, -1] - (w[0, 0]*dataset[:, 0] + w[1, 0]*dataset[:, 1]))**2)

class LinearModel():
    def __init__(self):
        self.w = None

    def forward(self, x, w, save=True):
        if save:
            self.w = w
        return np.dot(x, self.w)
```

Lets explore how this loss function looks like. 

<iframe id="igraph" scrolling="no" 
style="border:none; margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/loss_surface.html" height="400" width="400"></iframe>

