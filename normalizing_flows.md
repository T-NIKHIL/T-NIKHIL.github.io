---
layout: page
permalink: /normalizing_flows/
show_excerpts: true
---

<p align='justify'>
    Normalizing Flows are based on a simple formula called 
    the change of variables formula.
    To understand how this formula works, 
    consider a set of samples $x$ drawn 
    from a normal distribution and 
    we have a smooth and invertible function $f$.
    I denote the samples $x$ before transformation as $x_0$
    and those after transformation as $x_{1}$. 
    Now these samples are distributed according to a new distribution $p_{1}$.
    We can compute what this new distribution looks like by applying
    the conservation of probability mass which states that 
    the probability of an event occurring in interval $|dx_{1}|$  
    must equal the probability of event occurring in interval $|dx_{0}|$.
</p>

$$
    p_{1}(x_{1})|dx_{1}| = p_{0}(x_{0})|dx_{0}|
$$

<p align='justify'>
    Further rearranging the terms we get
</p>

$$
    p_{1}(x_{1}) = p_{0}(x_{0}) \frac{|dx_{0}|}{|dx_{1}|}
$$

$$
    p_{1}(x_{1}) = p_{0}(f^{-1}(x_{1})) \frac{|df^{-1}(x_{1})|}{|dx_{1}|}
$$

<p align='justify'>
    The above equation holds is for the 1D case, 
    when dealing with multi-dimensional probability distributions
    we compute the determinant
</p>

$$
    p_{1}(x_{1}) = p_{0}(f^{-1}(x_{1})) | \det \frac{df^{-1}(x_{1})}{dx_{1}} |
$$

<p align='justify'>
    If $f(x)$ is a linear transformation i.e $ x_1 = f(x_0) = A x_0 $ then 
    the change of variables can be written as :
</p>

$$
    p_{1}(x_{1}) = p_{0}(f^{-1}(x_{1})) | \det A^{-1} |
$$

<p align='justify'>
    Lets look at an example to see this formula in action : 
</p>

```python
# Parameters for p0 dist
mu = 0
sigma = 0.5
rng = np.random.default_rng(seed=seed)
bins = 50
samples = 1000

# I use a normal distribution for p0
x0_samples = rng.normal(mu, sigma, size=(samples, 1))
plt.figure(figsize=(5, 3))
plt.hist(x0_samples, bins=bins, label='Empirical p0 dist')
plt.legend()
plt.show()

# Defining a smooth and invertible function f
f = lambda x: np.exp(x) # x**2
f_inv = lambda x : np.log(x) # np.sqrt(x)
df_invdx = lambda x : 1/x # 0.5*(x**(-0.5))

# Generating transformed samples
x1_samples = f(x0_samples)
plt.figure(figsize=(5, 3))
count, bins, ignored = plt.hist(x1_samples, bins=bins, label='Empirical p1 dist')
plt.legend()
plt.show()
```

<div style="text-align: center;">
    <img src="/assets/blogs/normalizing_flows/empirical_p0_p1.pdf" alt="Empirical p0 and p1 distributions" width="300" height="400">
    <figcaption>Figure 1. Empirical $p_0$ and $p_1$ distributions</figcaption>
</div>

```python
def p0(x, mu=0, sigma=1):
  return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((x - mu)**2)/(2*sigma**2))

def logp0(x, mu=0, sigma=1):
  return np.log(1/np.sqrt(2*np.pi*sigma**2)) - ((x - mu)**2)/(2*sigma**2)

plt.figure(figsize=(5, 3))
plt.hist(x1_samples, bins=bins, density=True, alpha=0.6, color='skyblue', label='Empirical p1 dist')
# Plot theoretical PDF using Change of Variables
x1_range = np.linspace(0.1, np.max(bins), samples)
p1 = p0(f_inv(x1_range), mu, sigma) * np.abs(df_invdx(x1_range))
plt.plot(x1_range, p1, 'r-', lw=2, label='Theoretical (Change of Variables)')
plt.legend()
```

<div style="text-align: center;">
    <img src="/assets/blogs/normalizing_flows/theoretical_p1.pdf" alt="Theoretical p1 distribution" width="300" height="200">
    <figcaption>Figure 2. Comparison of empirical $p_1$ distribution versus computed from Change of variables formula</figcaption>
</div>

<h5 id="Sec1"><b> Composing multiple bijections </b></h5>

<p align='justify'>
    Normalizing flows are able to learn complex distributions
    starting from simple priors by applying a sequence of invertible 
    transformations whose Jacobians are much more efficient to
    compute than one single complex invertible mapping.
</p>


$$
  x_{0} \xrightarrow{f_{1}} x_{1} \xrightarrow{f_{2}} \dots \xrightarrow{f_{L}} x_{L}
$$

<p align='justify'>
    Each $f_{k}$ for $k \in \{k=1 \dots L\}$ follows a smooth bijection. 
    If the initial samples $x_{0}$ follows the distribution $p_{0}$ then,
    each subsequent sample follows distributions $p_{1}$ to $p_{L}$. 
    The transformation path of the samples is called <i>flow</i> while 
    the path traced by the distributions is called the <i>normalizing flow</i>. 
    The change of variables formula can be applied to compute the theoretical
    distribution at each stage :
</p>

$$
  p_{k}(x_{k}) = p_{k-1}(x_{k-1})| \det \frac{\partial f_{k}(x_{k-1})}{\partial x_{k-1}}|^{-1}
$$

$$
  p_{k}(x_{k}) = p_{\rm{prior}}(x_{0}) \prod_{i=1}^{L-1} | \det \frac{\partial f_{k}(x_{k})}{\partial x_{k}}|^{-1}
$$

$$
  \log p_{k}(x_{k}) = \log p_{\rm{prior}}(x_{0}) - \sum_{i=1}^{L-1} \log | \det \frac{\partial f_{k}(x_{k})}{\partial x_{k}}|
$$

<p align='justify'>
    The expression reflect how much each transformation stretches or 
    contracts volume as determined by the Jacobian of the determinant. 
    Accumulation of local volume changes along the transformation path 
    serves as the final probability density under the composed map.
</p>

```python
# Example of multiple bijections
x2_samples = f(x1_samples)
plt.figure(figsize=(5, 3))
count, bins, ignored = plt.hist(x2_samples, bins=bins, label='Empirical p2 dist')
plt.legend()
plt.show()

plt.figure(figsize=(5, 3))
plt.hist(x2_samples, bins=bins, density=True, alpha=0.6, color='skyblue', label='Empirical p2 dist')
# Plot theoretical PDF using Change of Variables
x2_range = np.linspace(0.1, np.max(bins), samples)
p2 = p1 * np.abs(df_invdx(x2_range))
plt.plot(x2_range, p1, 'r-', lw=2, label='Theoretical (Change of Variables)')
plt.legend()
```

<div style="text-align: center;">
    <img src="/assets/blogs/normalizing_flows/empirical_and_theoretical_p2.pdf" alt="Empirical and Theoretical p2 distribution" width="300" height="400">
    <figcaption>Figure 3. Empirical $p_2$ (top) and computed theoretical $p_2$ (bottom)</figcaption>
</div>

<p align='justify'>
    Much literature exists that look at composing transformations
    for which the Jacobian calculation is fast. 
    One such example are Residual flows [1][2] where the transformation looks like :
</p>

$$
    x_k = x_{k-1} + v(x_{k-1}) 
$$

<p align='justify'>
    The log-determinant of the jacobian looks like :
</p>

$$
    \log | \det \frac{\partial f_{k}(x_{k-1})}{\partial x_{k-1}}| = \log | \det (I + \frac{\partial v(x_{k-1}) }{\partial x_{k-1}})|
$$

$$
    = \det | \log (I + \frac{\partial v(x_{k-1}) }{\partial x_{k-1}})|
$$

$$
    = \rm{Tr}( \log (I + \frac{\partial v(x_{k-1}) }{\partial x_{k-1}}))
$$

MORE UPDATES TO THIS BLOG COMING SOON !

<h4><b> References </b></h4>

<p align='justify' id="1">[1] Behrmann, J.; Grathwohl, W.; Chen, R. T. Q.; Duvenaud, D.; Jacobsen, J.-H. Invertible Residual Networks. <em>arXiv</em> <b>2019</b> <a href="https://doi.org/10.48550/arXiv.1811.00995">https://doi.org/10.48550/arXiv.1811.00995</a>.</p>

<p align='justify' id="2">[2] Chen, R. T. Q.; Behrmann, J.; Duvenaud, D.; Jacobsen, J.-H. Residual Flows for Invertible Generative Modeling. <em>arXiv</em> <b>2020</b> <a href="https://doi.org/10.48550/arXiv.1906.02735">https://doi.org/10.48550/arXiv.1906.02735</a>.</p>