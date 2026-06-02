---
layout: page
permalink: /normalizing_flows/
show_excerpts: true
---

<p align='justify'>
    Normalizing Flows are based on the change of variables formula.
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
    Further rearranging the terms and substituting $x_0 = f^{-1}(x_1)$
</p>

$$
    p_{1}(x_{1}) = p_{0}(x_{0}) \frac{|dx_{0}|}{|dx_{1}|}
$$

$$
    p_{1}(x_{1}) = p_{0}(f^{-1}(x_{1})) \frac{|df^{-1}(x_{1})|}{|dx_{1}|}
$$

<p align='justify'>
    The above equation holds for the 1D case but
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
    The transformation path of the samples is called <b>flow</b> while 
    the path traced by the distributions is called the <b>normalizing flow</b>. 
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
    The expression reflects how much each transformation stretches or 
    contracts volume as determined by the Jacobian.
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
    Much literature exists that looks at composing transformations
    for which the Jacobian calculation is fast. 
    One such example are Residual flows [1][2] where the transformation looks like :
</p>

$$
    x_k = f(x_{k-1}) = x_{k-1} + g(x_{k-1})
$$

<p align='justify'>
    The log-determinant of the jacobian looks like :
</p>

$$
    \log | \det \frac{\partial f_{k}(x_{k-1})}{\partial x_{k-1}}| = \log | \det (I + \frac{\partial g(x_{k-1}) }{\partial x_{k-1}})|
$$

$$
    = \det | \log (I + \frac{\partial g(x_{k-1}) }{\partial x_{k-1}})|
$$

$$
    = \textrm{Tr}( \log (I + \frac{\partial g(x_{k-1}) }{\partial x_{k-1}}))
$$

<h5 id="Sec2"><b> From discrete time updates to continuous time updates (Neural Ordinary Differential Equations (NODES))  </b></h5>

<p align='justify'>
    In the limit of infinite layers or transformations the updates
</p>

$$
    x_k = f(x_{k-1}) = x_{k-1} + g(x_{k-1}) 
$$

<p align='justify'>
    can be formulated within the framework of <b>NODES</b> 
    (also called as <b>Continuous Normalizing Flows</b>).
</p>

$$
    \frac{dx_t}{dt} = g_{\phi}(x_{t}, t) 
$$

<p align='justify'>
    The goal is to train a neural network that can learn 
    this velocity field $g_{\phi}(x_{t}, t)$ which transport 
    points starting from  $p_{\rm{prior}}(x_{0}, 0)$ to the
    desired data distribution $p_{\rm{data}}(.)$.
    A continuous time variant of the discrete change of variables
    formula was introduced by Chen <i>et. al.</i>[3] and
    is referred to as the <b>instantaneous change of variables</b>. 
    It can be derived from the discrete change of variables 
    formula but is a bit involved which is why I 
    recommend the reader to check out the derivation for
    themselves in Appendix A of the paper [3].
</p>

$$
    \frac{\partial \log p(x_t,t)}{\partial t} = - \nabla_x . g_{\phi}(x_{t}, t) 
$$

<p align='justify'>
    The log density of the terminal PDF induced 
    by the neural ODE $ \frac{dx_t}{dt} = g_{\phi}(x_{t}, t) $
    is given by 
</p>

$$
    \log p(x_T, T) = \log p_{\rm{prior}}(x_0, 0) - \int_0^T \nabla_x . g_{\phi}(x_{t}, t) dt
$$

<p align='justify'>
    which allows one to construct objective
    functions $\mathcal{L}_{\rm{NODE}} = \mathbb{E}_{x \sim p_{\rm{data}}} [\log p(x_T, T)] $ that maximize the likelihood and 
    generate gradients for updating the model parameters $\phi$.
    Maximizing $\mathcal{L}_{\rm{NODE}}$ requires backpropogating
    through the ODE solver. 
    The adjoint sensitivity method computes gradients via an 
    auxiliary ODE with O(1) memory complexity, 
    but NODE training remains expensive due to numerical
    integration at each step.
</p>

>
The instantaneous change of variables is 
a special case of the Fokker-Planck equation with 
$D(x_t, t) = 0$.
>
$$
    \begin{align}
        \frac{\partial p(x_t, t)}{\partial t} &= -\nabla_x . (p(x_t, t)g(x_t, t)) + \nabla_x^2(p(x_t, t)D(x_t, t)) \\
        \frac{\partial p(x_t, t)}{\partial t} &= -\nabla_x . (p(x_t, t)g(x_t, t)) \\
        \frac{\partial p(x_t, t)}{\partial t} &= -g(x_t, t)\nabla_x p(x_t, t) - p(x_t, t)\nabla_x g(x_t, t)  \quad \textrm{(PRODUCT RULE)} \\
    \end{align}
$$
>
From the chain rule of partial derivatives we can write $\frac{\partial p(x_t, t)}{\partial t}$ as
>
$$
    \begin{align}
        \frac{\partial p(x_t, t)}{\partial t} &= \frac{\partial p(x_t, t)}{\partial x_t}\frac{\partial x_t}{\partial t} + \frac{\partial p(x_t, t)}{\partial t} \\
        \frac{\partial p(x_t, t)}{\partial t} &= \frac{\partial p(x_t, t)}{\partial x_t}g(x_t, t) + \frac{\partial p(x_t, t)}{\partial t}
    \end{align}
$$
>
Substituting the product rule in the chain rule we get the instantaneous change of variables formula
>
$$
    \frac{\partial p(x_t, t)}{\partial t} = \cancel{g(x_t, t)\nabla_x p(x_t, t)} - \cancel{g(x_t, t)\nabla_x p(x_t, t)} - p(x_t, t)\nabla_x g(x_t, t){\partial t}
$$
$$  
    \frac{\partial \log p(x_t, t)}{\partial t} = -\nabla_x g(x_t, t)
$$
>

<h5 id="Sec3"><b> Viewing datapoints as particles </b></h5>

<p align='justify'>
    The <b>Langevin equation</b> describes the forces acting on a particle as a sum of a viscous force proportional to the particle's velocity $\gamma v$ and a noise term $dW_t$.
</p>

$$
  m\frac{dv}{dt} = -\gamma v + F(x) + \sigma dW_t
$$

<p align='justify'>
    At equilibrium when the frictional forces $|\gamma v|$ dominate the inertial forces $|ma|$ then we get the <b>overdamped Langevin Equation</b> 
</p>

$$
    \begin{align}
        \gamma \frac{dx}{dt} &= F(x) + \sigma dW_t \\
        \frac{dx}{dt} &= \frac{1}{\gamma} F(x) + \frac{\sigma}{\gamma} dW_t \\
        \frac{dx}{dt} &= -\frac{1}{\gamma} \frac{dU}{dx} + \frac{\sigma}{\gamma} dW_t
    \end{align}
$$

<p align='justify'>
    From the <a href="https://en.wikipedia.org/wiki/Einstein_relation_(kinetic_theory)">Einstein relation</a> $ D = \mu k_B T $ where $\mu $ is the ratio of the 
    particle's terminal drift velocity to an applied force. 
    We can see from the above equation that $\gamma$ is inverse of mobility $\mu$ 
    hence $ D = \frac{k_B T}{\gamma} $.
    $D$ is also parametrized as $D=\frac{\sigma^2}{2}=\frac{\sigma^2}{2\gamma^2}$ 
    (see <a href="https://en.wikipedia.org/wiki Fokker–Planck_equation">Fokker Planck equation</a>)
    Combining these 2 relationships, 
    we get the well known result from the <b>fluctuation-dissipation theorem</b> $\sigma = \sqrt{2\gamma k_B T}$
</p>

$$
  \frac{dx}{dt} = -\beta D \frac{dU}{dx} + \sqrt{2D} dW_t
$$

<p align='justify'>
    More generally the Itô process describes the flow of a particle $dx(t)$ 
    through a <b>drift term $g(x(t), t)$</b> and a 
    <b>diffusion coefficient $h(x(t), t)$</b>.
    The diffusion coefficient modulates a continuous time stochastic process $W(t)$ 
    called the <b>Wiener process</b> (standard Brownian Motion).  
</p>

$$
  dx_t = g(x_t, t)dt + h(x_t, t)dW_t
$$

<p align='justify'>
    Gaussian increments : For $ 0<=s<t$ $ w(t) - w(s) = N(0, (t-s)I_{D})$ 
    The increments are independent of the past.
    The change in the distribution/density of the particles is defined by the Fokker Planck equation :
</p>

$$
  \frac{\partial p(x_t, t)}{\partial t} = - \nabla_x .(p(x_t, t)g(x_t, t)) + \frac{1}{2} \nabla_x^2 p(x_t, t)h^2(x_t, t)
$$

<p align='justify'>
    From the overdamped langevin equation, substituting $\frac{h^2(x_t, t)}{2} = D$ and $g(x_t, t)=-\beta D \nabla U$ we get 
</p>

$$
    \begin{align}
        \frac{\partial p(x_t, t)}{\partial t} &= \nabla_x . (p(x_t, t) \beta D \nabla_x U) + \nabla_x^2 p(x_t, t)D \\
        \frac{\partial p(x_t, t)}{\partial t} &= \nabla_x . (p(x_t, t) \beta D \nabla_x U)+ \nabla_x . \nabla_x p(x_t, t)D \\
        \frac{\partial p(x_t, t)}{\partial t} &= \nabla_x . (p(x_t, t) \beta D \nabla_x U)+ \nabla_x . (D \nabla_x p(x_t, t) + \cancelto{0}{p(x_t, t) \nabla_x D} ) \\
        \frac{\partial p(x_t, t)}{\partial t} &= \nabla_x . (p(x_t, t) \beta D \nabla_x U)+ \nabla_x . D \nabla_x p(x_t, t) \\
        \frac{\partial p(x_t, t)}{\partial t} &= \nabla_x . D(\beta \nabla_x U + \nabla_x)p(x_t, t)
    \end{align}
$$

<p align='justify'>
    The above equation is called the <b>Smolchowski equation</b>. 
    Defining a flow produces a unique density evolution. 
    However, describing a density evolution does not prescribe a unique flow.
    We will come back to this idea in the <b>Flow Matching</b> section.
</p>

>
**Theoretical probability distribution for the Weiner process**
>
$$
  g(x_t, t) = 0 ; h(t) = 1
$$
>
$$
  \frac{\partial p(x_t, t)}{\partial t} = \frac{1}{2} \frac{\partial^2 p(x_t, t)}{\partial x^2}
$$
>
Using initial conditions $p(x, 0) = \delta(x)$ and applying separation of variables we get the following solution :
>
$$
  p(x_t, t) = \frac{1}{\sqrt{2\pi t}} e^{\frac{-x^2}{2t}}
$$
>

<p align='justify'>
    Lets simulate the weiner process and 
    compare histogram of particle positions to
    the theoretical distribution obtained from the
    Fokker Planck equation.
</p>

```python
# Examples of particle ODEs
class ParticleODE():
  def __init__(self, N, T, dt, seed, process):
    self.seed = seed
    self.T = T
    self.N = N
    self.dt = dt
    self.process = process

  def weiner(self, x):
    rng = np.random.default_rng(seed=self.seed)
    dW = rng.normal(0, np.sqrt(self.dt), size=self.N)
    dx = dW
    return dx

  def overdamped_langevin(self, x, t, beta=0.2, D=0.2, c=2):
    # force = lambda x : 0.2*(x + 0.1) - 0.02*(x - 0.1)**3
    dUdx = lambda x : c
    dxdt = -beta*D*dUdx(x) + np.sqrt(2*D)*self.weiner(x)
    return dxdt

  def potential(self, x):
    # Double Well potential
    # return -0.1*(x + 0.1)**2 + 0.005*(x - 0.1)**4
    # Linear potential
    return 2*x

  def step(self, x):
    if self.process == 'weiner':
      return self.weiner(x)
    elif self.process == 'overdamped_langevin':
      return self.overdamped_langevin(x)
    else:
      raise Exception("Process not Defined !")
```

```python
T = 3
N = 500
x = np.zeros((N))
dt = 0.1
t = np.arange(0, T, dt)
x_hist = np.zeros((len(t), N))

for i, j in enumerate(t):
  particle_ode = ParticleODE(N, T, dt, seed, process='weiner')
  x += particle_ode.step(x)
  x_hist[i] = x

total_frames = T
samples_per_frame = N

fig, ax = plt.subplots(figsize=(6, 4))
rlim = -5
llim = 5
bin_width = 0.2
bins = np.arange(rlim, llim, bin_width)

# Define the animation function (called sequentially for each frame)
def update(frame):
  ax.clear()

  # Set static plot limits and labels
  ax.set_xlim(rlim, llim)
  ax.set_ylim(0, 1)
  ax.set_title("Weiner Process")
  ax.set_xlabel("Value")
  ax.set_ylabel("Density")
  ax.grid(True, linestyle="--", alpha=0.6)

  current_data_size = samples_per_frame
  current_data = x_hist[frame, :]

  ax.hist(
      current_data,
      bins=bins,
      density=True,
      color="royalblue",
      edgecolor="black",
      alpha=0.7
  )

  x_range = np.arange(-5, 5, 0.1)
  p = lambda x, t: (1/np.sqrt(4*np.pi*(t+0.0000001)))*np.exp((-x**2)/(4*(t+0.0000001)))

  # For the wiener process
  ax.plot(x_range, p(x_range, t[frame]), c='orange')

  ax.text(
          -3.8,
          0.45,
          f"Total Samples: {current_data_size}",
          fontsize=12,
          bbox=dict(facecolor="white", alpha=0.8),
      )

# Create the animation object
# frames=100 means the update function will be called 100 times
# interval=20 means 20 milliseconds between frames in the live preview
# Omit init_func and set 'blit=False' because ax.clear() redraws everything.
ani = animation.FuncAnimation(
    fig, update, frames=len(t), blit=False, interval=20
)

# Save the animation as a GIF using the Pillow writer
# fps=30 means 30 frames per second in the final GIF file
output_filename = "weiner_process.gif"
ani.save(output_filename, writer="pillow", fps=5)

print(f"Animation successfully saved as {output_filename}")
plt.show()
```
<div style="text-align: center;">
    <img src="/assets/blogs/normalizing_flows/weiner_process.gif" alt="Weiner Process" width="400" height="250">
    <figcaption> Figure 4. Weiner Process </figcaption>
</div>

<p align='justify'>
    Using the above code we can also simulate 
    the overdamped langevin process. 
    In this case the theoretical distribution
    computed from the Smolchowski equation is 
    (Ref : <a href="https://en.wikipedia.org/wiki/Fokker–Planck_equation">Wikipedia</a>)
</p>

$$
    P(x, t | x_0, t_0) = \frac{1}{\sqrt{4\pi D(t- t_0)}} \exp \left[ \frac{-1}{2}\frac{(x - x_0 + D\beta c(t - t_0))^2}{4D(t-t_0)} \right]
$$

<p align='justify'>
    To create the below simulation I used 
    $D=1, \beta=0.2, c=2$
</p>

<div style="text-align: center;">
    <img src="/assets/blogs/normalizing_flows/overdamped_langevin.gif" alt="Overdamped Langevin Process" width="400" height="250">
    <figcaption> Figure 5. Overdamped Langevin </figcaption>
</div>

<h5 id="Sec4"><b> Flow Matching </b></h5>

Will be updated soon !

<script src="https://giscus.app/client.js"
        data-repo="T-NIKHIL/T-NIKHIL.github.io"
        data-repo-id="R_kgDOMMGitw"
        data-category="Announcements"
        data-category-id="DIC_kwDOMMGit84C7ixL"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="light_high_contrast"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>

<h4><b> References </b></h4>

<p align='justify' id="1">[1] Behrmann, J.; Grathwohl, W.; Chen, R. T. Q.; Duvenaud, D.; Jacobsen, J.-H. Invertible Residual Networks. <em>arXiv</em> <b>2019</b> <a href="https://doi.org/10.48550/arXiv.1811.00995">https://doi.org/10.48550/arXiv.1811.00995</a>.</p>

<p align='justify' id="2">[2] Chen, R. T. Q.; Behrmann, J.; Duvenaud, D.; Jacobsen, J.-H. Residual Flows for Invertible Generative Modeling. <em>arXiv</em> <b>2020</b> <a href="https://doi.org/10.48550/arXiv.1906.02735">https://doi.org/10.48550/arXiv.1906.02735</a>.</p>

<p align='justify' id="3">[3] Chen, R. T. Q.; Rubanova, Y.; Bettencourt, J.; Duvenaud, D. Neural Ordinary Differential Equations. <em>arXiv</em> <b>2019</b> <a href="https://doi.org/10.48550/arXiv.1806.07366">https://doi.org/10.48550/arXiv.1806.07366</a>.</p>
