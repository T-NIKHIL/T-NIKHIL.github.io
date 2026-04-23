---
layout: page
permalink: /msa/
show_excerpts: true
---

<p align='justify'>
    Multiple Sequence Alignment (MSA) is a process of aligning
    2 or more protein, RNA or DNA sequences to maximize regions
    of sequence similarity. Gaps are used to align sequences with
    different lengths. Too many gaps can cause the alignment to become
    meaningless and so a gap penalty is used during the alignment process
    to ensure judicious usage of gaps. The final alignment score is computed
    by using substitution matrices. In the case of proteins this is 
    done using BLOSUM (BLOcks SUbstitution Matrix) as shown in Figure 1. 
    In practice, different substitution matrices are tailored to 
    detecting similarities among sequences that are diverged by varying degrees.
    The BLOSUM-62 matrix, shown in Figure 1, is one of the best for
    detecting weak protein similarities. 
</p>

<figure align="center">
    <div style="text-align: center;">
        <img src="/assets/blogs/msa/BLOSUM62.png" alt="BLOSUM-62" width="300" height="250">
    </div>
    <figcaption>Figure 1. BLOSUM-62 substitution matrix. Figure
    by <a href="//commons.wikimedia.org/wiki/User:Ppgardne" title="User:Ppgardne">Ppgardne</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=119457674">Link</a>
    </figcaption>
</figure>

<figure align="center">
    <img src="/assets/blogs/msa/hmm_parametrization.jpg" 
    alt="Parametrizing an HMM">
    <figcaption>Figure 2. Parametrizing a Hidden Markov Model (HMM)</figcaption>
</figure>

<figure align="center">
    <img src="/assets/blogs/msa/search_using_parametrized_hmm.jpg" 
    alt="Search using parametrized HMM">
    <figcaption>Figure 3. Searching a genetics database using a parametrized HMM.</figcaption>
</figure>