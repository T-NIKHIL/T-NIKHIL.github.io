---
layout: page
permalink: /alphafold3/
show_excerpts: true
---

<p align='justify'>
    <b><span style="color:red">DISCLAIMER</span> : BELOW CONTENT IS NOT AI GENERATED AND REPRESENTS THE AUTHORS VIEWS AND INTERPRETATIONS. FOR QUERIES USE THE COMMENT BOX BELOW.</b>
</p> 

<p align='justify'>
    The <a href="https://en.wikipedia.org/wiki/Protein_folding">protein folding</a> 
    problem refers to the challenge : <em>Can we predict a protein's 
    structure completely from its sequence?</em>. 
    While this is possible using X-ray crystallography and 
    Nuclear Magnetic Resonance (NMR), 
    these techniques face challenges such as:
</p>
<ol>
    <li> low success rates in getting high quality protein crystals 
        (for ex: intrinsically disordered proteins fall in this category. 
        More details on how to grow crystals
        <a href="https://www.iucr.org/news/newsletter/volume-32/number-3/how-to-grow-crystals-for-x-ray-crystallography">here</a>),</li>
    <li> limitations on the protein sizes that can be crystallized 
        (for ex: while individual proteins in the 
        <a href="https://en.wikipedia.org/wiki/Nuclear_pore_complex">
        Nuclear Pore Complex (NPC)</a> can be crystallized and examined, 
        the overall assembly remains elusive
        to study via experimental techniques),</li>
    <li> tedious process with low throughput rates.</li>
</ol>
<p align='justify'>
    An alternative approach would be to simulate the protein folding 
    process using computers. However, our current compute capabilities allow
    access to simulation times on the scale of 10s of nanoseconds,
    enabling folding simulations of only small proteins.<a href="#1">[1]</a>
    Google DeepMind's first iteration at this problem, called AlphaFold1,
    uses Convolutional Neural Networks (CNNs) to predict torsion and
    distance distributions (called distograms) from Multiple Sequence Alignment (MSA) features. 
    Potentials were constructed based on these distributions and 
    the initial structure from the predicted torsion and distance distributions
    was minimized using gradient descent.<a href="#2">[2]</a> Using this strategy the
    DeepMind team could predict accurate protein backbone structures.<a href="#2">[2]</a>
</p>

<p align='justify'>
    The following year the DeepMind team unveiled a more powerful model,
    based on the transformer architecture, 
    that could now predict accurate protein side conformations as well. 
    This model, called AlphaFold2, could predict structures with a median
    backbone accuracy of 0.96 Å r.m.s.d.<sub>95</sub> 
    (Cα root-mean-square deviation at 95% residue coverage) on the CASP14 test domains.
    In addition, most predicted structures had 
    <a href="https://en.wikipedia.org/wiki/Template_modeling_score">
    Template Modelling (TM) scores</a>
    greater than 0.9.<a href="#3">[3]</a>
    AlphaFold2 marks the start of a series of computational models that 
    predict protein structures on the scale of experimental accuracy.
</p>

<figure align="center">
    <img src="/assets/blogs/alphafold3/alphafold1_and_alphafold2_architectures.jpg" alt="AF1 and AF2 architectures">
    <figcaption>Figure 1. AlphaFold1 and AlphaFold2 model diagrams</figcaption>
</figure>

<p align='justify'>
    AlphaFold3 (AF3) was released by the DeepMind team in 2023 and
    is their latest open source model as of date that goes beyond just predicting
    protein structures and includes RNA, DNA, docking of ligands and ions,
    bio-molecular complexes and multimers.<a href="#4">[4]</a> 
    In this technical blog I will go over some of the details of the model
    architecture, results on benchmarks and some of the limitations of this 
    model.
</p>

<h4><b> The journey from token to structure </b></h4>

<p align='justify'>
    Below figure is a snapshot of the screen the user
    encounters when using the 
    <a href="https://alphafoldserver.com">AlphaFold server</a>.
    It provides a text box for the user to paste in their
    protein, DNA or RNA sequence and a dropdown menu 
    to choose from the different types of ions and ligands
    AF3 supports.
</p>

<figure align="center">
    <img src="/assets/blogs/alphafold3/AF3_input_sec1.jpg" alt="AF3 input screen">
    <figcaption>Figure 2. Overview of AF3 input screen on the AlphaFold Server. Selection of entity type.</figcaption>
</figure>

<p align='justify'>
    AF3 also provides supports for structure prediction of multimeric chains and
    for different post translation modifications of amino acid residues and
    DNA/RNA nucleotides.
</p>

<figure align="center">
    <img src="/assets/blogs/alphafold3/AF3_input_sec2.jpg" alt="AF3 input screen">
    <figcaption>Figure 3. Overview of AF3 input screen on the AlphaFold Server. 
    Selection for post translation modifications.</figcaption>
</figure>

<p align='justify'>
    Once the user submits the job. AF3 runs the user input data through
    a data processing pipeline, generates features to feed the model and
    then predicts the 3D atomic coordinates and per atom 
    confidence (pLDDT) and alignment metrics.
</p>

<figure align="center">
    <img src="/assets/blogs/alphafold3/AF3_input_sec3.jpg" alt="AF3 input screen">
    <figcaption>Figure 4. Overview of AF3 input screen on the AlphaFold Server.
    Structure Prediction and confidence metrics</figcaption>
</figure>

<p align='justify'>
    When the user submits the job on the AlphaFold server, 
    AF3 runs a data processing pipeline where :
</p>
<ol>
    <li> input sequences are tokenized and features extracted from the tokenized sequences <a href="#Sec1.1">(Section 1.1)</a>,</li>
    <li> reference conformers generated for each residue/nucleotide <a href="#Sec1.2">(Section 1.2)</a>,</li>
    <li> MSA run on the input sequence and the search results are featurized <a href="#Sec1.3">(Section 1.3)</a>,</li>
    <li> template searches for single entities based on the retrieved MSA results <a href="#Sec1.4">(Section 1.4)</a>.</li>
</ol>

<h5 id="Sec1.1"><b> 1.1. Tokenization of input sequences </b></h5>

<p align='justify'>
    The amino acids, nucleotides, ligands and ions
    are represented using numerical representation called tokens. 
    Each standard amino acid and nucleotide are represented using
    single tokens while modified amino acids, ligands and ions
    are tokenized per-atom. For example, Serine which is a standard
    amino acid is represented by 1 token while ibuprofen which contains
    15 heavy atoms is represented using 15 tokens. 
    There are in total 32 classes of molecules : 
    20 standard amino acids + 1 unknown, 
    4 standard DNA nucleotides + 1 unknown,
    4 standard RNA nucleotides + 1 unknown,
    gap (from the MSA),
    ligands and ions are treated as unknown.
    Two examples are shown below, 
    where in one case a protein chain
    is comprised of standard amino acids and 
    another case consisting of multiple chains.
    The token features overall attempt to distinguish
    between the different amino acids and nucleotides in a chain
    from those present in different chains, as in the case of multimers.
    The token_bonds feature is a 2D matrix which indicates
    whether a bond exists between token <em>i</em> and <em>j</em> and
    is restricted to just inter ligand bonds and bonds between
    ligand and polymer which are less than 2.4 Å.
</p>

<figure align="center">
    <img src="/assets/blogs/alphafold3/token_features.jpg" alt="token features">
    <figcaption>Figure 5. Features constructed from tokenized sequences.</figcaption>
</figure>

<h5 id="Sec1.2"><b> 1.2. Generation of reference conformers (Training only) </b></h5>

<p align='justify'>
    Reference conformers for each monomer in the chains are created using 
    RDKit's ETKDG3 confomer generation algorithm. Data from the mmCIF file
    is used to create a set of features shown in Figure 6. 
    Conformer generation done only during training. 
    At inference time, a dummy CIF with all atom coordinates zeroed is used.
</p>

<figure align="center">
    <img src="/assets/blogs/alphafold3/conformer_features.jpg" alt="conformer features">
    <figcaption>Figure 6. Features constructed from generated conformers</figcaption>
</figure>

<h5 id="Sec1.3"><b> 1.3. Multiple Sequence Alignment (MSA) searches </b></h5>

<p align='justify'>
    The process of aligning 2 or more protein, DNA or RNA sequences
    to maximize regions of sequence similarity is called Multiple 
    Sequence Alignment (MSA). More details on MSA can be found in this
    <a href="/msa/">blogpost</a>. MSA is useful in structure prediction
    as correlated mutations are evolutionary signals
    that AF3 can use to infer whether a pair of residues are in close
    proximity with each other. AF3 uses Hidden Markov Models (HMMs) to build the
    MSA for the query sequence because traditional sequence alignment
    algorithms do not provide site specific substitution probabilities. 
    Unlike traditional HMMs which take the form of cyclic graphs,
    HMMs from MSA, also called profiles, have a directional information flow
    from left to right. An example HMM is shown in Figure 7 where starting
    from the left end of the sequence, the arrows indicate the most probable 
    state to enter next. The states are indicated as M, D and I which represent
    an amino acid, deletion or insertion state respectively.
</p>

<figure align="center">
    <img src="/assets/blogs/alphafold3/MSA_features.jpg" alt="MSA features">
    <figcaption>Figure 7. Features constructed from MSA search</figcaption>
</figure>

<h5 id="Sec1.4"><b> 1.4. Creating structural priors by searching for templates </b></h5>

<p align='justify'>
    Using the constructed MSA profile in the previous step, 
    AF3 next runs a search across genetic databases to find
    structural priors for the input sequence. This is done
    only for individual chains so the model does not know
    how different chains are in proximity with each other.
    AF3 uses upto 4 templates during training and inference.
    The template features can be divided based on sequence
    and structure. While AF1 predicts distograms, AF3 uses
    distograms for the template as an input.
</p>

<figure align="center">
    <img src="/assets/blogs/alphafold3/template_features.jpg" alt="template features">
    <figcaption>Figure 8. Features constructed from template search</figcaption>
</figure>

<h5 id="Sec2"><b> 2. High Level Overview of AF3 Architecture </b></h5>

<p align='justify'>
    Okay so far we have collected the data and
    featurized it. Now we are ready to pass this data
    through the AF3 model and predict 3D structures and
    confidence metrics. AF3 primarily operates on two
    internal representations called the pair <em>p</em> and single
    representations.
    These representations operate on a fine-grained and
    coarse-grained scale. 
    On the fine-grained scale they encode relationships
    between 
    
     atomic level
     These are continuously updated as 
    they are passed through multiple module layers until
    they reach the Diffusion Module where starting from
    gaussian distributed 3D atomic coordinates, 
    the module iteratively denoises the coordinates
    conditioned based on the refined single and pair
    representations.
</p>

<script src="https://giscus.app/client.js"
        data-repo="T-NIKHIL/T-NIKHIL.github.io"
        data-repo-id="R_kgDOMMGitw"
        data-category="Q&A"
        data-category-id="DIC_kwDOMMGit84C7ixN"
        data-mapping="pathname"
        data-strict="1"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="light_high_contrast"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>

<h4><b> References </b></h4>

<p align='justify' id="1">[1] Scheraga, H. A.; Khalili, M.; Liwo, A. Protein-Folding Dynamics: Overview of Molecular Simulation Techniques. <em>Annu. Rev. Phys. Chem.</em> <b>2007</b>, 58 (1), 57–83.<a href="https://doi.org/10.1146/annurev.physchem.58.032806.104614">https://doi.org/10.1146/annurev.physchem.58.032806.104614</a>.</p>

<p align='justify' id="2">[2] Senior, A. W.; Evans, R.; Jumper, J.; Kirkpatrick, J.; Sifre, L.; Green, T.; Qin, C.; Žídek, A.; Nelson, A. W. R.; Bridgland, A.; Penedones, H.; Petersen, S.; Simonyan, K.; Crossan, S.; Kohli, P.; Jones, D. T.; Silver, D.; Kavukcuoglu, K.; Hassabis, D. Improved Protein Structure Prediction Using Potentials from Deep Learning. <em>Nature</em> <b>2020</b>, 577 (7792), 706–710.<a href="https://doi.org/10.1038/s41586-019-1923-7">https://doi.org/10.1038/s41586-019-1923-7</a>.</p>

<p align='justify' id="3">[3] Jumper, J.; Evans, R.; Pritzel, A.; Green, T.; Figurnov, M.; Ronneberger, O.; Tunyasuvunakool, K.; Bates, R.; Žídek, A.; Potapenko, A.; Bridgland, A.; Meyer, C.; Kohl, S. A. A.; Ballard, A. J.; Cowie, A.; Romera-Paredes, B.; Nikolov, S.; Jain, R.; Adler, J.; Back, T.; Petersen, S.; Reiman, D.; Clancy, E.; Zielinski, M.; Steinegger, M.; Pacholska, M.; Berghammer, T.; Bodenstein, S.; Silver, D.; Vinyals, O.; Senior, A. W.; Kavukcuoglu, K.; Kohli, P.; Hassabis, D. Highly Accurate Protein Structure Prediction with AlphaFold. <em>Nature</em> <b>2021</b>, 596 (7873), 583–589. <a href="https://doi.org/10.1038/s41586-021-03819-2">https://doi.org/10.1038/s41586-021-03819-2</a>.</p>

<p align='justify' id="4">[4] Abramson, J.; Adler, J.; Dunger, J.; Evans, R.; Green, T.; Pritzel, A.; Ronneberger, O.; Willmore, L.; Ballard, A. J.; Bambrick, J.; Bodenstein, S. W.; Evans, D. A.; Hung, C.-C.; O’Neill, M.; Reiman, D.; Tunyasuvunakool, K.; Wu, Z.; Žemgulytė, A.; Arvaniti, E.; Beattie, C.; Bertolli, O.; Bridgland, A.; Cherepanov, A.; Congreve, M.; Cowen-Rivers, A. I.; Cowie, A.; Figurnov, M.; Fuchs, F. B.; Gladman, H.; Jain, R.; Khan, Y. A.; Low, C. M. R.; Perlin, K.; Potapenko, A.; Savy, P.; Singh, S.; Stecula, A.; Thillaisundaram, A.; Tong, C.; Yakneen, S.; Zhong, E. D.; Zielinski, M.; Žídek, A.; Bapst, V.; Kohli, P.; Jaderberg, M.; Hassabis, D.; Jumper, J. M. Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3. <em>Nature</em> <b>2024</b>, 630 (8016), 493–500. <a href="https://doi.org/10.1038/s41586-024-07487-w">https://doi.org/10.1038/s41586-024-07487-w</a>.</p>
