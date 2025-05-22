---
#
# By default, content added below the "---" mark will appear in the home page
# between the top bar and the list of recent posts.
# To change the home page layout, edit the _layouts/home.html file.
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
#
layout: home
---

<div style="display: flex; align-items: flex-start; margin-bottom: 20px;">
  <div style="flex: 1;">
    <p align='justify'> 
        Welcome! I am currently a 3rd year Ph.D. student in the 
        <a href="https://engineering.jhu.edu/chembe/" style="color: #0055ff">Department of Chemical and Biomolecular Engineering</a>
        at the Johns Hopkins University. I work with 
        <a href="https://chemistry.jhu.edu/directory/rigoberto-hernandez/" style="color: #0055ff">Dr. Rigoberto Hernandez</a>
        to build data-driven models for material 
        property prediction across multiple length scales.
        These models are used in conjunction with 
        black box optimization algorithms to design 
        materials with bespoke properties.
        We are currently interested in building
        these models for Metal Halide Perovskite Solar Cells.
        Learn about this work 
        <a href="https://doi.org/10.1039/D3MH01484C" style="color : #0055ff">here</a>.
    </p>
    <p align="justify">
        Prior to joining in the Ph.D. program, I completed my 
        Masters degree in the same lab where I worked on developing 
        simulation tools to study the dynamics of unfolding
        of large proteins. My interest in studying biomolecules
        grew from my undergraduate research where I worked on 
        building pseudo-kinetic models for Loop-Mediated
        Isothermal Amplication of DNA. 
    </p>
    <p align="justify">
        My goal is to develop tools to aid scientists 
        in extracting insights from their data and 
        accelerate the discovery of new materials.
    </p>
  </div>
  <div style="margin-left: 20px; margin-top: 5px">
    <img src="/assets/headshot.jpeg" alt="Profile Picture" style="float:right; width:150px; margin-right:20px;">
  </div>
</div>

<style>
  .scrollable-div {
    height: 300px;
    overflow-y: scroll;
    overflow: -moz-scrollbars-vertical;
  }

  .scrollable-div::-webkit-scrollbar {
    -webkit-appearance: none;
    width: 5px;
  }

  .scrollable-div::-webkit-scrollbar-thumb {
    border-radius: 5px;
    background-color: rgba(0,0,0,.2);
    -webkit-box-shadow: 0 0 1px rgba(100,100,100,0.2);
  }
</style>

<script>
  function filterNews(year) {
    const newsItems = document.querySelectorAll('#news-list li');
    newsItems.forEach(item => {
      if (item.getAttribute('data-year') === year) {
        item.style.display = 'list-item';
      } else {
        item.style.display = 'none';
      }
    });
  }
</script>

<div style="margin-top: 20px; margin-right: 20px">
  <h2 style="color: #fc4444;">News Flash !</h2>
  <div class="scrollable-div">
  <ol id="news-list">
      <li data-year="2025">
      <p align="justify">
        [May 22] Almost advanced to the final round Merck Innovation Cup 2025 for team Smart Mnaufacturing.
      </p>
    </li>
    <li data-year="2025">
      <p align="justify">
        [Apr 28] Won the Graduate TA award for teaching EN 545.635 Software Carpentry to graduate students in the Chemical and Biomolecular Engineering department at Johns Hopkins University.
      </p>
    </li>
    <li data-year="2024">
      <p align="justify">
        [Sep 11] Judge for JHU AICheE UG and MSE Poster Symposium.
      </p>
    </li>
    <li data-year="2024">
      <p align="justify">
        [July 9] Mentoring Bryan Zhan, a highschooler interning at PNNL for the summer.
        Going through the basics of Python and Machine Learning with him.
      </p>
    </li>
    <li data-year="2024">
      <p align="justify">
        [July 24 - July 26] Reviewer for <a href="https://ml4lms.bio" style="color: #0055ff">ML4LMS Workshop</a>.
        Part of ICML 2024.<br>
        <a href="https://ml4lms.bio/committees/" style="color: #0055ff">Program Committee</a>
      </p>
    </li>
    <li data-year="2024">
      <p align="justify">
        [July 2] Started my internship at Pacific Northwest National Lab ! 
        Working with Dr. Jinhui Tao to develop machine learning models for designing
        crystal growth modifiers to yield calcite crystals with desired morphologies.
      </p>
    </li>
    <li data-year="2024">
      <p align="justify">
        [March 27 - March 29] Team lead in <a href="https://ac-bo-hackathon.github.io" style="color: #0055ff">Bayesian Optimization Hackathon for Chemistry and Materials</a>.
        Worked with Maitreyee Sharma Priyadarshini, Gigi (Yiren) Wang and Jarett Ren to 
        use BO with local GP to find Covalent Organic Frameworks (COF) with the best methane storage capacity.<br>
        <a href="https://ac-bo-hackathon.github.io/project-localGPs_for_COF/" style="color: #0055ff">Project Description</a>
        <a href="https://ac-bo-hackathon.github.io>" style="color: #0055ff">Code link</a> 
      </p>
    </li>
    <li data-year="2024">
      <p align="justify">
        [March 1] Selected for round 2 in <a href="https://www.emdgroup.com/en/research/open-innovation/innovation-cup.html" style="color: #0055ff">Merck Innovation Cup</a>.
      </p>
    </li>
    <li data-year="2024">
      <p align="justify">
        [Feb 9 - Feb 10] Participated in <a href="https://www.greenhacksjhu.com/past-events/spring-2024" style="color: #0055ff">Greenhacks</a> 
        to come up with solutions to reduce the climate and ecological impact of commercial farming. 
      </p>
    </li>
    <li data-year="2023">
      <p align="justify">
        [Sep 27] Presented in AI-X Foundry Symposium.<br>
        <a href="https://x.com/hernandez_lab/status/1707125651319177363?s=61" style="color: #0055ff">Twitter Link</a>
      </p>
    </li>
    <li data-year="2023">
      <p align="justify">
        [July 18] Taught in a Summer workshop on Python and Machine Learning Fundamentals organized by Dr. Pratyush Tiwary from UMD.
        Contributed to tutorials explaining how to use PyTorch for building Machine Learning models. <br>
        <a href="https://scotch.wangyq.net" style="color: #0055ff"> Workshop Link</a>
      </p>
    </li>
    <li data-year="2022">
      <p align="justify">
        [Aug 21 - Aug 25] Presented at American Chemical Society Conference, Chicago, IL, USA, 2022.
        Talk title : "Mutational Assay of an Actophorin Protein using Adaptive Steered Molecular Dynamics"
      </p>
    </li>
  </ol>
  </div>
  <div>
    <a onclick="filterNews('2025')" style="color: #0055ff">2025</a>
    <a onclick="filterNews('2024')" style="color: #0055ff">2024</a>
    <a onclick="filterNews('2023')" style="color: #0055ff">2023</a>
    <a onclick="filterNews('2022')" style="color: #0055ff">2022</a>
  </div>
</div>


