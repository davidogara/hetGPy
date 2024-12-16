---
title: '`hetGPy`: Heteroskedastic Gaussian Process Modeling in Python'
tags:
  - Python
  - Gaussian Processes
  - computer experiments
  - Bayesian Optimization
authors:
  - name: David O'Gara
    orcid: 0000-0002-1957-400X
    affiliation: 1 
    corresponding: true
  - name: Mickaël Binois
    orcid: 0000-0002-7225-1680
    affiliation: 2
  - name: Roman Garnett
    orcid: 0000-0002-0152-5453
    affiliation: [1, 3]
  - name: Ross A Hammond
    orcid: 0009-0005-1046-9296
    affiliation: [1,4,5,6]
affiliations:
 - name: Division of Computationl and Data Sciences, Washington University in St. Louis, USA
   index: 1
 - name: Acumes team, Université Côte d'Azur, Inria, Sophia Antipolis, France
   index: 2
 - name: Department of Computer Science, McKelvey School of Engineering, Washington University in St. Louis, USA
   index: 3
 - name: School of Public Health, Washington University in St. Louis, USA
   index: 4
 - name: Center on Social Dynamics and Policy, Brookings Institution, Washington DC, USA
   index: 5
 - name: Santa Fe Institute, Santa Fe, USA
   index: 6
date: 18 October 2024
bibliography: paper.bib
---

# Summary
Computer experiments are ubiquitous in the physical and social sciences. When experiments are time-consuming to run, emulator (or surrogate) models are often used to map the input–output response surface of the simulator, treating it as a black box. The workhorse function of emulator models is Gaussian process regression (GPR). GPs provide flexible, non-linear regression targets with good interpolation properties and uncertainty quantification. However, it is well-known that naïve GPR scales cubically with input size, and specifically involves intensive computation of matrix determinants and solving linear systems when fitting hyperparameters [@garnett_bayesian_2023;@gramacy_surrogates_2020]. Further, naïve GPR with noisy observations typically assumes an independent, identically-distributed noise process, but many data-generating mechanisms, especially those found in stochastic computer simulation, exhibit non-equal variance noise (also known as heteroskedasticity) [@baker_analyzing_2020]. The software package `hetGP` [@binois_hetgp_2021] alleviates both of these concerns when the dataset of interest contains multiple observations per location (i.e. a dataset of size $N$ has $n<<N$ unique input locations). The `hetGP` framework can calculate matrix inverses and determinants (and therefore the log likelihood and its gradients) on the order $O(n^3)$ rather than $O(N^3)$ time in the naïve case [@binois_practical_2018]. This is accomplished via multiple applications of the [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) [@harville_matrix_1997;@binois_practical_2018] and leveraging the replication structure. Further, `hetGP` also models the mean and noise processes as two coupled GPs, allowing smooth noise dynamics over parameter space [@binois_practical_2018;@gramacy_surrogates_2020;@gramacy_surrogates_2020]. The package has been used in a variety of contexts, such as statistics [@binois_practical_2018;@binois_replication_2019], biology [@lazaridis_genetic_2022] and computational epidemiology [@shattock_impact_2022]. We present a Python reimplementation `hetGPy`, developed in part due to Python’s widespread use in computer simulation [@downey_modeling_2023;@kinser_modeling_2022]. 

# Statement of Need

Python is a popular language for software development, data science, and computer experimentation. Its object orientated framework, high-level functionality, and third-party libraries such as `numpy` [@harris_array_2020], `scipy` [@virtanen_scipy_2020], `PyTorch` [@paszke_pytorch_2019] and `scikit-learn` [@pedregosa_scikit-learn_2011] make it a powerful tool for academic and industry professionals alike. Python is especially popular for computer simulation, with one particular example being the widespread use of Python models of COVID-19 spread [@aylett-bullock_j_2021;@kerr_covasim_2021;@ogara_traceomicron_2023]. `hetGPy` is well-posed for sequential design of Python models, mirroring the functionality of hetGP without having to rely on intermediate libraries such as `reticulate` [@ushey_reticulate_2023] or `rpy2`[@Gautier:2008], or in a more laborious case, converting a Python simulation to R. 

The state of the art for GPR in Python is `GPyTorch` [@gardner_gpytorch_2018], facilitated by black-box matrix–matrix multiplication (BBMM) which is extremely computationally efficient on GPUs. Other GPR routines for Python exist as well and can be found in libraries such as `PyMC` [@abril-pla_pymc_2023], `GPflow` [@matthews_gpflow_2017], and `GPJax` [@pinder_gpjax_2022]. However, while it is possible to jointly model the mean and variance as coupled GPs with these libraries, they do not take advantage of replication in datasets, meaning that under large degrees of replication and input-dependent noise, as is common in stochastic computer experiments [@baker_analyzing_2020], `hetGPy` will be more computationally efficient. \autoref{fig:hetgpy-example} (a) shows a heteroskedastic GP fit to a simulated motorcycle accident dataset [@silverman_aspects_1985]. While it is possible to model input-dependent noise in `GPyTorch` or `BoTorch` [@balandat_botorch_2020], specifying a smooth noise process in the method of [@binois_practical_2018] would require a custom implementation. \autoref{fig:hetgpy-example} (b) illustrates the results of a simple one-dimensional example where we compare the model fits using both `hetGPy` and `GPyTorch`. While both models result in similar predictions, `hetGPy` is far faster, performing exact inference in less than 1 second, while GPyTorch takes over 10 seconds.

![The main features of `hetGPy`. Panel (a) shows a heteroskedastic fit to the motorcycle data [@silverman_aspects_1985]. Panel (b) shows that `hetGPy` yields faster training than (naïve) `GPyTorch` with similar performance under high replication. Bolded and dashed lines indicate the predictive mean and 90\% predictive intervals for homoskedastic GPR, respectively. Data were sampled from the f1d function $f(x)=(6x-2)^2\sin{(12x-4)}$ [@forrester_engineering_2008] in `hetGP` with between 1 and 20 replicates at each design location. Model training times in seconds are next to legend labels.\label{fig:hetgpy-example}](analysis/hetGPy-Fig1.svg)
 

hetGPy also has two intermediate goals: (1) efficient computations, accomplished via implementation on numpy arrays and (2) minimal dependencies, the core of which are `numpy` for efficient array-based computation and `scipy` which contains the definitive implementation of the L-BFGS-B algorithm in Fortran [@morales_remark_2011;@byrd_limited_1995] used for maximum likelihood estimation of hyperparameters. Our experiments indicate `hetGPy` is able to learn response surfaces efficiently, and in the case of high replication, do so on CPUs more efficiently than the default implementation in `GPyTorch`, as shown for a suite of test problems in Table 1. The test problems contain 1,000 unique design locations and an average of 5 replications per location, meaning $n=1,000$ and $N \approx 5,000$ which shows why `hetGPy` is far more efficient in this setting.  As a comparator, we also conduct a set of experiments using stochastic kriging (SK) [@ankenman_stochastic_2010], a precursor method to @binois_practical_2018 that also allows for maximum likelihood estimation with replication, and under the case of homoskedasticity, is nearly equivalent to homoscedastic GPR in `hetGP` and `hetGPy` [@gramacy_surrogates_2020]. We implement SK with a custom `GPyTorch` likelihood that accounts for replication under homoskedasticity. Specifically, given a dataset X with unique designs $(X_1,...,X_k)$ each replicated $(n_1,...,n_k)$ times, we pre-average the outputs $Y_i$ at each unique input location, and then estimate the diagonal of the noise matrix as  $(\sigma^2⁄n_i )$. We see that for the SK case, `hetGPy` and `GPyTorch` have training times on a similar order of magnitude. The package `hetGPy` is under active development and is well-posed to engage with the wider Python community for future extension such as arbitrary kernel functions with auto-differentiation methods facilitated by `PyTorch` or `jax` [@jax2018github].



 
![](analysis/table.pdf)

**Table 1:** Comparing training times across a suite of test problems and libraries. All experiments reflect exact GPR with homoscedastic noise. Optimization problems were selected from [@picheny_benchmark_2013] with implementations from [@surjanovic_virtual_nodate]. Experiments consisted of a Latin Hypercube design of 1,000 unique locations, with between 1 and 10 replicates. iid Gaussian noise was added to each resulting dataset. Experiments were conducted on a 2019 MacBook Pro with a 2.6 GHz 6-Core Intel Core i7 processor, 16GB RAM on macOS Sonoma 14.7.1.\label{tab:hetgpy-experiments}

# References
