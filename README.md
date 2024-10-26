# tram-dag
Code to reproduce the experiments for the TRAM-DAG paper.

## Comparison with Existing Normalizing Flows (NFs)
The code to reproduce experiments comparing TRAM-DAG with existing normalizing flows (NFs) is in the folder `comparison`.

### Figures Comparing to VACA
To generate figures for L1 (observational) and L2 (interventional) comparisons on the VACA dataset, use the file `comparison/Figure_Triangle_Linear_Bimodal.R`.

### Figures Comparing to CAREFL
Figure 5, titled "Results of the counterfactual queries as posed in CAREFL," can be reproduced using `comparison/carefl_fig5.r`.

## Simulation Study
The code to reproduce the simulation study is in the folder `summerof24`. This study explores TRAM-DAG performance on continuous and ordered categorical data types.

### Figures and Data for the Continuous Case
Figures and data for the continuous case are generated using `triangle_structured_continuous.R`. Set specific command-line arguments to define the function form and model type. Alternatively, modify `args <- c(1, 'ls')` around line 7.

- **Example for Figure 7** (Complex TRAM and Complex DGP): set `args <- c(4, 'cs')`.

### Figures and Data for Ordered Data Types
Figures for the ordered data type experiments are generated with `triangle_structured_mixed.R`. To reproduce Figure 8, use `args <- c(1, 'ls')`.

# System and Package Information for TRAM-DAG

## Key Python Libraries
The experiments were conducted with the following versions:
- **TensorFlow**: 2.13.0
- **TensorFlow Probability**: 0.21.0

## R Environment
- **R version**: 4.2.3 (2023-03-15)

## Session Information
A detailed record of all R packages, versions, and system settings is saved in [session_info.txt](session_info.txt) for full reproducibility.