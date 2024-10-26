# tram-dag
Code to reproduce the experiments for the TRAM-DAG paper.

## Comparison with Existing Normalizing Flows (NFs)
The code to reproduce the experiments comparing TRAM-DAG with existing normalizing flows (NFs) is in the folder `comparison`.

### Figures Comparing to VACA
The code to reproduce figures for the L1 (observational) and L2 (interventional) comparisons on the VACA dataset is in the file `comparison/Figure_Triangle_Linear_Bimodal.R`.

### Figures Comparing to CAREFL
The code to reproduce Figure 5, titled "Results of the counterfactual queries as posed in CAREFL," is in the file `comparison/carefl_fig5.r`.

## Simulation Study
The code to reproduce the simulation study is in the folder `summerof24`. This study explores TRAM-DAG performance on both continuous and ordered categorical data types.

### Figures and Data for the Continuous Case
These figures and data have been generated using the file `triangle_structured_continuous.R`. To reproduce the results, you can run the file with specific command-line arguments. The first argument specifies different functional forms of `f(x2)`, while the second argument specifies the model type. Alternatively, you can modify the line `args <- c(1, 'ls')` around line 7.

For example:
- To reproduce Figure 7 (Complex TRAM and Complex DGP), run `triangle_structured_continuous.R` with `args <- c(4, 'cs')`, or set these values directly in the code.

### Figures and Data for the Ordered Data Types
The figures and data for the ordered data type experiments are generated using the file `triangle_structured_mixed.R`. To reproduce Figure 8 in the paper, set `args <- c(1, 'ls')` in the code.