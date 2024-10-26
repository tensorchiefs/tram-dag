# tram-dag
Code to reproduce the experiments for the TRAM-DAG paper 

## Comparison with exiting NFs
The code to reproduce the experiments comparing TRAM-DAG with existing NFs is 
in the folder `comparison`.

### Figures comparing to VACA
The code to reproduce Figures for the L1 and L2 comparison on the VACA dataset is in the file 
`Figure_Triangle_Linear_Bimodal.R`.

### Figures comparing to Carfel
TODO

## Simulation study
The code to reproduce the simulation study is in the folder `summerof24`.

### Figures and data for the continuous case
These figures and data have been generated using the file `triangle_structured_continous.R`. 
To reproduce the results you can run the file with differnt command line arguments.The first is for different functional forms of `f(x2)` 
the second codes the model. Alternatively, you can modify the line `args <- c(1, 'ls') #` at around line 7. 


For Figure 7 (Complex TRAM and Complex DGP ), use `triangle_structured_continous.R` with `args <- c(4, 'cs')` 

### Figures and data for the ordered data types
The figures and data have been generated using the file `triangle_structured_mixed.R`. Use 
'args <- c(1, 'ls')' to reproduce Figure 8 in the paper.
