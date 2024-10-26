# Readme

### Date (last run of the script)

``` r
format(Sys.time(), "%a %b %d %X %Y")
```

    [1] "Wed Sep 06 14:04:51 2023"

Note that the coefficients have been \[0.5, 0.5\]

Codebase: \[https://github.com/tensorchiefs/carefl/commit/\]

Command:

    NOT AVAILIBE. CODE HAS BEEN RUN PROBABLY AROUND AUGUST 9 with
    main.py -c -n 2500

The directory contains several CSV - files:

``` r
#List the files in the current getSrcDirectory
list.files()
```

     [1] "Readme_files"      "Readme.md"         "Readme.qmd"       
     [4] "Readme.rmarkdown"  "X_org.csv"         "X.csv"            
     [7] "xCF_onX1_pred.csv" "xCF_onX1_true.csv" "xCF_onX2_pred.csv"
    [10] "xCF_onX2_true.csv" "xObs.csv"         

-   X_org.csv training data before scaling
-   X scaled data used for training
-   xObs.csv Counterfactual values

## The Training Data

``` r
X = read.csv("X.csv", header = FALSE)
summary_stats(as.data.frame(X))
```

      Column                Mean          Std_Dev            Kurtosis
    1     V1 -0.0159889323138897  1.0329149971236 0.00117986949599631
    2     V2 0.00515871989482681 0.98052145376413 0.00167760145060213
    3     V3  0.0086969530293083 1.00020006002001   0.125452312851095
    4     V4   0.260568509559905 1.00020006002001 0.00578220171587256
              Shapiro_W            Shapiro_p               Min              Max
    1 0.962783977230874   7.179496752163e-25 -6.25045038004951  6.4982787171781
    2 0.955491962466005 5.66156269960336e-27  -6.2383564299485 6.74534589240224
    3 0.307391813690583 1.26462402913678e-70 -19.9930727807444 25.3687902742826
    4 0.895015328276807 2.54202540343256e-38 -3.94614647879877 11.3027034250946

``` r
dim(X)
```

    [1] 2500    4

## Strength of the interventions

    xvals = np.arange(-3, 3, .1) #See counterfactuals()

This is equivalent to

``` r
seq(-3,2.9,0.1)
```

     [1] -3.0 -2.9 -2.8 -2.7 -2.6 -2.5 -2.4 -2.3 -2.2 -2.1 -2.0 -1.9 -1.8 -1.7 -1.6
    [16] -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1
    [31]  0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  1.1  1.2  1.3  1.4
    [46]  1.5  1.6  1.7  1.8  1.9  2.0  2.1  2.2  2.3  2.4  2.5  2.6  2.7  2.8  2.9
