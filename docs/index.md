# Full Index
Detailed documentation on each respository script.


scripts/X1_pbmc3k/

    X1_pbmc.ipynb -> Prepping the PBMC data.
    
                        INPUT: * NONE.
                        OUTPUT: * data/test_data/pbmc.h5ad
                        
    X2_cytocipher_pbmc.ipynb -> Running the enrichment
                            methods on pbmc single cell data 
                            at a high resolution, showing performance to identify 
                            over/under-clustering instances.
                            
                            INPUT: * data/test_data/pbmc.h5ad
                            OUTPUT: * NONE, plots within
                            
    X3_diagnostic_plots.ipynb -> Runs the comparison of different 
                            aggregation methods to reduce bias of significance
                            as a result of increased number of cells per cluster.
                            
                            INPUT: * data/test_data/pbmc.h5ad
                            OUTPUT: * NONE, plots within.

scripts/X2_simulation/

    X1_gen_test_data.R -> Uses splatter to simulate scRNA-seq.
                            
            NOTE: this took too much memory to run locally, so has was run on
            linux server.
                            
            INPUT: * data/test_data/pbmc.h5ad
            OUTPUT: * data/sim_data/splat_NODE.h5ad
        
     X2_simulate.ipynb -> Runs the simulation of the data.
        
                                    INPUT: * data/sim_data/splat_NODE.h5ad
                                    OUTPUT: * data/sim_data/splat_MyDE.h5ad
                                    
     X3_run_python_enrichment.ipynb -> Runs the enrichment methods on 
                                        the simulation data with additional 
                                        AUROC calculations and also with 
                                        negative gene set filtering for 
                                        all methods.
                                                                
                                 INPUT: * data/sim_data/splat_MyDE.h5ad
                                 OUTPUT: * NONE, plots within.
                                 
scripts/X3_e18_hypo_neurons/

    X1_run_python_enrichment_e18Hypo.ipynb -> Runs the enrichment 
                                                   methods on the E18 hypo data, 
                                                   including ROC curves and
                                                   negative gene sets 
                                                   consideration with each 
                                                   scoring approach.
                                                            
                             INPUT: * data/dev_data/np-da_bfs.h5ad
                             OUTPUT: * NONE, plots within.  

scripts/X4_pancreas/

    This analysis is provided as a Cytocipher tutorial in a separate repo. See:
    
    https://github.com/BradBalderson/Cytocipher/blob/main/tutorials/cytocipher_pancreas.ipynb

scripts/X5_prostate/

    X1_prostate.ipynb -> Cytocipher on prostate cancer, using this dataset:
                      https://www.prostatecellatlas.org/ 
                      
                              INPUT: Auto-downloaded in notebook to:
                                    * data/test_data/
                              OUTPUT: * NONE, plots within

scripts/X6_tabula_sapiens/

    X1_TabulaSapiens.ipynb -> Downloading Tabula Sapiens.
                                 
                                 INPUT: * NONE
                                 OUTPUT: * data/test_data/TabulaSapiens.h5ad
                                         * data/test_data/TabulaSapiens_Cytociphered.h5ad
                                         -> Include info on how this was 
                                             downloaded in notebook.
    
    X2_TabulaSapiens_coexpr.ipynb -> Runs Cytocipher using Coexpr scores on the 
                                     full Tabula Sapiens data.
                     
                                 INPUT: * data/test_data/TabulaSapiens_Cytociphered.h5ad
                                 OUTPUT: 
                                 * data/test_data/coexpr_tabula-full_results.pkl
                                 
    X3_TabulaSapiens_code.ipynb -> Runs Cytocipher using Code scores on the 
                                     full Tabula Sapiens data.
                     
                                 INPUT: * data/test_data/TabulaSapiens_Cytociphered.h5ad
                                 OUTPUT: 
                                 * data/test_data/code_tabula-full_results.pkl
      
    X4_TabulaSapiens_scanpy.ipynb -> Runs Cytocipher using scanpy scores on the 
                                     full Tabula Sapiens data.
                     
                                 INPUT: * data/test_data/TabulaSapiens_Cytociphered.h5ad
                                 OUTPUT: 
                                 * data/test_data/scanpy_tabula-full_results.pkl
                                 
    X5_TabulaSapiens_giotto.ipynb -> Subsets the TabulaSapiens data and performs
                                     Giotto enrichment..
                                     
                                 INPUT: * data/test_data/TabulaSapiens_Cytociphered.h5ad
                                 OUTPUT:  
                                      * data/test_data/TabulaSapiens_subset.h5ad   
                                      * data/test_data/giotto_tabula_results.pkl 
    
    X6_TabulaSapiens_sub_coexpr.ipynb -> Running the Tabula Sapiens with a 
                                         subset of the data with coexpr scoring.
                                 
                                 INPUT: 
                                      * data/test_data/TabulaSapiens_subset.h5ad
                                 OUTPUT: 
                                      * data/test_data/coexpr_tabula_results.pkl
                                 
     X7_TabulaSapiens_sub_code.ipynb -> Running the Tabula Sapiens with a subset 
                                                  of the data with code scoring.
                                 
                                 INPUT: 
                                      * data/test_data/TabulaSapiens_subset.h5ad
                                 OUTPUT: 
                                        * data/test_data/code_tabula_results.pkl
                           
     X8_TabulaSapiens_sub_scanpy.ipynb -> Running the Tabula Sapiens with all 
                                            the data with Scanpy scoring.
                                 
                              INPUT: * data/test_data/TabulaSapiens_subset.h5ad
                              OUTPUT: * data/test_data/scanpy_tabula_results.pkl                        
                                                                     
    X9_TabulaSapeins_TINY.ipynb -> Making a tiny version of the Tabula Sapiens 
                                 data, representing a very tiny subset of the 
                                 cell types so that it's possible to run using 
                                 scSHC.
                                 
                     INPUT: * data/test_data/TabulaSapiens_subset.h5ad
                     OUTPUT: * data/test_data/TabulaSapiens_tiny.h5ad
    
    X10_scSHC_counts.R -> Running scSHC method with the counts data.
                             
                 INPUT: * data/test_data/TabulaSapiens_tiny.h5ad
                        * data/test_data/roc_pcuts.txt
                 OUTPUT: * data/test_data/Tabula-small_scHC-clusts_counts.pkl
                         -> Cell cluster labels at different p-value cutoffs.

    X11_TabulaSapiens_benchmark.ipynb -> Visualising results from benchmarking 
                                    the different methods, on both the full 
                                    tabula sapiens dataset, as well as the 
                                    smaller subset.
                                     
                     INPUT: * data/test_data/TabulaSapiens.h5ad
                            * data/test_data/TabulaSapiens_subset.h5ad
                            * data/test_data/TabulaSapiens_tiny.h5ad
                            * data/test_data/giotto_tabula_results.pkl
                            * data/test_data/code_tabula_results.pkl
                            * data/test_data/code_tabula-full_results.pkl
                            * data/test_data/coexpr_tabula_results.pkl
                            * data/test_data/coexpr_tabula-full_results.pkl
                            * data/test_data/scanpy_tabula_results.pkl
                            * data/test_data/scanpy_tabula-full_results.pkl
                            * data/test_data/Tabula-small_scHC-clusts_counts.pkl
                     OUTPUT: * NONE, plots within.

