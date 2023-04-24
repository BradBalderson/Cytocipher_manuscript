# Uses splatter to simulate scRNA-seq.
#
#        Using splatter for the simulation:
#        https://bioconductor.org/packages/devel/bioc/vignettes/splatter/inst/doc/splatter.html
#
#        INPUT: * data/test_data/pbmc.h5ad
#        OUTPUT: * data/sim_data/splat_NODE_orBATCH.h5ad

####################################################################################
                          # Environment Setup #
####################################################################################
work_dir <- './' #Set to where git clone Cytocipher_manscript
setwd(work_dir)

#Set to the conda environment used!
Sys.setenv(RETICULATE_PYTHON = "~/opt/miniconda3/envs/cytocipher_ms/bin/python/")

library(reticulate)
sc <- import("scanpy")
pd <- import("pandas")

library(stringr)
library(splatter)

data_dir <- 'data/test_data/'
out_dir <- 'data/sim_data/' 

#####################################################################################
                          # Loading the data #
#####################################################################################
data <- sc$read_h5ad(paste0(data_dir, 'pbmc.h5ad'))
labels <- as.character(data$obs[,'leiden'])

#####################################################################################
          # Now simulating the data with similar conditions to the above #
#####################################################################################
counts <- data$layers[['counts']]
counts <- t(counts) # genes * cells

#### Estimating parameters based on this count data
params <- splatEstimate(counts)
# Let's save these parameters so we don't need to fit again, since takes a long
# time to run!!!!
saveRDS(params, paste0(out_dir, 'splat_params.rds'))

#####################################################################################
         # NO Group DE; will add the group DE in python!!! #
#####################################################################################
params <- readRDS( paste0(out_dir, 'splat_params.rds') )

new_params_noDE <- setParams(params, nGenes=10000, batchCells=3000)

sim <- splatSimulate(new_params_noDE, method="groups")

####### Saving to AnnData
counts <- as.data.frame( as.matrix(sim@assays@data$counts) )
obs <- as.data.frame( colData( sim ) )
var <- as.data.frame( rowData( sim ) )

data <- sc$AnnData( r_to_py(t(counts)),
                obs=r_to_py(obs), var=r_to_py(var))

data$write_h5ad(paste0(out_dir, 'splat_NODE.h5ad'), compression='gzip')




