# Running scSHC method with the counts data.
#                             
#                  INPUT: * data/test_data/TabulaSapiens_tiny.h5ad
#                         * data/test_data/roc_pcuts.txt
#                  OUTPUT: * data/test_data/Tabula-small_scHC-clusts_counts.pkl
#                           -> Cell cluster labels at different p-value cutoffs.

################################################################
			# Environment Setup #
################################################################
data_dir <- 'data/test_data/'
out_dir <- data_dir

# Need to git clone scSHC, and then add the directory below...
# See here for repo: https://github.com/igrabski/sc-SHC
code_dir <- 'path/to/sc-SHC/'

#### Need to add the python conda environment path!
Sys.setenv(RETICULATE_PYTHON = "/opt/miniconda/miniconda3/envs/cytocipher_ms/bin/python")
library(reticulate)
source(paste0(code_dir, 'significance_analysis.R'))
### To install, in addition to the packages listed at the top of the R script,
### also needed to install BiocManager::install("glmGamPoi")

sc <- import("scanpy")

##########################################################################
                         # Loading the data #
##########################################################################
data <- sc$read_h5ad(paste0(data_dir, 'TabulaSapiens_tiny.h5ad'))
clust_col <- 'overclusters'

#expr <- as.matrix( t(data$to_df()) ) #Needs to be genes*cells matrix!
expr <- as.data.frame( t(data$layers[['counts']]) )
colnames(expr) <- data$obs_names$values
rownames(expr) <- data$var_names$values
expr <- as.matrix( expr )
clusters <- as.character( data$obs[[clust_col]] )

# Loading in the p-value cutoffs.. #
p_cuts <- read.table(paste0(data_dir, 'roc_pcuts.txt'))[,1]

##########################################################################
	              	# Now running scHC #
##########################################################################
source(paste0(code_dir, 'significance_analysis.R'))
start_time <- Sys.time()
new_clusters <- testClusters(expr, clusters, var.genes=rownames(expr),
			    cores=15, alpha=0.95)
end_time <- Sys.time()

lapse <- end_time - start_time
print(lapse)

print(length(unique(new_clusters)))

# Time difference of 2.982415 mins
# For 500 cells and 2,180 genes with 15 CPUs. 

#### Saving results
sys <- import('sys')
sys$path <- c(sys$path, '/home/s4392586/myPython/BeautitfulCells/')
spl <- import('beautifulcells.preprocessing.load_data.simple_pickle')

clusts <- dict('new_clusts'=r_to_py(new_clusters))

spl$saveAsPickle(paste0(data_dir, 'Tabula-small_scHC-clusts_counts.pkl'), clusts)

