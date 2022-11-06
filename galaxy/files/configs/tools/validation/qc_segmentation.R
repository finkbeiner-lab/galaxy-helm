# QC for too many segmented objects.

# Include packages without triggering stderr
# options(warn=-1)
library(proto)
library(argparse)

# Set up argument handling
parser <- ArgumentParser()
parser$add_argument("input_file", nargs=1, help="OverlaysTablesResults/cell_data.csv")
parser$add_argument("max_objects", nargs=1, type="integer", help="Maximum number of cells expected.")
parser$add_argument("output_file", nargs=1, help="Well and timepoint summary.")

args <- parser$parse_args()
ifile <- args$input_file
max_cells <- args$max_objects
ofile <- args$output_file

flag_well_time <- function(max_poss_cells, dframe){
    # Takes data frame and expected number of cells.
    # Returns wells with too many objects (poor segementation parameters).
    too_many <- as.numeric(dframe$ObjectCount) > as.numeric(max_poss_cells)
    clean_df <- dframe[too_many, c("Sci_WellID", "Timepoint", "ObjectCount")]
    return(unique(clean_df))
    }

cell_data_summary <- data.frame(read.table(ifile, header=TRUE, sep = ','))
wells_times_with_too_many <- flag_well_time(max_cells, cell_data_summary)

write.table(wells_times_with_too_many, sep="\t", file=ofile, row.names=FALSE, col.names=TRUE)

paste("Wells and timepoints with more than:", max_cells, "cells", sep=" ")

if (nrow(wells_times_with_too_many) < 1){ 
    paste("No entries with more than:", max_cells, "cells", sep=" ")
    } else
      	print(wells_times_with_too_many)



