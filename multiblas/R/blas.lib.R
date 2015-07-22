#
#  blas.lib.R
#  multiblas
#
#  Created by MPB on 6/29/15.
#  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
#  License http://opensource.org/licenses/BSD-2-Clause
#          <YEAR> = 2015
#          <OWNER> = Quadrivio Corporation
#

blas.lib <- function(type = NA, processor = NA, index = NA, option = NA, label = NA,
    kernel.path.f = NA, kernel.path.d = NA, kernel.name.f = NA, kernel.name.d = NA,
    kernel.options = NA, work.item.sizes = NA, row.multiple = NA, col.multiple = NA,
    row.tile.size = NA, col.tile.size = NA, fill.on.host = FALSE, verbose = FALSE)
{
    # get option if necessary
    if (!is.na(option[[1]])) {
        if (class(option) != "multiblas.option") stop("option is of wrong type")
        
        if (!is.na(type)) stop("cannot specify both type and option")
        if (!is.na(processor)) stop("cannot specify both processor and option")
        if (!is.na(index)) stop("cannot specify both index and option")

    } else if (is.na(type)){
        if (!is.na(index)) {
            options <- multiblas.options()
            if (index < 1 || index > length(options)) {
                stop("invalid index")
                
            } else {
                option <- options[[index]]
            }
            
        } else {
            # default
            options <- multiblas.options("R")
            option <- options[[1]]
        }
        
    } else if (type == "R" || type == "r") {
        if (!is.na(index)) stop("cannot specify index and type")
        if (!is.na(option)) stop("cannot specify option and type")
        
        if (!is.na(processor)) stop("cannot specify processor with this type")
        
        type <- "R"
        
        options <- multiblas.options("R")
        option <- options[[1]]
        
    } else if (type == "Naive" || type == "naive") {
        if (!is.na(index)) stop("cannot specify index and type")
        if (!is.na(option)) stop("cannot specify option and type")
        
        if (!is.na(processor)) stop("cannot specify processor with this type")
        
        type <- "Naive"
        
        options <- multiblas.options("Naive")
        option <- options[[1]]
        
    } else if (type == "clBLAS" || type == "clblas") {
        if (!is.na(index)) stop("cannot specify index and type")
        if (!is.na(option)) stop("cannot specify option and type")
        
        type <- "clBLAS"
        
        options <- multiblas.options("clBLAS", processor)
        if (length(options) == 0) {
            stop("option not available")
            
        } else {
            option <- options[[1]]
        }
        
    } else if (type == "OpenCL" || type == "opencl") {
        if (!is.na(index)) stop("cannot specify index and type")
        if (!is.na(option)) stop("cannot specify option and type")
        
        type <- "OpenCL"
        
        options <- multiblas.options("OpenCL", processor)
        if (length(options) == 0) {
            stop("option not available")
            
        } else {
            option <- options[[1]]
        }
        
    } else {
        stop("invalid type")
    }
    
    if (!is.na(option[[1]])) {
        blas <- list()
        class(blas) <- "blas.lib"

        blas$type <- option$type
        blas$path <- option$path
        blas$platform <- option$platform
        blas$device <- option$device
        blas$processor <- option$processor

        if (!is.na(blas$device[[1]])) {
            blas$context <- .Call(opencl_context_C, blas$device)
            blas$queue <- .Call(opencl_queue_C, blas$context, blas$device)
            
        } else {
            blas$context <- NA
            blas$queue <- NA
        }

        if (is.na(label)) {
            blas$label <- option$label
            
        } else {
            blas$label <- label
        }
        
        blas$verbose <- verbose
    }
    
    if (blas$type == "R") {
        blas$crossprod <- function(x) { .Call(crossprod_blas_C, x) }
        
    } else if (blas$type == "Naive") {
        blas$crossprod <- function(x) { .Call(crossprod_naive_C, x) }
        
    } else if (blas$type == "clBLAS") {
        blas$crossprod <- function(x) { .Call(crossprod_clblas_C, blas$device, x) }
        
    } else if (blas$type == "OpenCL") {
        if (is.na(kernel.path.f)) {
            kernel.path.f <- system.file("crossprod_f.cl", package = "multiblas")
        }
        
        if (is.na(kernel.path.d)) {
            kernel.path.d <- system.file("crossprod_d.cl", package = "multiblas")
        }
        
        if (!is.null(kernel.name.f) && is.na(kernel.name.f)) {
            kernel.name.f <- "crossprod_f_naive"
        }
        
        if (!is.null(kernel.name.d) && is.na(kernel.name.d)) {
            kernel.name.d <- "crossprod_d_naive"
        }
        
        blas$crossprod.kernel.path.f <- kernel.path.f
        blas$crossprod.kernel.path.d <- kernel.path.d
        blas$crossprod.kernel.name.f <- kernel.name.f
        blas$crossprod.kernel.name.d <- kernel.name.d
        
        blas$crossprod.kernel.options <- kernel.options
        if (is.na(kernel.options)) kernel.options <- ""
        
        if (is.null(kernel.name.f)) {
            blas$crossprod.kernel.f <- NULL
            
        } else {
            source.f <- paste(readLines(kernel.path.f), collapse="\n")
            blas$crossprod.kernel.f <- tryCatch(.Call(opencl_kernel_C, blas$context, blas$device,
            kernel.name.f, source.f, kernel.options, verbose),
            error = function(e) {cat("crossprod_f.cl", e$message); NULL})
        }
        
        if (is.null(kernel.name.d)) {
            blas$crossprod.kernel.d <- NULL
            
        } else {
            source.d <- paste(readLines(kernel.path.d), collapse="\n")
            blas$crossprod.kernel.d <- tryCatch(.Call(opencl_kernel_C, blas$context, blas$device,
                kernel.name.d, source.d, kernel.options, verbose),
                error = function(e) {cat("crossprod_d.cl", e$message); NULL})
        }
        
        blas$work.item.sizes <- work.item.sizes
        blas$row.multiple <- row.multiple
        blas$col.multiple <- col.multiple
        blas$row.tile.size <- row.tile.size
        blas$col.tile.size <- col.tile.size
        blas$fill.on.host <- fill.on.host
        
        blas$crossprod <- function(x) {
            single <- attr(x, "Csingle")
            if (!is.null(single) && single) {
                if (is.null(blas$crossprod.kernel.f)) {
                    stop("single-precision kernel not available")
                }
                
            } else {
                if (is.null(blas$crossprod.kernel.d)) {
                    stop("double-precision kernel not available")
                }
            }
            
            if (is.na(work.item.sizes[1])) {
                work.item.sizes = c(1, 1, 1)
            }
            
            class(work.item.sizes) <- "integer"
            
            if (is.na(row.multiple)) {
                row.multiple = 1
            }
            
            class(row.multiple) <- "integer"
            
            if (is.na(col.multiple)) {
                col.multiple = 1
            }
            
            class(col.multiple) <- "integer"
            
            if (is.na(row.tile.size)) {
                row.tile.size = 1
            }
            
            class(row.tile.size) <- "integer"
            
            if (is.na(col.tile.size)) {
                col.tile.size = 1
            }
            
            class(col.tile.size) <- "integer"
            
            .Call(opencl_calc_x_C, blas$context, blas$crossprod.kernel.f, blas$crossprod.kernel.d,
                blas$queue, x, work.item.sizes, row.multiple, col.multiple,
                row.tile.size, col.tile.size, fill.on.host, verbose)
        }
        
    } else {
        blas$crossprod <- NA
    }

    blas$fun <- sqrt
    
	return(blas)
}
