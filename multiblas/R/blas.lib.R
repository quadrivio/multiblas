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

crossprod.defaults <- list(
    kernel.path.f = system.file("crossprod_f.cl", package = "multiblas"),
    kernel.path.d = system.file("crossprod_d.cl", package = "multiblas"),
    kernel.name.f = "crossprod_f_naive",
    kernel.name.d = "crossprod_d_naive",
    kernel.options = "",
    work.item.sizes = c(1, 1, 1),
    row.multiple = 1,
    col.multiple = 1,
    row.tile.size = 1,
    col.tile.size = 1
)

gemm.defaults <- list(
    kernel.path.f = system.file("gemm_f.cl", package = "multiblas"),
    kernel.path.d = system.file("gemm_d.cl", package = "multiblas"),
    kernel.name.f = "gemm_f_naive",
    kernel.name.d = "gemm_d_naive",
    kernel.options = "",
    work.item.sizes = c(1, 1, 1),
    row.multiple = 1,
    col.multiple = 1,
    row.tile.size = 1,
    col.tile.size = 1
)

blas.lib <- function(type = NA, processor = NA, index = NA, option = NA, label = NA,
kernel.info = NA, fill.on.host = FALSE, verbose = FALSE)
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
        
    } else if (type == "Base" || type == "base") {
        if (!is.na(index)) stop("cannot specify index and type")
        if (!is.na(option)) stop("cannot specify option and type")
        
        if (!is.na(processor)) stop("cannot specify processor with this type")
        
        type <- "Base"
        
        options <- multiblas.options("Base")
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
        
        blas$gemm <- function(A, transposeA = FALSE, B, transposeB = FALSE, C = NA, alpha = 1.0, beta = 0.0) {
            .Call(gemm_blas_C, A, transposeA, B, transposeB, C, alpha, beta)
        }
        
    } else if (blas$type == "Naive") {
        blas$crossprod <- function(x) { .Call(crossprod_naive_C, x) }
        
        blas$gemm <- function(A, transposeA = FALSE, B, transposeB = FALSE, C = NA, alpha = 1.0, beta = 0.0) {
            .Call(gemm_naive_C, A, transposeA, B, transposeB, C, alpha, beta)
        }
        
    } else if (blas$type == "Base") {
        blas$crossprod <- function(x) { crossprod(x) }
        
        blas$gemm <- function(A, transposeA = FALSE, B, transposeB = FALSE, C = NA, alpha = 1.0, beta = 0.0) {
            if (transposeA) A <- t(A)
            if (transposeB) B <- t(B)
            if (is.na(C)) C <- 0.0
            
            result <- alpha * A %*% B + beta * C
            
            singleA <- attr(A, "Csingle")
            singleA <- !is.null(singleA) && singleA
            
            singleB <- attr(B, "Csingle")
            singleB <- !is.null(singleB) && singleB
            
            singleC <- attr(C, "Csingle")
            singleC <- !is.null(singleC) && singleC
            
            if (singleA || singleB || singleC) {
                attr(result, "Csingle") <- TRUE
            }
            
            return(result)
        }
        
    } else if (blas$type == "clBLAS") {
        blas$crossprod <- function(x) { .Call(crossprod_clblas_C, blas$device, x) }
        
        blas$gemm <- function(A, transposeA = FALSE, B, transposeB = FALSE, C = NA, alpha = 1.0, beta = 0.0) {
            .Call(gemm_clblas_C, blas$device, A, transposeA, B, transposeB, C, alpha, beta)
        }
        
    } else if (blas$type == "OpenCL") {
        crossprod.info <- crossprod.defaults
        gemm.info <- gemm.defaults
        
        if (!is.na(kernel.info)) {
            crossprod.info <- replace(crossprod.info, names(kernel.info$crossprod), kernel.info$crossprod)
            gemm.info <- replace(gemm.info, names(kernel.info$gemm), kernel.info$gemm)
        }
        
        get.kernel.source.f <- function(info) {
            if (is.null(info$kernel.name.f)) {
                kernel.f <- NULL
                
            } else {
                source.f <- paste(readLines(info$kernel.path.f), collapse="\n")
                kernel.f <- tryCatch(.Call(opencl_kernel_C, blas$context, blas$device,
                info$kernel.name.f, source.f, info$kernel.options, verbose),
                error = function(e) {cat(info$kernel.path.f, e$message, "\n"); NULL})
            }
            
            return(kernel.f)
        }
        
        get.kernel.source.d <- function(info) {
            if (is.null(info$kernel.name.d)) {
                kernel.d <- NULL
                
            } else {
                source.d <- paste(readLines(info$kernel.path.d), collapse="\n")
                kernel.d <- tryCatch(.Call(opencl_kernel_C, blas$context, blas$device,
                info$kernel.name.d, source.d, info$kernel.options, verbose),
                error = function(e) {cat(info$kernel.path.d, e$message, "\n"); NULL})
            }
            
            return(kernel.d)
        }
        
        crossprod.info$kernel.f <- get.kernel.source.f(crossprod.info)
        crossprod.info$kernel.d <- get.kernel.source.d(crossprod.info)
        blas$crossprod.info <- crossprod.info
        
        gemm.info$kernel.f <- get.kernel.source.f(gemm.info)
        gemm.info$kernel.d <- get.kernel.source.d(gemm.info)
        blas$gemm.info <- gemm.info
        
        blas$crossprod <- function(x) {
            cat("crossprod(x)\n")
            
            single <- attr(x, "Csingle")
            if (!is.null(single) && single) {
                if (is.null(crossprod.info$kernel.f)) {
                    stop("single-precision kernel not available")
                }
                
            } else {
                if (is.null(crossprod.info$kernel.d)) {
                    stop("double-precision kernel not available")
                }
            }
            
            .Call(opencl_calc_x_C, blas$context, crossprod.info$kernel.f, crossprod.info$kernel.d,
            blas$queue, x, as.integer(crossprod.info$work.item.sizes), as.integer(crossprod.info$row.multiple), as.integer(crossprod.info$col.multiple),
            as.integer(crossprod.info$row.tile.size), as.integer(crossprod.info$col.tile.size), fill.on.host, verbose)
        }
        
        blas$gemm <- function(A, transposeA = FALSE, B, transposeB = FALSE, C = NA, alpha = 1.0, beta = 0.0) {
            singleA <- attr(A, "Csingle")
            singleB <- attr(B, "Csingle")
            singleC <- attr(C, "Csingle")
            
            singleA <- !is.null(singleA) && singleA
            singleB <- !is.null(singleB) && singleB
            singleC <- !is.null(singleC) && singleC
            
            if (singleA || singleB || singleC) {
                if (is.null(gemm.info$kernel.f)) {
                    stop("single-precision kernel not available")
                }
                
            } else {
                if (is.null(gemm.info$kernel.d)) {
                    stop("double-precision kernel not available")
                }
            }
            
            .Call(opencl_calc_gemm_C, blas$context, gemm.info$kernel.f, gemm.info$kernel.d,
            blas$queue, A, transposeA, B, transposeB, C,
            alpha, beta, as.integer(gemm.info$work.item.sizes), as.integer(gemm.info$row.multiple), as.integer(gemm.info$col.multiple),
            as.integer(gemm.info$row.tile.size), as.integer(gemm.info$col.tile.size), fill.on.host, verbose)
        }
        
    } else {
        blas$crossprod <- NA
        blas$gemm <- NA
    }
    
    return(blas)
}
