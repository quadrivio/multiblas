#
#  multiblas.options.R
#  multiblas
#
#  Created by MPB on 6/29/15.
#  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
#  License http://opensource.org/licenses/BSD-2-Clause
#          <YEAR> = 2015
#          <OWNER> = Quadrivio Corporation
#

multiblas.options <- function(types = NA, processors = NA)
{
    options <- list()
    class(options) <- "multiblas.options"
    
    if (is.na(types) || "R" %in% types || "r" %in% types) {
        so.list <- list.files(file.path(R.home(), "lib"), pattern="\\.so")
        dylib.list <- list.files(file.path(R.home(), "lib"), pattern="\\.dylib")
        dll.list <- list.files(file.path(R.home(), "bin", "x64"), pattern="\\.dll")
        
        if (length(dylib.list) > 0) {
            path <- Sys.readlink(file.path(R.home(), "lib", "libRblas.dylib"))
            
        } else if (length(so.list) > 0) {
            path <- Sys.readlink(file.path(R.home(), "lib", "libRblas.so"))
            
        } else if (length(dylib.list) > 0) {
            path <- file.path(R.home(), "bin", "x64", "Rblas.dll")
            
        } else {
            path <- NA
        }
        
        if (is.na(path)) {
            label <- "Rblas"
            
        } else {
            label <- basename(path)
        }
        
        option <- list(type="R", platform=NA, device=NA, processor=NA, label=label, path=path)
        class(option) <- "multiblas.option"
        
        options <- append.option(options, option)
    }
    
    if (is.na(types) || "Naive" %in% types || "naive" %in% types) {
        label <- "Naive"
        option <- list(type="Naive", platform=NA, device=NA, processor=NA, label=label, path=NA)
        class(option) <- "multiblas.option"
        
        options <- append.option(options, option)
    }
    
    if (is.na(types) || "clBLAS" %in% types || "clblas" %in% types) {
        platforms <- .Call(opencl_platforms_C)
        for (platform in platforms) {
            devices <- .Call(opencl_devices_C, platform)
            for (device in devices) {
                processor.info <- .Call(property_device_C, device, "CL_DEVICE_TYPE")
                processor = switch(processor.info,
                "CL_DEVICE_TYPE_CPU" = "CPU",
                "CL_DEVICE_TYPE_GPU" = "GPU",
                "CL_DEVICE_TYPE_ACCELERATOR" = "APU")
                
                if (is.na(processors) || processor %in% processors) {
                    option <- list(type="clBLAS", platform=platform, device=device, processor=processor, label=device$name, path=NA)
                    class(option) <- "multiblas.option"
                    
                    options <- append.option(options, option)
                }
            }
        }
    }
    
    if (is.na(types) || "OpenCL" %in% types || "opencl" %in% types) {
        platforms <- .Call(opencl_platforms_C)
        for (platform in platforms) {
            devices <- .Call(opencl_devices_C, platform)
            for (device in devices) {
                processor.info <- .Call(property_device_C, device, "CL_DEVICE_TYPE")
                processor = switch(processor.info,
                "CL_DEVICE_TYPE_CPU" = "CPU",
                "CL_DEVICE_TYPE_GPU" = "GPU",
                "CL_DEVICE_TYPE_ACCELERATOR" = "APU")
                
                if (is.na(processors) || processor %in% processors) {
                    option <- list(type="OpenCL", platform=platform, device=device, processor=processor, label=device$name, path=NA)
                    class(option) <- "multiblas.option"
                    
                    options <- append.option(options, option)
                }
            }
        }
    }
    
    return(options)
}

print.multiblas.options <- function(x, ...) {
    if (length(x) > 0) {
        for (index in 1:length(x)) {
            cat(paste("[[", index, "]] ", sep=""))
            print(x[[index]])
            cat("\n")
        }
    }
}

print.multiblas.option <- function(x, ...) {
    cat(paste(x$type, " ", x$label, sep=""))
}

append.option <- function(options, option) {
    options[[length(options) + 1]] = option
    return(options)
}

