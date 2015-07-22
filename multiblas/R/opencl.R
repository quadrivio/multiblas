#
#  opencl.R
#  multiblas
#
#  Created by MPB on 6/30/15.
#  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
#  License http://opensource.org/licenses/BSD-2-Clause
#          <YEAR> = 2015
#          <OWNER> = Quadrivio Corporation
#

null.externalptr <- function() {
    z <- .Call(null_externalptr_C)
    
    return(z)
}

is.externalptr.null <- function(ptr) {
    z <- .Call(is_externalptr_null_C, ptr)
    
    return(z)
}

opencl.platforms <- function() {
    z <- .Call(opencl_platforms_C)
    
    return(z)
}

print.opencl.platforms <- function(x, ...) {
    if (length(x) > 0) {
        for (index in 1:length(x)) {
            cat(paste("[", index, "] ", x[[index]]$name, "\n", sep=""))
        }
    }
}

print.opencl.platform <- function(x, ...) {
    cat(x$name)
}

opencl.devices <- function(platform) {
    # make sure types are correct before calling C function
    
    # x
    if (class(platform) != "opencl.platform") {
        stop("requires argument of class 'opencl.platform'")
    }
    
    z <- .Call(opencl_devices_C, platform)
    
    return(z)
}

print.opencl.devices <- function(x, ...) {
    if (length(x) > 0) {
        for (index in 1:length(x)) {
            cat(paste("[", index, "] ", x[[index]]$name, "\n", sep=""))
        }
    }
}

print.opencl.device <- function(x, ...) {
    cat(x$name)
}

property <- function(x, name) UseMethod("property")

property.opencl.platform <- function(x, name) {
    info <- .Call(property_platform_C, x, name)
    
    return(info)
}

property.opencl.device <- function(x, name) {
    info <- .Call(property_device_C, x, name)
    
    return(info)
}

