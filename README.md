## multiblas

---

_This project is a work in progress. Documentation and testing are not complete._

### Description

This project implements routines for matrix operations on CPU and AMD GPU processors. The project
includes the following components:

* A set of C++ source files for the core algorithms; these can be used independently of R. 
* A set of R package files that encapsulate the C++ source and can be installed and used
within R.
* A set of Mac OS XCode project files that can be used for developing and testing the
C++ files.
* A makefile that can be used to compile the project from the Mac OS X or Linux command line.

### Prerequisites

This project requires installation of the AMD clBLAS library.

On OS X, the project uses the built-in Accelerate framework.
On Linux, the project requires installation of the OpenCL and OpenBLAS libraries.

See http://blog.quadrivio.com for installation instructions.

### Usage

* C++

To use the C++ source files within a separate C++ project, copy the contents of
multiblas/src. Compile with the preprocessor definition `RPROJECT` equal to 0 to remove the
R-specific code, _i.e._, use `-DRPROJECT=0` in gcc or insert `#define RPROJECT 0` at the
top of the file shim.h. Messages will be sent to stderr. Logical and runtime
errors will be thrown with `std::logic_error` or `std::runtime_error`.

* R

To create a package usable within R, build and check the R package as follows:

```
# OS X
cd path_to_top_directory
export PKG_CPPFLAGS="-I/usr/local/clblas/include"
export PKG_LIBS='$(BLAS_LIBS) -framework OpenCL -L/usr/local/clblas/lib -lclBLAS'
R CMD build multiblas
R CMD check multiblas_0.90-1.tar.gz

# Linux
cd path_to_top_directory
export PKG_CPPFLAGS="-I/usr/lib64/openblas/include -I/opt/AMDAPPSDK-3.0-0-Beta/include -Dnullptr='NULL' -I/opt/clBLAS-2.4.0-Linux-x64/include"
export PKG_LIBS='$(BLAS_LIBS) -L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64 -lOpenCL -L/opt/clBLAS-2.4.0-Linux-x64/lib64 -lclBLAS'
R CMD build multiblas
R CMD check multiblas_0.90-1.tar.gz
```

(If you download and expand the zip file, the top directory is named multiblas-master.) Then, install the package within R as follows:

```
# OS X
setwd("path_to_top_directory")
Sys.setenv(PKG_CPPFLAGS = "-I/usr/local/clblas/include")
Sys.setenv(PKG_LIBS = "$(BLAS_LIBS) -framework OpenCL -L/usr/local/clblas/lib/ -lclBLAS")
install.packages("multiblas_0.90-1.tar.gz", repos = NULL, type = "source")
library('multiblas')

# Linux
setwd("path_to_top_directory")
Sys.setenv(PKG_CPPFLAGS = "-I/usr/lib64/openblas/include -I/opt/AMDAPPSDK-3.0-0-Beta/include -Dnullptr='NULL' -I/opt/clBLAS-2.4.0-Linux-x64/include")
Sys.setenv(PKG_LIBS = "$(BLAS_LIBS) -L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64 -lOpenCL -L/opt/clBLAS-2.4.0-Linux-x64/lib64 -lclBLAS")
install.packages("multiblas_0.90-1.tar.gz", repos = NULL, type = "source")
library('multiblas')
```

When the C++ files are compiled for use in R, the preprocessor parameter `RPROJECT` is
automatically defined to 1. The code in shim.h then causes messages to be rerouted to
the R function `Rprintf()` instead of to stderr. Errors are signaled via the R
function `error()` instead of by throwing exceptions.

* XCode and command-line

The project includes `main.cpp` to allow the code to run outside of R. Running the project from
XCode or the command-line causes a few sample matrix multiplications to be calculated.

To build using the command line, `cd` to the directory containing `makefile` and type `make`.

### Development Environment

Max OS 10.10.5  
CentOS 6.6  
R 3.2.1  
XCode 6.4   

This project has not been tested or modified for Windows.

### License

BSD
