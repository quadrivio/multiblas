## template

---

### Description

This is a project template.

### Usage

* C++

To use the C++ source files within a separate C++ project, copy the contents of
template/src. Compile with the preprocessor definition `RPROJECT` equal to 0 to remove the
R-specific code, _i.e._, use `-DRPROJECT=0` in gcc or insert `#define RPROJECT 0` at the
top of the file shim.h. Messages will be sent to stderr. Logical and runtime
errors will be thrown with `std::logic_error` or `std::runtime_error`.

* R

To create a package usable within R, build and check the R package with the standard
command-line tools:

```
cd path_to_top_directory
R CMD build template
R CMD check template_0.00-0.tar.gz
```

Then, install the package within R in the usual way:

```
setwd("path_to_top_directory")
install.packages("template_0.00-0.tar.gz", repos = NULL, type = "source")
library('template')
```

When the C++ files are compiled for use in R, the preprocessor parameter `RPROJECT` is
automatically defined to 1. The code in shim.h then causes messages to be rerouted to
the R function `Rprintf()` instead of to stderr. Errors are signaled via the R
function `error()` instead of by throwing exceptions. `R_CheckUserInterrupt()` is called
periodically to allow interruption of lengthy calculations.

* Xcode

The XCode project includes `main.cpp` and other files outside of the R package. These
files contain some integration and unit tests, and provide a command-line interface to
the algorithms.

To compile the package with XCode copy the `template` and `templateXC` directories into the
same parent directory. Create an empty `bin` directory in the parent directory as well.
Launch the file `templateXC.xcodeproj` to start XCode. The default scheme builds a
command-line tool in the neighboring bin/ directory, compiled under the Debug build
configuration. For use as a tool, you probably want the Release configuration, so choose
Edit Scheme, choose the "Run template" section, select the Info tab, and change the Build
Configuration to Release.

To run the test code, create a New Scheme, and name it "template test". Choose Edit Scheme,
choose the "Run template" section, select the Arguments tab, and add "--test" to the section
Arguments Passed On Launch. Select the Info tab, and change the Build Configuration to
Release. If you want to dig into the test code, you can add the argument "-v" for verbose
output and keep the Build Configuration set to Debug.

To run ad-hoc test code or to experiment with other code in the project, modify the 
develop() function in develop.cpp. Create a New Scheme, and name it "template develop".
Choose Edit Scheme, choose the "Run template" section, select the Arguments tab, and add the
argument, "--develop".

To compile the R-specific code in XCode, 1) Add to the project the R Framework (which is likely to be at /Library/Frameworks/R.framework);
2) change the preprocesser macro RPACKAGE to have a value of 1 (instead of 0), by editing Target > Build Settings > Apple LLVM 6.0 - Preprocessing > Preprocessor Macros.

### Development Environment

Max OS 10.9.5  
R 3.1.3  
XCode 6.2  

This project has not yet been tested or modified for Windows. It has been very lightly tested on Linux (CentOS 6.6).

### License

BSD
