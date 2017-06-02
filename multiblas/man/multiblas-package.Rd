\name{multiblas-package}
\alias{multiblas-package}
\docType{package}
\title{multiblas Package}
\description{
    A package to enable calling CPU and OpenCL BLAS libraries.
}
\details{
\tabular{ll}{
    Package: \tab multiblas\cr
    Type: \tab Package\cr
    Version: \tab 0.91-1\cr
    Date: \tab 2015-09-08\cr
    License: \tab BSD + file LICENSE\cr
}
    This package enables calling CPU and OpenCL BLAS libraries.
}
\author{
    Michael Budiansky, Quadrivio Corporation
    software@quadrivio.com
}

\keyword{package}
\keyword{multiblas}

\examples{
    blas <- blas.lib(type="R")
    print(blas)

    multiblas.options(type="OpenCL")
}
