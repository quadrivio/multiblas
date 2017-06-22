\name{blas.lib}
\alias{blas.lib}
\alias{print.blas.lib}
\title{blas.lib}
\description{
An object containing methods for calling a version of BLAS routines.
}
\usage{
blas.lib(type, processor, index, option, label, kernel.info, fill.on.host, verbose)
\method{print}{blas.lib}(x, \dots)
}

\arguments{
\item{type}{input string }
\item{processor}{input string }
\item{index}{input number }
\item{option}{input object }
\item{label}{input string }
\item{kernel.info}{input list }
\item{fill.on.host}{input boolean }
\item{verbose}{input boolean }
\item{x}{input object }
\item{\dots}{other stuff}
}
\value{
Returns object containing BLAS methods.
\item{crossprod}{Function crossprod(x) where x is input matrix.}
\item{gemm}{Function gemm(A, transposeA = FALSE, B, transposeB = FALSE, C = NA, alpha = 1.0, beta = 0.0)}
}
\author{
Michael Budiansky
quadrivio@7402.org
}
\examples{
blas <- blas.lib(type="R")
print(blas)
}

\keyword{multiblas}
