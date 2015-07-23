\name{blas.lib}
\alias{blas.lib}
\alias{print.blas.lib}
\title{blas.lib}
\description{
    An object containing methods for calling a version of BLAS routines.
}
\usage{
blas.lib(type, processor, index, option, label, kernel.path.f, kernel.path.d,
    kernel.name.f, kernel.name.d, kernel.options, work.item.sizes, row.multiple, col.multiple,
    row.tile.size, col.tile.size, fill.on.host, verbose)
    \method{print}{blas.lib}(x, \dots)
}

\arguments{
    \item{type}{input string }
    \item{processor}{input string }
    \item{index}{input number }
    \item{option}{input object }
    \item{label}{input string }
    \item{kernel.path.f}{input string }
    \item{kernel.path.d}{input string }
    \item{kernel.name.f}{input string }
    \item{kernel.name.d}{input string }
    \item{kernel.options}{input string }
    \item{x}{input object }
    \item{work.item.sizes}{input numbers }
    \item{row.multiple}{input number }
    \item{col.multiple}{input number }
    \item{row.tile.size}{input number }
    \item{col.tile.size}{input number }
    \item{fill.on.host}{input boolean }
    \item{verbose}{input boolean }
    \item{\dots}{other stuff}
}
\value{
    Returns object containing BLAS methods.
    \item{crossprod}{Function crossprod(x) where x is input matrix.}
    \item{gemm}{Function gemm(A, transposeA = FALSE, B, transposeB = FALSE, C = NA, alpha = 1.0, beta = 0.0)}
}
\author{
    Michael Budiansky, Quadrivio Corporation
}
\examples{
    blas <- blas.lib(type="R")
    print(blas)
}

\keyword{multiblas}
