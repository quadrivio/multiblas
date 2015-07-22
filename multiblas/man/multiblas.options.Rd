\name{multiblas.options}
\alias{multiblas.options}
\alias{multiblas.option}
\alias{print.multiblas.options}
\alias{print.multiblas.option}
\title{multiblas.options}
\description{
    Function returning library options.
}
\usage{
    multiblas.options(types, processors)
    \method{print}{multiblas.options}(x, \dots)
    multiblas.option
    \method{print}{multiblas.option}(x, \dots)
}

\arguments{
    \item{types}{input string }
    \item{processors}{input vector or string}
    \item{x}{input object }
    \item{...}{other stuff}
}
\value{
    Returns dataframe listing library options.
}
\author{
    Michael Budiansky, Quadrivio Corporation
}
\examples{
    multiblas.options(type="OpenCL")
}

\keyword{multiblas}
