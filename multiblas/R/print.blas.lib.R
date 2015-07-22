#
#  print.blas.lib.R
#  multiblas
#
#  Created by MPB on 6/29/15.
#  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
#  License http://opensource.org/licenses/BSD-2-Clause
#          <YEAR> = 2015
#          <OWNER> = Quadrivio Corporation
#

print.blas.lib <-
function(x, ...)
{
    # print label
	cat(x$label, "\n")
}
