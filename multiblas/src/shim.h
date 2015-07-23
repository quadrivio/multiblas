//
//  shim.h
//  entree
//
//  Created by MPB on 5/7/13.
//  Copyright (c) 2013 Quadrivio Corporation. All rights reserved.
//  License http://opensource.org/licenses/BSD-2-Clause
//          <YEAR> = 2013
//          <OWNER> = Quadrivio Corporation
//

//
// Switch code compilation for use in a R package as opposed use in stand-alone C++ project
//

#ifndef entree_shim_h
#define entree_shim_h

#ifndef RPACKAGE
#define RPACKAGE 1
#endif

// see utils.h for default definitions for LOGIC_ERROR_IF and RUNTIME_ERROR_IF

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#if RPACKAGE

// ---------- use this block to include R headers ----------
#ifndef NO_C_HEADERS
#define NO_C_HEADERS
#endif
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <cmath>
#include <iostream>
#include <sstream>
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
// ---------------------------------------------------------

#include <streambuf>
#include <string>

// ========== Macros ===============================================================================

#define CERR gRerr

#define LOGIC_ERROR_IF(x, msg) if (x) { r_logic_error(__FILE__, __LINE__, msg); }

#define RUNTIME_ERROR_IF(x, msg) if (x) { r_runtime_error(__FILE__, __LINE__, msg); }

// ========== Function Headers =====================================================================

// call R error function with custom message include source file name and line number
void r_logic_error(const std::string& path, int line, const std::string& msg);

// call R error function with custom message; include source file name and line number if DEBUG
// is defined
void r_runtime_error(const std::string& path, int line, const std::string& msg);

// ========== Globals ==============================================================================

// need these to pick up values of TRUE and FALSE as defined in R/Boolean.h
extern const Rboolean RTRUE;
extern const Rboolean RFALSE;

extern std::ostream gRerr;

#else // ==========================================================================================

// ========== Macros ===============================================================================

#define CERR std::cerr

#endif // ==========================================================================================

#endif
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
