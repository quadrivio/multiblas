//
//  shim.cpp
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

#include "shim.h"

#if RPACKAGE

// need these to pick up values of TRUE and FALSE as defined in R/Boolean.h
const Rboolean RTRUE = TRUE;
const Rboolean RFALSE = FALSE;

#include <sstream>

using namespace std;

// ========== Local Class Declarations =============================================================

// custom streambuf that sends text to R function Rprintf()
class RStreambuf : public std::streambuf {
protected:
    virtual RStreambuf::int_type overflow(RStreambuf::int_type c);
    virtual std::streamsize xsputn(const RStreambuf::char_type* s, std::streamsize n);
};

// ========== Globals ==============================================================================

RStreambuf gRStreambuf;

std::ostream gRerr(&gRStreambuf);

// ========== Local Classes ========================================================================

// custom streambuf that sends text to R function Rprintf()

RStreambuf::int_type RStreambuf::overflow(RStreambuf::int_type c)
{
    if (c != EOF) {
        Rprintf("%c", c);
    }
            
    return c;
}

std::streamsize RStreambuf::xsputn(const RStreambuf::char_type* s, std::streamsize n)
{
    if (n > 0) {
        Rprintf("%.*s", n, s);
    }
    
    return n;
}

// ========== Functions ============================================================================

// call R error function with custom message include source file name and line number
void r_logic_error(const std::string& path, int line, const std::string& msg)
{
    string file;
    
#if defined _WIN32 || defined _WIN64
    size_t start = path.find_last_of("\\/");
#else
    size_t start = path.find_last_of("/");
#endif
    
    if (start == string::npos) {
        file = path;
        
    } else {
        file = path.substr(start + 1);
    }
    
    ostringstream oss;
    oss << msg << " at " << file << " line " << line << endl;
    
    error(oss.str().c_str());
}

// call R error function with custom message; include source file name and line number if DEBUG
// is defined
void r_runtime_error(const std::string& path, int line, const std::string& msg)
{
#ifdef DEBUG
    string file;
    
#if defined _WIN32 || defined _WIN64
    size_t start = path.find_last_of("\\/");
#else
    size_t start = path.find_last_of("/");
#endif
    
    if (start == string::npos) {
        file = path;
        
    } else {
        file = path.substr(start + 1);
    }
    
    cerr << msg << " at " << file << " line " << line << endl;
#endif
    
    error(msg.c_str());
}


#else

#endif
