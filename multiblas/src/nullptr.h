//
//  nullptr.h
//  multiBLAS.XC
//
//  Created by michael on 6/21/17.
//  Copyright © 2017 Quadrivio Corporation. All rights reserved.
//

// nullptr compatibility

#ifndef nullptr_h
#define nullptr_h

#include <cstddef>

// thanks to https://stackoverflow.com/questions/42401431/how-do-i-reliably-detect-support-for-nullptr
#if defined(__GNUC__)
#  define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#  if defined(__GXX_EXPERIMENTAL_CXX0X__) || (__cplusplus >= 201103L)
#    define GCC_CXX11
#  endif
#  if (GCC_VERSION < 40600) || !defined(GCC_CXX11)
#    define NO_CXX11_NULLPTR
#  endif
#endif

#if defined(_MSC_VER)
#  if (_MSC_VER < 1600)
#    define NO_CXX11_NULLPTR
#  endif
#endif

#if defined(__clang__)
#  if !__has_feature(cxx_nullptr)
#    define NO_CXX11_NULLPTR
#  endif
#endif

// needed to add this, because XCode cstddef can define nullptr
#ifdef nullptr
#define NULLPTR_VIA_MACRO
#endif

#if defined(NO_CXX11_NULLPTR) && !defined(NULLPTR_VIA_MACRO)
#  define nullptr 0
#endif


#endif /* nullptr_h */
