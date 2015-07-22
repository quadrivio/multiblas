//
//  utils.cpp
//  entree
//
//  Created by MPB on 5/8/13.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//  License http://opensource.org/licenses/BSD-2-Clause
//          <YEAR> = 2015
//          <OWNER> = Quadrivio Corporation
//

//
// Miscellaneous utility functions
//

#include <fstream>  // must precede .h includes

#include "utils.h"

#include "shim.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#if defined _WIN32 || defined _WIN64
#include <windows.h>
#else
#include <sys/stat.h>
#endif

using namespace std;

// ========== Functions ============================================================================

// throw std::logic_error with custom message include source file name and line number
void throw_logic_error(const std::string& path, int line, const std::string& msg)
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
    oss << msg << " at " << file << " line " << line;
    throw logic_error(oss.str());
}

// throw std::runtime_error with custom message; include source file name and line number if DEBUG
// is defined
void throw_runtime_error(const std::string& path, int line, const std::string& msg)
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
    
    CERR << msg << " at " << file << " line " << line << endl;
#endif
    
    throw runtime_error(msg);
}

// convert time_t to local time string in ISO format
string localTimeString(const time_t t)
{
    const size_t MAXSIZE = 256;
    char str[MAXSIZE];
    strftime(str, MAXSIZE, "%Y-%m-%d %H:%M:%S", localtime(&t));
    
    return string(str);
}

// return true if entire string is parsable as number
bool isNumeric(const std::string str)
{
    istringstream iss(str);
    double d;
    iss >> d;
    
    bool result = !iss.fail() && iss.eof();
    
    return result;
}

// return value from parsing string as double
double toDouble(const std::string str)
{
    istringstream iss(str);
    double d;
    iss >> d;

    return d;
}

// return value from parsing string as long
long toLong(const std::string str)
{
    istringstream iss(str);
    long k;
    iss >> k;
    
    return k;
}

// return value from parsing string as int
int toInteger(const std::string str)
{
    istringstream iss(str);
    int k;
    iss >> k;
    
    return k;
}

// write string to file
void stringToFile(const std::string& str, const std::string& path)
{
    ofstream ofs(path.c_str());
    
    ofs << str;
    
    ofs.close();
}

// read string from file
void fileToString(const std::string& path, std::string& str)
{
    ifstream ifs(path.c_str());
    
    ostringstream oss;
    oss << ifs.rdbuf();
    
    str = oss.str();
    
    ifs.close();
}

// return current working directory
std::string getWorkingDirectory()
{
    const size_t BUFSIZE = 2048;
    char cwd[BUFSIZE];
    if (getcwd(cwd, BUFSIZE) == NULL) {
        RUNTIME_ERROR_IF(true, "unable to get working directory");
    };

    return cwd;
}

// return error message for bad path
std::string badPathErrorMessage(const std::string& file)
{
    string msg = "bad path ";

#if defined _WIN32 || defined _WIN64
    msg += getWorkingDirectory() + "\\" +  file;
#else
    msg += getWorkingDirectory() + "/" + file;
#endif
    
    return msg;
}

// for debugging and testing; create directory
void makeDir(const std::string& path)
{
#if defined _WIN32 || defined _WIN64
    bool result = CreateDirectory(path.c_str(), NULL);
    RUNTIME_ERROR_IF(!result, "makeDir fail");
    
#else
    int result = mkdir(path.c_str(), 700);
    RUNTIME_ERROR_IF(result != 0, "makeDir fail");
#endif
}

// for debugging and testing; remove directory
void removeDir(const std::string& path)
{
#if defined _WIN32 || defined _WIN64
    bool result = RemoveDirectory(path.c_str());
    RUNTIME_ERROR_IF(!result, "removeDir fail");
    
#else
    int result = rmdir(path.c_str());
    RUNTIME_ERROR_IF(result != 0, "removeDir fail");
#endif
}

// for debugging and testing; location at which to set a breakpoint
void noop()
{
}

// ========== Tests ================================================================================

// component tests
void ctest_utils(int& totalPassed, int& totalFailed, bool verbose)
{
    int passed = 0;
    int failed = 0;
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // throw_logic_error
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // throw_runtime_error
        
    // ~~~~~~~~~~~~~~~~~~~~~~
    // localTimeString
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // isNumeric
    
    if (isNumeric("123")) passed++; else failed++;
    if (isNumeric(" 12.3")) passed++; else failed++;
    if (isNumeric(".123")) passed++; else failed++;
    if (!isNumeric("123A")) passed++; else failed++;
    if (!isNumeric("B")) passed++; else failed++;
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // toDouble
    
    if ((int)toDouble("2.0") == 2) passed++; else failed++;
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // toLong
    
    if (toLong("128") == 128) passed++; else failed++;
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // stringToFile
    // fileToString
    
    remove("foo.txt");
    
    string outStr = "Hello World\nFoo\tBar";
    stringToFile(outStr, "foo.txt");
    
    string inStr;
    fileToString("foo.txt", inStr);
    
    if (outStr == inStr) passed++; else failed++;
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // getWorkingDirectory
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // badPathErrorMessage
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // makeDir
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // removeDir
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // noop
    
    // for TDD purists:
    noop();
    passed++;
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    
    if (verbose) {
        CERR << "utils.cpp" << "\t" << passed << " passed, " << failed << " failed" << endl;
    }
    
    totalPassed += passed;
    totalFailed += failed;
}

void cover_utils(bool verbose)
{
    // ~~~~~~~~~~~~~~~~~~~~~~
    // throw_logic_error
    
    try {
        throw_logic_error("foo", 1, "message A");
        
    } catch(logic_error e) {
        if (verbose) {
            CERR << "one \"message A\" error follows:" << endl;
            CERR << e.what() << endl;
        }
    }
    
#if defined _WIN32 || defined _WIN64
    string path = "foo\\bar";
    
#else
    string path = "foo/bar";
#endif
    
    try {
        throw_logic_error(path, 2, "message B");
        
    } catch(logic_error e) {
        if (verbose) {
            CERR << "one \"message B\" error follows:" << endl;
            CERR << e.what() << endl;
        }
    }
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // throw_runtime_error
    
#ifdef DEBUG
    CERR << "one \"message C\" error follows:" << endl;
#endif
    
    try {
        throw_runtime_error("foo", 3, "message C");
        
    } catch(runtime_error e) {
        if (verbose) {
            CERR << "one \"message C\" error follows:" << endl;
            CERR << e.what() << endl;
        }
    }
 
    // ~~~~~~~~~~~~~~~~~~~~~~
    // localTimeString
    
    time_t t;
    time(&t);
    string str = localTimeString(t);
    if (verbose) {
        CERR << str << endl;
    }
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // isNumeric
    
    isNumeric("123");
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // toDouble
    
    toDouble("1.23");
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // toLong
    
    toLong("123");
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // stringToFile
    
    stringToFile("foo", "foo.txt");
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // fileToString
    
    fileToString("foo.txt", str);
    remove("foo.txt");
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // getWorkingDirectory
    
    getWorkingDirectory();
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // badPathErrorMessage

    badPathErrorMessage("foo");

    // ~~~~~~~~~~~~~~~~~~~~~~
    // makeDir
    
    makeDir("foo");
    
    // ~~~~~~~~~~~~~~~~~~~~~~
    // removeDir
    
    removeDir("foo");
}
