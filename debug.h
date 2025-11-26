#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#include <stdlib.h>
#include <cstddef>

#ifdef NDEBUG
    #define ASSERT(condition) ((void)0)
#else
    #define ASSERT(condition) \
        do { \
            if (!(condition)) { \
                fprintf(stderr, "Assertion failed: %s\n", #condition); \
                fprintf(stderr, "  File: %s\n", __FILE__); \
                fprintf(stderr, "  Line: %d\n", __LINE__); \
                fprintf(stderr, "  Function: %s\n", __func__); \
                __builtin_trap();  /* Triggers debugger */ \
            } \
        } while(0)
#endif

#endif // DEBUG_H