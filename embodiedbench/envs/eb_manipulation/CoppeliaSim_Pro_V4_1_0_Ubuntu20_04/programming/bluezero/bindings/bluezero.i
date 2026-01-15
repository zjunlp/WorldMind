%module bluezero
%{
#include "bluezero.h"
%}

/*
 * see http://www.swig.org/Doc1.3/Library.html#Library_nn8
 * for:
 *  - string I/O
 *  - returning string buffers
 */
%include "std_string.i"

%include "bluezero.h"

//%apply (char *STRING, int LENGTH) { (char *data, size_t len) };

