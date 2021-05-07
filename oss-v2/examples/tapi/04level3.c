/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <stdio.h>
#include "blis.h"

int main( int argc, char** argv )
{
	dim_t m, n, k;
	inc_t rsa, csa;
	inc_t rsb, csb;
	inc_t rsc, csc;

	double* a;
	double* b;
	double* c;
	double  alpha, beta;

	// Initialize some basic constants.
	double zero = 0.0;
	double one  = 1.0;
	double two  = 2.0;


	//
	// This file demonstrates level-3 operations.
	//


	//
	// Example 1: Perform a general matrix-matrix multiply (gemm) operation.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m =  n = k = atoi(argv[1]);
	rsc = 1; csc = m;
	rsa = 1; csa = m;
	rsb = 1; csb = k;
	c = malloc( m * n * sizeof( double ) );
	a = malloc( m * k * sizeof( double ) );
	b = malloc( k * n * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;

	// Initialize the matrix operands.
	bli_drandm( 0, BLIS_DENSE, m, k, a, rsa, csa );
//	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
//               m, k, &one, a, rsa, csa );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               k, n, &one, b, rsb, csb );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, n, &zero, c, rsc, csc );

//	bli_dprintm( "a: randomized", m, k, a, rsa, csa, "%4.1f", "" );
//	bli_dprintm( "b: set to 1.0", k, n, b, rsb, csb, "%4.1f", "" );
//	bli_dprintm( "c: initial value", m, n, c, rsc, csc, "%4.1f", "" );

	// c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
	
	bli_dgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
	           m, n, k, &alpha, a, rsa, csa, b, rsb, csb,
	                     &beta, c, rsc, csc ); 
	
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, n, &zero, c, rsc, csc );

	double in = omp_get_wtime();
	bli_dgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
	           m, n, k, &alpha, a, rsa, csa, b, rsb, csb,
	                     &beta, c, rsc, csc );

	printf("Time = %.8f\n", omp_get_wtime() - in);
//	bli_dprintm( "c: after gemm", m, n, c, rsc, csc, "%4.1f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( c );

	return 0;
}

// -----------------------------------------------------------------------------

