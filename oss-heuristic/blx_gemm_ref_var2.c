
/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

#include "blis.h"
#include "blix.h"

#include <nanos6/debug.h>
#define FUNCPTR_T gemmsup_fp
int OSS_TASKS;
int OSS_TASKSCPU;
int OSS_CORES=64;
int OSS_BSX;
int OSS_BSY;
int oss_5th;
int oss_3rd;

typedef struct mem_ompss{
	struct mem_s mem_a_ompss;
}mem_ompss_t;


static bool parse_unsigned_long_oss(const char *name, int *pvalue)
{
	char *env, *end;
	int value;

	env = getenv(name);
	if(env == NULL)
		return 1;
	
	while(isspace((unsigned char) *env))
		++env;
	if(*env == '\0')
		return 1;
	
	errno = 0;
	value = strtoul(env, &end, 10);

	while(isspace ((unsigned char) *end))
		++end;
	if(*end != '\0')
		return 1;
	*pvalue = value;
	return 0;
}


typedef void (*FUNCPTR_T)
     (
       bool             packa,
       bool             packb,
       conj_t           conja,
       conj_t           conjb,
       dim_t            m,
       dim_t            n,
       dim_t            k,
       void*   restrict alpha,
       void*   restrict a, inc_t rs_a, inc_t cs_a,
       void*   restrict b, inc_t rs_b, inc_t cs_b,
       void*   restrict beta,
       void*   restrict c, inc_t rs_c, inc_t cs_c,
       stor3_t          eff_id,
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread
     );

//
// -- var2 ---------------------------------------------------------------------

     static FUNCPTR_T GENARRAY(ftypes_var2,gemm_ref_var2);

void blx_gemm_ref_var2
     (
       trans_t trans,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       stor3_t eff_id,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	/*oss: get environment variables to define the parallelization */
	
	if(parse_unsigned_long_oss("BLIS_OSS_TASKS", &OSS_TASKS) == 1){
		OSS_TASKS = 64; /* change here accordingly to the number of cores available in the architecture */
	}
	
        const num_t     dt      = bli_obj_dt( c );

        /* Define that both A and B matrices will be packed into panels */
        bli_rntm_set_pack_a( TRUE, rntm );
        bli_rntm_set_pack_b( TRUE, rntm );

        /* Get the status (TRUE or FALSE) of packing A and B */
        const bool      packa   = bli_rntm_pack_a( rntm );
        const bool      packb   = bli_rntm_pack_b( rntm );

        const conj_t    conja   = bli_obj_conj_status( a );
        const conj_t    conjb   = bli_obj_conj_status( b );

        /* Get dimmensions m, n, and k */
        const dim_t     m       = bli_obj_length( c );
        const dim_t     n       = bli_obj_width( c );
              dim_t     k;

        /* Acquire buffer at object's submatrices offset A and B*/
        void* restrict  buf_a   = bli_obj_buffer_at_off( a );
        void* restrict  buf_b   = bli_obj_buffer_at_off( b );

        inc_t   rs_a;
        inc_t   cs_a;
        inc_t   rs_b;
        inc_t   cs_b;

        if ( bli_obj_has_notrans( a ) )
        {
                k       = bli_obj_width( a );
                rs_a    = bli_obj_row_stride( a );
                cs_a    = bli_obj_col_stride( a );
        }
        else /*if (bli_obj_has_trans( a ) ) */
 {
                //Assign the variables with an implicit transposition.
                k       = bli_obj_length( a );
                rs_a    = bli_obj_col_stride( a );
                cs_a    = bli_obj_row_stride( a );
        }
        if ( bli_obj_has_notrans( b ) )
        {
                rs_b    = bli_obj_row_stride( b );
                cs_b    = bli_obj_col_stride( b );
        }
        else /* if (bli_obj_has_trans( b ) ) */
        {
                //Assign the variables with an implicit transposition.
                rs_b    = bli_obj_col_stride( b );
                cs_b    = bli_obj_row_stride( b );
        }

        void* restrict buf_c     = bli_obj_buffer_at_off( c );
        const inc_t     rs_c    = bli_obj_row_stride( c );
        const inc_t     cs_c    = bli_obj_col_stride( c );

        void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

        // Index into the type combination array to extract the correct function pointer
        //FUNCPTR_T f = ftypes_var1n[dt];
        FUNCPTR_T f = ftypes_var2[dt];

#if 1
        // Optimize some storage/packing cases by transforming them into others.
        // These optimizations are expressed by changing trans and/or eff_id
        bli_gemmsup_ref_var1n2m_opt_cases( dt, &trans, packa, packb, &eff_id, cntx);
#endif
	if ( bli_is_notrans( trans ) )
        {
                // Invoke the function.
                f
                (
                  packa,
                  packb,
                  conja,
                  conjb,
                  m,
                  n,
                  k,
                  buf_alpha,
                  buf_a, rs_a, cs_a,
                  buf_b, rs_b, cs_b,
                  buf_beta,
                  buf_c, rs_c, cs_c,
                  eff_id,
                  cntx,
                  rntm,
                  thread
                );
        }
        else
        {
                // Invoke the function (transposing the operation)
 f
                (
                  packb,
                  packa,
                  conjb,        // swap the conj values.
                  conja,
                  n,            // swap the m and n dimensions.
                  m,
                  k,
                  buf_alpha,
                  buf_b, cs_b, rs_b, // swap the positions of A and B.
                  buf_a, cs_a, rs_a, // swap the strides of A and B.
                  buf_beta,
                  buf_c, cs_c, rs_c, // swap the strides of C.
                  bli_stor3_trans( eff_id ), //transpose the stor3_t id
                  cntx,
                  rntm,
                  thread
                );

        }
}

#undef GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
        ( \
          bool          packa, \
          bool          packb, \
          conj_t        conja, \
          conj_t        conjb, \
          dim_t         m, \
          dim_t         n, \
          dim_t         k, \
          void*         restrict alpha, \
          void*         restrict a, inc_t rs_a, inc_t cs_a, \
          void*         restrict b, inc_t rs_b, inc_t cs_b, \
          void*         restrict beta, \
          void*         restrict c, inc_t rs_c, inc_t cs_c, \
          stor3_t       stor_id, \
          cntx_t*       restrict cntx, \
          rntm_t*       restrict rntm, \
          thrinfo_t*    restrict thread \
          ) \
{ \
\
        const num_t dt = PASTEMAC(ch,type); \
\
        /* If m or n is zero, return immediately. */ \
        if ( bli_zero_dim2( m, n ) ) return; \
\
        /* If k < 1 or alpha is zeor, scale by beta and return. */ \
        if ( k < 1 || PASTEMAC(ch,eq0)( *(( ctype* )alpha) ) ) \
        { \
                        PASTEMAC(ch,scalm) \
                        ( \
                          BLIS_NO_CONJUGATE, \
                          0, \
                          BLIS_NONUNIT_DIAG, \
BLIS_DENSE, \
                          m, n, \
                          beta, \
                          c, rs_c, cs_c \
                        ); \
                return; \
        } \
        /* Query the context for various blocksizes. */ \
        const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
        const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
        const dim_t NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
        const dim_t MC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx );  \
        const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
 	dim_t KC; \
        if ( packa && packb ) \
        { \
                KC = KC0; \
        } \
        else if ( packb ) \
        { \
                if ( stor_id == BLIS_RRR || stor_id == BLIS_CCC )KC = KC0; \
                else if ( stor_id == BLIS_RRC || stor_id == BLIS_CRC ) KC = KC0; \
                else if ( stor_id == BLIS_RCR || stor_id == BLIS_CCR ) KC = (( KC0 / 4 ) / 4 ) * 4; \
                else KC = KC0; \
        } \
        else if ( packa ) \
        { \
                if ( stor_id == BLIS_RRR || stor_id == BLIS_CCC ) KC = (( KC0 / 2) / 2 ) * 2; \
                else if ( stor_id == BLIS_RRC || stor_id == BLIS_CRC ) KC = KC0; \
                else if ( stor_id == BLIS_RCR || stor_id == BLIS_CCR ) KC = (( KC0 / 4 ) / 4 ) * 4; \
                else KC = KC0; \
        } \
        else \
        { \
                if ( stor_id == BLIS_RRR || stor_id == BLIS_CCC ) KC = KC0; \
                else if ( stor_id == BLIS_RRC || stor_id == BLIS_CRC ) KC = KC0; \
                else if ( m <= MR && n <= NR ) KC = KC0; \
                else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2; \
                else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
                else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4; \
                else KC = (( KC0 / 5 ) / 4 ) * 4; \
        } \
\
        /* Query the maximum blocksize for NR, which implies a maximum blocksize extension for the final iteration */ \
\
        const dim_t NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx ); \
        const dim_t NRE = NRM - NR; \
\
        /* Compute partitioning step values for each matrix of each loop. */ \
        const inc_t jcstep_c = cs_c; \
        const inc_t jcstep_b = cs_b; \
\
        const inc_t pcstep_a = cs_a; \
        const inc_t pcstep_b = rs_b; \
\
        const inc_t icstep_c = rs_c; \
        const inc_t icstep_a = rs_a; \
\
        const inc_t jrstep_c = cs_c * NR; \
 	const inc_t irstep_c = rs_c * MR; \
\
        /* Query the context for the gemm microkernel address and cast it to its function pointer type. */ \
        PASTECH(ch, gemm_ukr_ft) \
                gemm_ukr = bli_cntx_get_l3_vir_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
\
        ctype           ct[ BLIS_STACK_BUF_MAX_SIZE \
                                / sizeof( ctype )] \
                                __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const bool      col_pref        = bli_cntx_l3_vir_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx ); \
        const inc_t     rs_ct   = ( col_pref ? 1 : NR ); \
        const inc_t     cs_ct   = ( col_pref ? MR : 1 ); \
        ctype* restrict zero = PASTEMAC(ch,0); \
        ctype* restrict a_00    = a; \
        ctype* restrict b_00    = b; \
        ctype* restrict c_00    = c; \
        ctype* restrict alpha_cast = alpha; \
        ctype* restrict beta_cast = beta; \
\
        /* Make local copies of beta and one scalars to prevent any unecessary sharing of cache lines between the cores' caches. */ \
        ctype           beta_local = *beta_cast; \
        ctype           one_local = *PASTEMAC(ch,1); \
\
        auxinfo_t       aux; \
\
        /* Determine whether we are using more than one thread. */ \
        /* Create and initialize the thread info */ \
        array_t* restrict array = bli_sba_checkout_array( 1 ); \
        bli_sba_rntm_set_pool( 0, array, rntm ); \
        /*bli_membrk_rntm_set_membrk( rntm ); */\
        bli_pba_rntm_set_pba( rntm ); \
        thrinfo_t* new_thread = bli_sba_acquire( rntm, sizeof( thrinfo_t ) ); \
	bli_thrinfo_init_single(new_thread); \
\
/*BLIS-OSS: DEFINING THE NUMBER OF TASKS */ \
	dim_t OSS_NC = NC, OSS_MC = MC, OSS_KC = KC; \
	OSS_MC = MC; \
	int OSS_L3 = 16; \
	if(n < 512){ \
		OSS_L3 = 2; \
	} else if(n >= 512 && n < 1024){ \
		OSS_L3 = 4; \
	}else if (n >= 1024 && n < 4096){ \
		OSS_L3 = 8; \
	} else{ \
		OSS_L3 = 16; \
	} \
	if(m <= 512){ \
		if(m <= 256){ \
			OSS_MC = 36;  \
			OSS_BSY = 4; \
			OSS_BSX = 1; \
		} else{ \
			OSS_MC = 36; \
			OSS_BSY = 1; \
			OSS_BSX = 1; \
		} \
	} \
	OSS_NC = n/OSS_L3; \
	OSS_KC = KC; \
	int OSS_BEST_TASKS = m/MC, OSS_NUM_TASKS_AUX=0, OSS_SOBRA_AUX=0, OSS_TOTAL_NUM_TASKS_AUX=0; \
	if(m%MC != 0){ \
		OSS_BEST_TASKS++; \
	} \
	int OSS_TOTAL_BEST_TASKS = OSS_BEST_TASKS * OSS_L3; \
	int OSS_TOTAL_BEST_SOBRA = OSS_TOTAL_BEST_TASKS % OSS_TASKS; \
	int OSS_BSY_BEST=0; \
	if(m > 512){ \
		OSS_MC = MC; \
		OSS_BSY = 1; \
		OSS_BSX = 1; \
		for(OSS_BSY = 1; OSS_BSY <= 32; OSS_BSY = OSS_BSY*2){ \
				OSS_NUM_TASKS_AUX = (m / (OSS_MC * OSS_BSY)); \
				if( m%(OSS_MC*OSS_BSY) != 0) {\
					OSS_NUM_TASKS_AUX++; \
				} \
				OSS_TOTAL_NUM_TASKS_AUX = OSS_NUM_TASKS_AUX * OSS_L3; \
				OSS_SOBRA_AUX = OSS_TOTAL_NUM_TASKS_AUX % OSS_TASKS; \
				if(OSS_SOBRA_AUX > OSS_TASKS/2){ \
					OSS_SOBRA_AUX = OSS_TASKS - OSS_SOBRA_AUX; \
				} \
				if((OSS_SOBRA_AUX <= OSS_TOTAL_BEST_SOBRA)){ \
					OSS_TOTAL_BEST_TASKS = OSS_TOTAL_NUM_TASKS_AUX; \
					OSS_TOTAL_BEST_SOBRA = OSS_SOBRA_AUX; \
					OSS_BSY_BEST = OSS_BSY; \
				} \
		} \
		if(OSS_BSY_BEST > 0){ \
			OSS_BSY = OSS_BSY_BEST; \
		} else { \
			OSS_BSY = 1; \
		} \
	} \
	\
	/* Compute the JC loop thread range for the current thread. */ \
        dim_t jc_start = 0, jc_end = n; \
        const dim_t n_local = jc_end - jc_start; \
        /* Compute number of primary and leftover components of the JC loop. */ \
	const dim_t jc_left = n_local % OSS_NC; \
        \
	\
	\
	/* Allocating mem_a pannel */ \
        mem_ompss_t mem_new_a[OSS_CORES]; \
	const dim_t m_pack = ( OSS_MC / MR + ( OSS_MC % MR ? 1 : 0 ) ) * MR; \
	const dim_t k_pack = KC; \
	siz_t size_needed = sizeof(ctype) * m_pack * k_pack; \
	for(int t=0; t < OSS_CORES; t++){ \
		/*bli_membrk_acquire_m(rntm, size_needed, BLIS_BUFFER_FOR_A_BLOCK, &mem_new_a[t].mem_a_ompss); */\
		bli_pba_acquire_m(rntm, size_needed, BLIS_BUFFER_FOR_A_BLOCK, &mem_new_a[t].mem_a_ompss); \
	} \
\
\
\
        /* Loop over the m dimension (NC rows/columns at a time). */ \
	_Pragma("oss taskloop for grainsize(1) chunksize(OSS_BSX)") \
        for( dim_t jj = jc_start; jj < jc_end; jj += OSS_NC ) \
	{ \
		const dim_t nc_cur = ( OSS_NC <= jc_end - jj ? OSS_NC : jc_left ); \
\
       		mem_t mem_b = BLIS_MEM_INITIALIZER;  \
\
                ctype* restrict b_jc = b_00 + jj * jcstep_b; \
                ctype* c_jc = c_00 + jj * jcstep_c; \
\
                /* compute the PC loop thread range for the current thread. */ \
                const dim_t pc_start = 0, pc_end = k; \
                const dim_t k_local = k; \
\
                /* compute the number of primary and leftover components of the PC loop. */ \
                const dim_t pc_left = k_local % OSS_KC; \
		\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		int id_k = 0; \
                for ( dim_t pp = pc_start; pp < pc_end; pp += OSS_KC ) \
                { \
                        /* Calculate the thread's current PC block dimension. */ \
                        const dim_t kc_cur = ( OSS_KC <= pc_end - pp ? OSS_KC : pc_left ); \
\
        		/* mem_t mem_b = BLIS_MEM_INITIALIZER; */\
                        ctype* restrict a_pc = a_00 + pp * pcstep_a; \
                        ctype* restrict b_pc = b_jc + pp * pcstep_b; \
\
                        /* only apply beta to the first iteration of the pc loop. */ \
                        ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
                        ctype* b_use; \
			ctype* restrict b_pc_use; \
                        inc_t rs_b_use, cs_b_use, ps_b_use; \
\
                        /* Determine the packing buffer and related parameters for matrix
                           B. (If B will not be packed, then a_use will be set to point to
                           b and the _b_use strides will be set accordingly.) Then call
                           the packm sup variant chooser, which will call the appropriate
                           implementation based on the schema deduced from the stor_id. */ \
			PASTEMAC(ch, packm_sup_b) \
                        ( \
                          packb,                                /* This algorithm packs matrix B */ \
                          BLIS_BUFFER_FOR_B_PANEL,              /* to a "panel of B" */ \
                          stor_id, \
                          BLIS_NO_TRANSPOSE, \
                          OSS_KC, OSS_NC,                               /* This panel of B is (at most) KC x NC */ \
                          kc_cur, nc_cur, NR, \
                          &one_local, \
                          b_pc, rs_b, cs_b, \
                          &b_use, &rs_b_use, &cs_b_use, \
                                                & ps_b_use, \
                          cntx, \
                          rntm, \
                          &mem_b, \
                          new_thread \
                        ); \
\
			id_k++; \
                        /* Alias a_use so that it's clear this this is our current block of matrix A. */ \
                        b_pc_use = b_use; \
\
                        bli_auxinfo_set_ps_b( ps_b_use, &aux); \
\
                        /* Compute the IC loop thread range for the current thread. */ \
			dim_t ic_start = 0, ic_end = m; \
                        const dim_t m_local = ic_end - ic_start; \
\
                        /* Compute number of primary and leftover components of the IC loop. */ \
                        const dim_t ic_left = m_local % OSS_MC; \
\
                        /* Loop over the m dimension ( MC rows at a time ). */ \
                         _Pragma("oss taskloop for inout(c_jc[ii*icstep_c]) grainsize(1) chunksize(OSS_BSY)") \
			for ( dim_t ii = ic_start; ii < ic_end; ii += OSS_MC ) \
                        { \
				unsigned int OSS_MY_CPU = nanos6_get_current_virtual_cpu(); \
				PASTEMAC(ch,set0s_mxn)(MR, NR, ct, rs_ct, cs_ct ); \
                                /* Calculate the thread's current IC block dimension. */ \
                                const dim_t mc_cur = ( OSS_MC <= ic_end - ii ? OSS_MC : ic_left ); \
\
                                ctype* restrict a_ic = a_pc + ii * icstep_a; \
                                ctype* c_ic = c_jc + ii * icstep_c; \
\
                                ctype* a_use; \
                                inc_t rs_a_use, cs_a_use, ps_a_use; \
\
                                /* Set the bszid_t array and thrinfo_t pointer based on whether
                                   we will be packing B. If we won't be packing A, we alias to
                                   the _ic variables so that code further down can unconditionally
                                   reference the _pa variables. Note that *if* we will be packing
                                   A, the thrinfo_t node will have already been created by a
                                   previous call to bli_thrinfo_grow(), since bszid values of
                                   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
                                   bszid that is a normal bszid_t value). */ \
\
                                /* Determine the packing buffer and related parameters for matrix
                                   A. (If A will not be packed, then a_use will be set to point to
                                   a and the _a_use strides will be set accordingly.) Then call
                                   the packm sup variant chooser, which will call the appropriate
                                   implementation based on the schema deduced from the stor_id. */ \
                                PASTEMAC(ch,packm_sup_a) \
                                ( \
                                  packa,                        /* This algorithm packs matrix A to */ \
                                  BLIS_BUFFER_FOR_A_BLOCK,      /* a "block of A" */ \
                                  stor_id, \
                                  BLIS_NO_TRANSPOSE, \
                                  OSS_MC, OSS_KC,                       /* This block of A is (at most) MC x KC. */ \
                                  mc_cur, kc_cur, MR, \
                                  &one_local, \
                                  a_ic, rs_a, cs_a, \
                                  &a_use, &rs_a_use, &cs_a_use, \
                                  &ps_a_use, \
                                  cntx, \
                                  rntm, \
                                  &mem_new_a[OSS_MY_CPU].mem_a_ompss,  \
                                  new_thread \
                                ); \
\
                                /* Alias a_use so that it's clear this is our current block of matrix A. */ \
                                ctype* restrict a_ic_use = a_use; \
\
                                /* Embed the panel stride of A within the auxinfo_t object. */ \
                                bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
\
                                /* Compute number of primary and leftover components of the JR loop. */ \
                                dim_t jr_iter = ( nc_cur + NR -1 ) / NR; \
                                dim_t jr_left = nc_cur % NR; \
\
                                /* An optimization: allow the last jr iteration to contain up to NRE
                                   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
                                   these cases.) Note that this prevents us from declaring jr_iter and
                                   jr_left as const. NOTE: We forgo this optimization when packing B
                                   since packing an extended edge case is not yet supported. */ \
                                if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE ) \
                                { \
                                	jr_iter--; jr_left += NR; \
                                } \
\
                                /* first loop around micro-kernel*/ \
                                const dim_t ir_iter = ( mc_cur + MR -1 ) / MR; \
                                const dim_t ir_left = mc_cur % MR; \
                                /* Compute the JR loop thread range for the current thread. */ \
                                dim_t jr_start = 0, jr_end = jr_iter, jr_inc=1; \
\
                                /* Loop over the n dimension (NR columns at a time */ \
                                for( dim_t j = jr_start; j < jr_end; j += jr_inc ) \
                                { \
                                        ctype* restrict b2; \
                                        const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
\
                                        ctype* restrict b_jr = b_pc_use + j * ps_b_use; \
                                        ctype* c_jr = c_ic + j * jrstep_c;  \
                                	dim_t ir_start = 0, ir_end = ir_iter, ir_inc=1; \
\
                                        b2 = b_jr; \
                                        /* Loop over the m dimension (MR rows at a time). */ \
                                        for ( dim_t i = ir_start; i < ir_end; i += ir_inc) \
                                        { \
\
                                                const dim_t mr_cur = ( bli_is_not_edge_f( i, ir_iter, ir_left ) ? MR : ir_left ); \
                                                ctype* restrict a_ir = a_ic_use + i * ps_a_use; \
                                                ctype*  c_ir = c_jr + i * irstep_c; \
                                                ctype* restrict a2; \
\
                                                /* Compute the addresses of the next panels of A and B */ \
                                                a2 = bli_gemm_get_next_a_upanel( a_ir, ps_a_use, ir_inc); \
                                                if( bli_is_last_iter( i, ir_iter, 0, 1) ) \
                                                { \
                                                        a2 = a_ic_use; \
                                                        b2 = bli_gemm_get_next_b_upanel( b_jr, ps_b_use, jr_inc); \
                                                        if( bli_is_last_iter( j, jr_end, 0, 1) ) \
                                                                b2 = b_pc_use; \
                                                } \
\
                                                /* Save addresses of next panels of A and B to the auxinfo_t object */ \
                                                bli_auxinfo_set_next_a( a2, &aux ); \
                                                bli_auxinfo_set_next_b( b2, &aux );  \
\
                                                /* Handle interior and edge cases separately. */ \
                                                if( mr_cur == MR && nr_cur == NR) \
                                                { \
                                                        gemm_ukr \
                                                        ( \
                                                          kc_cur, \
                                                          alpha_cast, \
                                                          a_ir, \
                                                          b_jr, \
                                                          beta_use, \
                                                          c_ir, rs_c, cs_c, \
                                                          &aux, \
                                                          cntx \
                                                        ); \
                                                } \
                                                else \
                                                { \
                                                        gemm_ukr \
                                                        ( \
                                                          kc_cur, \
                                                          alpha_cast, \
                                                          a_ir, \
                                                          b_jr, \
                                                          zero, \
                                                          ct, rs_ct, cs_ct, \
                                                          &aux, \
                                                          cntx \
                                                        ); \
                                                       PASTEMAC(ch,xpbys_mxn)( mr_cur, nr_cur, ct, rs_ct, cs_ct, beta_cast, c_ir, rs_c, cs_c );   \
                                                } \
					} \
				} \
			} \
                } \
/*		for(int t = 0; t < size_mem_b; t++){ */\
		PASTEMAC(ch,packm_sup_finalize_mem_b) \
       		( \
          		packb, \
		        rntm,  \
		        &mem_b, \
		        /*&mem_new_b[t].mem_a_ompss, */\
			new_thread \
        	); \
		/*}*/ \
        } \
	_Pragma("oss taskwait") \
        /* Release any memory that was acquired for packing matrices A and B. */ \
        for(int t_omp = 0; t_omp < OSS_CORES; t_omp++){ \
		PASTEMAC(ch,packm_sup_finalize_mem_a) \
	        ( \
	        	packa, \
	        	rntm, \
	        	&mem_new_a[t_omp].mem_a_ompss, \
	        	new_thread \
	        ); \
	} \
}

INSERT_GENTFUNC_BASIC0( gemm_ref_var2 )

