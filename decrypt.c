/*
  This file is for Niederreiter decryption
*/

#include <stdio.h>
#include "decrypt.h"

#include "params.h"
#include "benes.h"
#include "util.h"
#include "synd.h"
#include "root.h"
#include "gf.h"
#include "bm.h"

/* Niederreiter decryption with the Berlekamp decoder */
/* intput: sk, secret key */
/*         c, ciphertext */
/* output: e, error vector */
/* return: 0 for success; 1 for failure */
int decrypt(unsigned char *e, const unsigned char *sk, const unsigned char *c)
{
	int i, w = 0; 
	
	unsigned char r[ SYS_N/8 ]; 
	gf g[ SYS_T+1 ]; // goppa polynomial
	gf L[ SYS_N ]; // support
	gf s[ SYS_T*2 ]; // random string s
	gf locator[ SYS_T+1 ]; // error locator 
	gf images[ SYS_N ]; // 
	gf t;

	// r = (c0,0,0,.......,0) -> c0 = mt = 12*64 = 768bit , and remaining n - mt zeroes = 3488 - 768 = 2720bits zeroes = 436 bytes 

	for (i = 0; i < SYND_BYTES; i++)       r[i] = c[i]; // appending c0 to r

	for (i = SYND_BYTES; i < SYS_N/8; i++) r[i] = 0; // appending 2720 zeroes
	// sk values for loading g
	//F7066E0E5103160E7600FE0E0300C00F670A1A039A027B0B33074D0281094F0CDD0BD40D9A0090012909AD043803B0009
	//400C30FDB01F40468059F097E08A20F8F06B00DD408F80761006C0838058A0F5B00940F3A0A8105C502E40DDF0D68008E
	//0DBA0D55089C06E5094908E105B6072C099904E701980F6C0AA50D9006510D
	for (i = 0; i < SYS_T; i++) { g[i] = load_gf(sk); sk += 2; } g[ SYS_T ] = 1; // load goppa polynomial from sk to g[] // 'load_gf' is utility function from util.c
	
	
	support_gen(L, sk); // (α0, α1, α2, . . .αn) from secret key having  condition bits // support_gen function is in benes.c
	//sk will have condition bits to generate support L
	// support gen employs apply_benes using r sequence of bits to be permuted
	/*

	in-place Benes network is a series of 2m − 1 stages of swaps applied to an array of
q = 2^m objects (a0, a1, . . . , aq−1). The first stage conditionally swaps a0 and a1, conditionally
swaps a2 and a3, conditionally swaps a4 and a5, etc., as specified by a sequence of q/2 control
bits (1 meaning swap, 0 meaning leave in place). The second stage conditionally swaps a0
and a2, conditionally swaps a1 and a3, conditionally swaps a4 and a6, etc., as specified by the
next q/2 control bits. This continues through the mth stage, which conditionally swaps a0
and aq/2, conditionally swaps a1 and aq/2+1, etc. The (m+1)st stage is just like the (m−1)st
stage (with new control bits), the (m + 2)nd stage is just like the (m − 2)nd stage, and so
on through the (2m − 1)st stage.
	
	*/
	

	synd(s, g, L, r); // string s having the syndrome of length 2t , g, L, r vector 
	/*
		key equation S(z)σ(z) ≡ w(z) mod g(z) implies
	*/

	bm(locator, s); // string s length 2t
	/*
		the set of error locations B = {i : 1 ≤ i ≤ n and σ(αi) = 0},

	*/

	root(images, locator, L); // 
	/*
	error vector e = (e1, e2, . . . , en) is defined by ei for i ∈ B and zeros elsewhere,
	*/
	
	
	for (i = 0; i < SYS_N/8; i++) 
		e[i] = 0;

	for (i = 0; i < SYS_N; i++)
	{
		t = gf_iszero(images[i]) & 1;

		e[ i/8 ] |= t << (i%8);
		w += t;

	}


return 1;

}

