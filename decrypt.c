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
	// g = 06F70E6E03510E1600760EFE00030FC00A67031A029A0B7B0733024D09810C4F0BDD0DD4009A0190092904AD033800B000940FC301DB04F40568099F087E0FA2068F0DB008D407F80061086C053
	//80F8A005B0F940A3A058102C50DE40DDF00680D8E0DBA0855069C09E5084905E107B6092C049901E70F980A6C0DA506900D510001
	/*
	g(x) = 1 + 3409x^63 + 1680x^62 + 3493x^61 + 2668x^60 + 3992x^59 + 487x^58 + 1177x^57 + 2348x^56 + 1974x^55 + 
	1505x^54 + 2121x^53 + 2533x^52 + 1692x^51 + 2133x^50 + 3514x^49 + 3470x^48 + 104x^47 + 3551x^46 + 3556x^45 + 
	709x^44 + 1409x^43 + 2618x^42 + 3988x^41 + 91x^40 + 3978x^39 + 1336x^38 + 2156x^37 + 97x^36 + 2040x^35 + 2260x^34 + 
	3504x^33 + 1679x^32 + 4002x^31 + 2174x^30 + 2463x^29 + 1384x^28 + 1268x^27 + 475x^26 + 4035x^25 + 148x^24 + 176x^23 + 
	824x^22 + 1197x^21 + 2345x^20 + 400x^19 + 154x^18 + 3540x^17 + 3037x^16 + 3151x^15 + 2433x^14 + 589x^13 + 1843x^12 + 
	2939x^11 + 666x^10 + 794x^9 + 2663x^8 + 4032x^7 + 3x^6 + 3838x^5 + 118x^4 + 3606x^3 + 849x^2 + 3694x^1 + 1783x^0

	*/
	#ifdef KAT
	{
	// 	// Print the polynomial expression
    // printf("g(x) = ");
    // for (int i = SYS_T; i >= 0; i--) {
    //     if (i == SYS_T) {
    //         printf("%d", g[i]); // Print the constant term
    //     } else {
    //         printf(" + %dx^%d", g[i], i);
    //     }
    // }
    // printf("\n");
	}
	#endif

	
	support_gen(L, sk); // (α0, α1, α2, . . .αn) from secret key having  condition bits // support_gen function is in benes.c
	//sk will have condition bits to generate support L
	// support gen employs apply_benes using r sequence of bits to be permuted
	#ifdef KAT
	{
	printf("1111111111***********\n");
	for(int i =0; i < SYS_N; i++) {
		 printf("%d = %02X\n ",i, L[i]);  // Modify this based on the representation of the gf type
    }
    printf("\n");
	}
	#endif
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

