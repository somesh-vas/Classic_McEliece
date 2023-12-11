#ifndef PARAMS_H
#define PARAMS_H

#define GFBITS 12 // Size of each element in the Galois field: 12 bits

#define SYS_N 3488 
#define SYS_T 64

#define COND_BYTES ((1 << (GFBITS-4))*(2*GFBITS - 1))
#define IRR_BYTES (SYS_T * 2)  // Size of irreducible polynomial in bytes

#define PK_NROWS (SYS_T*GFBITS) 
#define PK_NCOLS (SYS_N - PK_NROWS)
#define PK_ROW_BYTES ((PK_NCOLS + 7)/8)

#define SYND_BYTES ((PK_NROWS + 7)/8)

#define GFMASK ((1 << GFBITS) - 1) // Bitmask for the Galois field


#endif

