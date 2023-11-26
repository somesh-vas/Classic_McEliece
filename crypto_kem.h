#ifndef crypto_kem_H
#define crypto_kem_H

#include "crypto_kem_mceliece348864.h"


#define crypto_kem_dec crypto_kem_mceliece348864_dec
#define crypto_kem_SECRETKEYBYTES crypto_kem_mceliece348864_SECRETKEYBYTES

#define crypto_kem_CIPHERTEXTBYTES crypto_kem_mceliece348864_CIPHERTEXTBYTES
#define crypto_kem_PRIMITIVE "mceliece348864"

#endif
