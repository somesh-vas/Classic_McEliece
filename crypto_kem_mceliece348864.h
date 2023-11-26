#ifndef crypto_kem_mceliece348864_H
#define crypto_kem_mceliece348864_H


#define crypto_kem_mceliece348864_ref_SECRETKEYBYTES 6492
#define crypto_kem_mceliece348864_ref_CIPHERTEXTBYTES 96

#ifdef __cplusplus
extern "C" {
#endif
extern int crypto_kem_mceliece348864_ref_dec(unsigned char *,const unsigned char *,const unsigned char *);
#ifdef __cplusplus
}
#endif


#define crypto_kem_mceliece348864_dec crypto_kem_mceliece348864_ref_dec

#define crypto_kem_mceliece348864_SECRETKEYBYTES crypto_kem_mceliece348864_ref_SECRETKEYBYTES

#define crypto_kem_mceliece348864_CIPHERTEXTBYTES crypto_kem_mceliece348864_ref_CIPHERTEXTBYTES

#endif
