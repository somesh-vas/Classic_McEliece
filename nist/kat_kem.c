
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "crypto_kem.h"
#include "decrypt.h"
#include "params.h"

#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_CRYPTO_FAILURE  -4


int
main()
{
    
    int tv; //test_vector
    
    // unsigned char *ss1 = 0;
    // unsigned char *ss2 = 0;  
    unsigned char *sk1 = 0;
    unsigned char *ct1 = 0;      

    
    //  char sessionkeys[KATNUM][65];
    char ciphertexts[KATNUM][193];
     char secretkeys[KATNUM][12985];
    //  int random_error[KATNUM];
    
    //FILE *file = fopen("ss1.txt", "r");
    FILE *file1 = fopen("ct1.txt", "r");
    FILE *file2 = fopen("sk1.txt", "r");
    // FILE *file = fopen()
    FILE *file = fopen("ee.txt", "r");
    
    int error_encrypt_positions[KATNUM][64];
    //int error_decrypt_positions[64];

    if ( file == NULL || file1 == NULL || file2 == NULL) {
        perror("Error opening file");
        return 1;
    }    
    

    for (int i = 0; i < KATNUM; i++) {
        for (int j = 0; j < 64; j++) {
            if (fscanf(file, "%d", &error_encrypt_positions[i][j]) != 1) {
                printf("Error reading from file.\n");
                fclose(file);
                return 1;
            }
        }
    }

    // Read and store the content in the sessionkey array
    for (int i = 0; i < KATNUM; ++i) {
        if (fscanf(file2, "%12984s", secretkeys[i]) != 1) {
            fprintf(stderr, "Error reading from file");
            fclose(file2);
            return 1;
        }
    }
    // Read and store the content in the sessionkey array
    // for (int i = 0; i < KATNUM; ++i) {
    //     if (fscanf(file, "%64s", sessionkeys[i]) != 1) {
    //         fprintf(stderr, "Error reading from file");
    //         fclose(file);
    //         return 1;
    //     }
    // }

    for (int i = 0; i < KATNUM; ++i) {
        if (fscanf(file1, "%192s", ciphertexts[i]) != 1) {
            fprintf(stderr, "Error reading from file");
            fclose(file1);
            return 1;
        }
    }

    //fclose(file); 

    

    for (tv=0; tv<KATNUM; tv++) {
        
        // if (!ss1) ss1 = malloc(crypto_kem_BYTES);
        // if (!ss1) abort();
        // if (!ss2) ss2 = malloc(crypto_kem_BYTES);
        // if (!ss2) abort();       
        if (!sk1) sk1 = malloc(crypto_kem_SECRETKEYBYTES);
        if (!sk1) abort();
        if (!ct1) ct1 = malloc(crypto_kem_CIPHERTEXTBYTES);        
        if (!ct1) abort();
        

    
    for (int i = 0; i < crypto_kem_CIPHERTEXTBYTES; i++) {
        sscanf(ciphertexts[tv] + 2 * i, "%2hhX", &ct1[i]);
    }
    for (int i = 0; i < crypto_kem_SECRETKEYBYTES; i++) {
        sscanf(secretkeys[tv] + 2 * i, "%2hhX", &sk1[i]);
    }
    // for (int i = 0; i < crypto_kem_BYTES; i++) {
    //     sscanf(sessionkeys[tv] + 2 * i, "%2hhX", &ss2[i]);
    // }    
             
        unsigned char e[ SYS_N / 8];
        // sk1 + 40 skips the random string s bits
        #ifdef KAT
        {
            printf("Values from sk to sk + 40:\n");
    for (int i = 0; i < 41; i++) {
        printf("%02X ", sk1[i]);
    }
    printf("\n");
        }
        #endif
        decrypt(e, sk1 + 40, ct1);
        int k;
        int count = 0;
        int flag[64] = {0};
        for (k = 0; k < SYS_N; ++k) {
            
        if (e[k / 8] & (1 << (k & 7))) {
            //error_decrypt_positions[k];
            if(error_encrypt_positions[tv][count] != k){
                flag[count] = k;
                break;
                
            }
        
            }
            count ++;
        }
        #ifdef KAT
        {  
            // printf("comparing error positions");
        //     for(int i = 0; i < 64; i++){
        //         printf("%d ",flag[i]);
        //     }
        //     printf("\n");
        }
        #endif

#ifdef KAT
  {
//   int k; //sum = 0;
// printf("decrypt e: positions");
// for (k = 0; k < SYS_N; ++k) {
//     if (e[k / 8] & (1 << (k & 7))) {
//         printf(" %d", k);
//         //sum += k;
//     }
// }
// printf("\n");

// printf("Sum of positions: %d\n", sum);

  }
#endif
        
    }

    return KAT_SUCCESS;
}

