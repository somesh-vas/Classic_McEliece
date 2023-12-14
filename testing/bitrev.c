#include <stdio.h>

typedef unsigned short gf; // Assuming gf is a 16-bit unsigned type

gf bitrev(gf a) {
    a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8);
    a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4);
    a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2);
    a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1);
    return a >> 4;
}

int main() {
    FILE *outputFile = fopen("bitrev_output.txt", "w");
    if (outputFile == NULL) {
        fprintf(stderr, "Error opening the output file.\n");
        return 1;
    }

    for (int a = 0; a < 4096; a++) {
        gf result = bitrev((gf)a);
        fprintf(outputFile, "a: %d, bitrev(a): %u\n", a, result);
    }

    fclose(outputFile);

    return 0;
}
