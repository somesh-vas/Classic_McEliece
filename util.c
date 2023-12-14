/*
  This file is for loading/storing data in a little-endian fashion
*/

#include "util.h"

#include "params.h"
#include <stdio.h>

void store_gf(unsigned char *dest, gf a)
{
	dest[0] = a & 0xFF;
	dest[1] = a >> 8;
}

uint16_t load_gf(const unsigned char *src)
{	
	
	uint16_t a; // 2 byte 

	a = src[1]; 
	a <<= 8; // Left-shift by 8 bits (one byte)
	a |= src[0]; 

	return a & GFMASK; 
	/*
	GFMASK is defined as 4095, which has 12 bits set to 1), and it discards any higher-order bits in a beyond the 12th bit
	ex:
	a        : 0000 0110 1111 0111
	GFMASK   : 0000 1111 1111 1111
	-------------------------------
	result   : 0000 0110 1111 0111

		
	*/
}

uint32_t load4(const unsigned char * in)
{
	int i;
	uint32_t ret = in[3];

	for (i = 2; i >= 0; i--)
	{
		ret <<= 8;
		ret |= in[i];
	}

	return ret;
}

void store8(unsigned char *out, uint64_t in)
{
	out[0] = (in >> 0x00) & 0xFF;
	out[1] = (in >> 0x08) & 0xFF;
	out[2] = (in >> 0x10) & 0xFF;
	out[3] = (in >> 0x18) & 0xFF;
	out[4] = (in >> 0x20) & 0xFF;
	out[5] = (in >> 0x28) & 0xFF;
	out[6] = (in >> 0x30) & 0xFF;
	out[7] = (in >> 0x38) & 0xFF;
}

uint64_t load8(const unsigned char * in)
{
	int i;
	uint64_t ret = in[7];
	
	for (i = 6; i >= 0; i--)
	{
		ret <<= 8;
		ret |= in[i];
	}
	/*	Initially, ret is set to the last byte (in[7]), which is 0x08.
		In the loop, the code shifts ret left by 8 bits in each iteration 
		and then performs a bitwise OR with the current byte (in[i]). 
		This process effectively combines the bytes to form the 64-bit integer.
	*/
	return ret;
}

gf bitrev(gf a)
{
	a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8); // Swap Adjacent Bytes:
	a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4); // Swap Nibbles within Bytes:
	a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2); // Swap Pairs of Bits within Nibbles:
	a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1); // Swap Individual Bits within Pairs:
	
	return a >> 4; // Right Shift by 4 to Discard Lower 4 Bits:
}

