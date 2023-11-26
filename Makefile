kat_kem.rsp: kat
	./run

kat: Makefile nist/kat_kem.c benes.c bm.c  decrypt.c gf.c  root.c synd.c transpose.c util.c    
	./build
clean:
	rm -rf *.int, kat