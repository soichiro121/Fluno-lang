# Makefile
.PHONY: all clean

all: Makefile.coq
\t$(MAKE) -f Makefile.coq

Makefile.coq: _CoqProject
\tcoq_makefile -f _CoqProject -o Makefile.coq

clean: Makefile.coq
\t$(MAKE) -f Makefile.coq clean
\trm -f Makefile.coq Makefile.coq.conf
