CFLAGS=`root-config --cflags`
LDFLAGS=`root-config --ldflags --glibs` -lRooFit -lRooFitCore -lMinuit

#temp : temp.cpp RooGausDExpDict.cxx RooGausDExpDict.cxx libRooGausDExp.so
#g++ -o temp temp.cpp RooGausDExpDict.cxx RooGausDExpDict.cxx $(CFLAGS) $(LDFLAGS)

libRooGausDExp.so : libRooGausDExp.so.1.0
	ln -sf libRooGausDExp.so.1.0 libRooGausDExp.so

libRooGausDExp.so.1.0 : RooGausDExp.o RooGausDExpDict.o
	gcc -shared -Wl,-soname,libRooGausDExp.so.1 -o libRooGausDExp.so.1.0 RooGausDExp.o RooGausDExpDict.o

RooGausDExpDict.o : RooGausDExpDict.cxx
	g++ -c RooGausDExpDict.cxx -fPIC $(CFLAGS) $(LDFLAGS)

RooGausDExpDict.cxx : RooGausDExp.h LinkDef.h
	rootcint -f RooGausDExpDict.cxx -c RooGausDExp.h LinkDef.h

RooGausDExp.o : RooGausDExp.cxx RooGausDExp.h
	g++ -c RooGausDExp.cxx -fPIC $(CFLAGS) $(LDFLAGS)
