CFLAGS=`root-config --cflags`
LDFLAGS=`root-config --ldflags --glibs` -lRooFit -lRooFitCore -lMinuit

#temp : temp.cpp RooGausExpDict.cxx RooGausExpDict.cxx libRooGausExp.so
#g++ -o temp temp.cpp RooGausExpDict.cxx RooGausExpDict.cxx $(CFLAGS) $(LDFLAGS)

libRooGausExp.so : libRooGausExp.so.1.0
	ln -sf libRooGausExp.so.1.0 libRooGausExp.so

libRooGausExp.so.1.0 : RooGausExp.o RooGausExpDict.o
	gcc -shared -Wl,-soname,libRooGausExp.so.1 -o libRooGausExp.so.1.0 RooGausExp.o RooGausExpDict.o

RooGausExpDict.o : RooGausExpDict.cxx
	g++ -c RooGausExpDict.cxx -fPIC $(CFLAGS) $(LDFLAGS)

RooGausExpDict.cxx : RooGausExp.h LinkDef.h
	rootcint -f RooGausExpDict.cxx -c RooGausExp.h LinkDef.h

RooGausExp.o : RooGausExp.cxx RooGausExp.h
	g++ -c RooGausExp.cxx -fPIC $(CFLAGS) $(LDFLAGS)
