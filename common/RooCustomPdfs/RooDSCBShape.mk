CFLAGS=`root-config --cflags`
LDFLAGS=`root-config --ldflags --glibs` -lRooFit -lRooFitCore -lMinuit

#temp : temp.cpp RooDSCBShape.cxx RooDSCBShapeDict.cxx libRooDSCBShape.so
#g++ -o temp temp.cpp RooDSCBShape.cxx RooDSCBShapeDict.cxx $(CFLAGS) $(LDFLAGS)

libRooDSCBShape.so : libRooDSCBShape.so.1.0
	ln -sf libRooDSCBShape.so.1.0 libRooDSCBShape.so

libRooDSCBShape.so.1.0 : RooDSCBShape.o RooDSCBShapeDict.o
	gcc -shared -Wl,-soname,libRooDSCBShape.so.1 -o libRooDSCBShape.so.1.0 RooDSCBShape.o RooDSCBShapeDict.o

RooDSCBShapeDict.o : RooDSCBShapeDict.cxx
	g++ -c RooDSCBShapeDict.cxx -fPIC $(CFLAGS) $(LDFLAGS)

RooDSCBShapeDict.cxx : RooDSCBShape.h LinkDef.h
	rootcint -f RooDSCBShapeDict.cxx -c RooDSCBShape.h LinkDef.h

RooDSCBShape.o : RooDSCBShape.cxx RooDSCBShape.h
	g++ -c RooDSCBShape.cxx -fPIC $(CFLAGS) $(LDFLAGS)
