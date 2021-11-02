# RooCustomPdfs

First, set up ATLASLocalRootBase and ROOT:

```
setupATLAS
lsetup root
```

Now, make the dictionaries required to include these in `pyroot` scripts:

```
rootcint -f RooDSCBShapeDict.cxx -c RooDSCBShape.h
rootcint -f RooGausExpDict.cxx  -c RooGausExp.h
rootcint -f RooGausDExpDict.cxx -c RooGausDExp.h
```

Then, cross your fingers and `make`:

```
make -f RooDSCBShape.mk
make -f RooGausExp.mk
make -f RooGausDExp.mk
```

You can then include things in your pyroot macros as e.g.

```
import ROOT
ROOT.gSystem.Load('RooCustomPdfs/libRooDSCBShape.so')
from ROOT import RooDSCBShape
```

Submit complaints to @mattleblanc. This was tested with ROOT 6.04 on SL6.

:beers:
