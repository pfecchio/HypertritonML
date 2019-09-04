#include <iostream>
#include <vector>

#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

using namespace std;

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/Table2.h"

void GenerateTableFromData() {

  string dataDir  = getenv("HYPERML_DATA_2");
  string tableDir = getenv("HYPERML_TABLES_2");

  string inFileNameQ = "HyperTritonTree_18q.root";
  string inFileArgQ  = dataDir + "/" + inFileNameQ;

  string inFileNameR = "HyperTritonTree_18r.root";
  string inFileArgR  = dataDir + "/" + inFileNameR;

  string outFileName = "DataTable.root";
  string outFileArg  = tableDir + "/" + outFileName;

  TChain inputChain("_custom/fTreeV0");
  inputChain.AddFile(inFileArgQ.data());
  inputChain.AddFile(inFileArgR.data());

  TTreeReader fReader(&inputChain);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderValue<RCollision> RColl            = {fReader, "RCollision"};

  TFile outFile(outFileArg.data(), "RECREATE");
  Table2 tree("DataTable", "Data Table");

  TH1D eventCounter{"EventCounter", ";Centrality (%);Events", 100, 0, 100};

  while (fReader.Next()) {
    eventCounter.Fill(RColl->fCent);

    for (auto &RHyper : RHyperVec)
      tree.Fill(RHyper, *RColl);
  }

  outFile.cd();
  eventCounter.Write();
  tree.Write();
  outFile.Close();

  cout << "\nDerived tree from Data generated!\n" << endl;
}
