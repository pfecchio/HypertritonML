#include <iostream>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

using namespace std;

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/Table2.h"

void GenerateTableFromData(bool likeSign = false, bool kInt7 = false, string dataDir = "" , string tableDir = "")
{

  cout<<"CIAO"<<endl;
  if (dataDir == "") dataDir = getenv("HYPERML_TREES_2");
  if (tableDir == "") tableDir = getenv("HYPERML_TABLES_2");

  string pass = "3";
  string otf = getenv("HYPERML_OTF");

  string passString = "_pass" + pass + otf + ".root";

  string kintstring = kInt7 ? "KINT7" : "";
  string lsString = likeSign ? "LS" : "";

  string inFileNameQ = "HyperTritonTree_18q";
  string inFileArgQ = dataDir + "/" + inFileNameQ + lsString + passString;

  string inFileNameR = "HyperTritonTree_18r";
  string inFileArgR = dataDir + "/" + inFileNameR + lsString + passString;

  string inFileName15 = "HyperTritonTree_15o";
  string inFileArg15 = dataDir + "/" + inFileName15 + lsString + passString;

  string outFileName = "DataTable_18";
  string outFileArg = tableDir + "/" + outFileName + kintstring + lsString + passString;

  TChain inputChain("_custom/fTreeV0");
  inputChain.AddFile(inFileArgQ.data());
  inputChain.AddFile(inFileArgR.data());
  if (kInt7)
    inputChain.AddFile(inFileArg15.data());

  TTreeReader fReader(&inputChain);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};

  TFile outFile(outFileArg.data(), "RECREATE");
  Table2 tree("DataTable", "Data Table");

  TH1D eventCounter{"EventCounter", ";Centrality (%);Events", 100, 0, 100};

  while (fReader.Next())
  {
    if (kInt7)
    {
      if (!(RColl->fTrigger == 9 || RColl->fTrigger == 1 || RColl->fTrigger == 9 + 2 || RColl->fTrigger == 1 + 2 || RColl->fTrigger == 9 + 4 || RColl->fTrigger == 1 + 4))
        continue;
    }
    eventCounter.Fill(RColl->fCent);

    for (auto &RHyper : RHyperVec)
      tree.Fill(RHyper, *RColl);
  }

  outFile.cd();
  eventCounter.Write();
  tree.Write();
  outFile.Close();

  cout << "\nDerived tree from Data generated!\n"
       << endl;
}
