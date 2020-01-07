#include <iostream>
#include <vector>

#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

using namespace std;

#include "../../common/GenerateTable/AppliedTable3.h"
#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/Table3.h"

void GenerateTableFromData(bool appData = true) {

  string dataDir  = getenv("HYPERML_DATA_3");
  string tableDir = getenv("HYPERML_TABLES_3");

  string inFileName = "HyperTritonTree_18q.root";
  string inFileArg  = dataDir + "/" + inFileName;

  string outFileName = "DataTable.root";
  string outFileArg  = tableDir + "/" + outFileName;

  // read the tree
  TFile *inFile = new TFile(inFileArg.data(), "READ");
  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");

  TTreeReader fReader("fHypertritonTree", inFile);

  if (appData) {
    TTreeReaderArray<MLSelected> mlSel = {fReader, "MLSelected"};

    AppliedTable3 table("DataTable", "DataTable");

    while (fReader.Next()) {
      for (auto &sel : mlSel) {
        table.Fill(sel);
      }
    }

    outFile.cd();
    table.Write();

  } else {
    TTreeReaderValue<REvent> rEv             = {fReader, "REvent"};
    TTreeReaderArray<RHypertriton3> rHyp3Vec = {fReader, "RHypertriton"};

    Table3 table("BackgroundTable", "BackgroundTable");

    while (fReader.Next()) {
      for (auto &rHyp3 : rHyp3Vec) {
        table.Fill(rHyp3, *rEv);
      }
    }

    outFile.cd();
    table.Write();
  }

  inFile->Close();
  outFile.Close();

  cout << "\nDerived tree from Data generated!\n" << endl;
}