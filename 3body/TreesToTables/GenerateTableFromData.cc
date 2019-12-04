#include <boost/progress.hpp>
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
#include "../../common/GenerateTable/Table3.h"

void GenerateTableFromData() {

  string dataDir  = getenv("HYPERML_DATA_3");
  string tableDir = getenv("HYPERML_TABLES_3");

  string inFileName = "HyperTritonTree_18qr.root";
  string inFileArg  = dataDir + "/" + inFileName;

  string outFileName = "DataTable.root";
  string outFileArg  = tableDir + "/" + outFileName;

  // read the tree
  TFile *inFile = new TFile(inFileArg.data(), "READ");

  TTreeReader fReader("fHypertritonTree", inFile);
  TTreeReaderValue<REvent> rEv             = {fReader, "REvent"};
  TTreeReaderArray<RHypertriton3> rHyp3Vec = {fReader, "RHypertriton"};

  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");
  Table3 table("BackgroundTable", "BackgroundTable");

  // info for progress bar
  int n_entries = 22084810;

  // progress bar
  boost::progress_display show_progress(n_entries);

  while (fReader.Next()) {

    for (auto &rHyp3 : rHyp3Vec) {
      table.Fill(rHyp3, *rEv);
      ++show_progress;
    }
  }

  outFile.cd();
  table.Write();
  inFile->Close();
  outFile.Close();

  cout << "\nDerived tree from Data generated!\n" << endl;
}