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

#include "../../../common/GenerateTable/Common.h"
#include "../../../common/GenerateTable/KFTable3.h"

void PrintProgress(double percentage) {
  int barWidth = 80;
  std::cout << "[";

  int pos = barWidth * percentage;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(percentage * 100.0) << " %\r";
  std::cout.flush();
}

void GenerateDataTable() {
  string hypDataDir  = getenv("HYPERML_DATA_3");
  string hypTableDir = getenv("HYPERML_TABLES_3");
  string hypUtilsDir = getenv("HYPERML_UTILS");

  string inFileNameQ = "HyperTritonTree_18q.root";
  string inFileArgQ  = hypDataDir + "/KFP/" + inFileNameQ;
  string inFileNameR = "HyperTritonTree_18r.root";
  string inFileArgR  = hypDataDir + "/KFP/" + inFileNameR;
  string inFileNameLS = "HyperTritonTree_LS.root";
  string inFileArgLS  = hypDataDir + "/KFP/" + inFileNameLS;

  string outFileNameQ = "DataTableQ.root";
  string outFileArgQ  = hypTableDir + "/KFP/" + outFileNameQ;
  string outFileNameR = "DataTableR.root";
  string outFileArgR  = hypTableDir + "/KFP/" + outFileNameR;
  string outFileNameA = "AllDataTable.root";
  string outFileArgA  = hypTableDir + "/KFP/" + outFileNameA;
  string outFileNameLS = "LSTable.root";
  string outFileArgLS  = hypTableDir + "/KFP/" + outFileNameLS;

  TChain inputChain("Hyp3KF");
  // inputChain.AddFile(inFileArgQ.data());
  // inputChain.AddFile(inFileArgR.data());
  inputChain.AddFile(inFileArgLS.data());

  TTreeReader fReader(&inputChain);
  TTreeReaderValue<REvent3KF> RColl           = {fReader, "RCollision"};
  TTreeReaderArray<RHyperTriton3KF> RHyperVec = {fReader, "RHyperTriton"};

  // new flat tree with the features
  // TFile outFile(outFileArgQ.data(), "RECREATE");
  // TFile outFile(outFileArgR.data(), "RECREATE");
  // TFile outFile(outFileArgA.data(), "RECREATE");
  TFile outFile(outFileArgLS.data(), "RECREATE");
  Table3 table("Table", "Table", true);
  
  TH1D eventCounter{"EventCounter", ";Centrality (%);Events", 100, 0, 100};

  std::cout << "\n\nGenerating tables from data tree..." << std::endl;

  long double counter = 0;
  long long entries   = inputChain.GetEntries();

  while (fReader.Next()) {
    counter++;
    // get centrality for pt rejection
    eventCounter.Fill(RColl->fCent);

    for (auto& RHyper : RHyperVec) {
      table.Fill(RHyper, *RColl, nullptr);
    }
    PrintProgress(counter / entries);
  }

  outFile.cd();

  table.Write();
  eventCounter.Write();

  outFile.Close();

  cout << "\nTables from data generated!\n" << endl;
}
