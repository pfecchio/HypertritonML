#include <iostream>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TList.h>
#include <TRandom3.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include "Math/LorentzVector.h"

using namespace std;

#include "../../common/GenerateTable/AppliedTable3.h"
#include "../../common/GenerateTable/Common.h"

void GenerateDataTable() {
  gRandom->SetSeed(42);

  string dataDir  = getenv("HYPERML_DATA_3");
  string tableDir = getenv("HYPERML_TABLES_3");

  string inFileNameQ = "HyperTritonTreeData_18q.root";
  string inFileArgQ  = dataDir + "/" + inFileNameQ;

  string inFileNameR = "HyperTritonTreeData_18r.root";
  string inFileArgR  = dataDir + "/" + inFileNameR;

  string inAnResultsNameQ = "AnalysisResults_18q.root";
  string inAnResultsArgQ  = dataDir + "/" + inAnResultsNameQ;

  string inAnResultsNameR = "AnalysisResults_18q.root";
  string inAnResultsArgR  = dataDir + "/" + inAnResultsNameR;

  string outFileName = "DataTable.root";
  string outFileArg  = tableDir + "/" + outFileName;

  TFile *inAnResultsFileQ = new TFile(inAnResultsArgQ.data(), "READ");
  TFile *inAnResultsFileR = new TFile(inAnResultsArgR.data(), "READ");

  // read the tree
  TChain inputChain("fHypertritonTree");
  inputChain.AddFile(inFileArgQ.data());
  inputChain.AddFile(inFileArgR.data());

  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");

  TTreeReader fReader(&inputChain);
  TTreeReaderArray<MLSelected> mlSel = {fReader, "MLSelected"};

  AppliedTable3 table("DataTable", "DataTable");

  while (fReader.Next()) {
    for (auto &sel : mlSel) {
      table.Fill(sel);
    }
  }

  TList *listQ = (TList *)inAnResultsFileQ->Get("AliAnalysisTaskHypertriton3ML_summary");
  TList *listR = (TList *)inAnResultsFileR->Get("AliAnalysisTaskHypertriton3ML_summary");

  TH1F *hCentQ = (TH1F *)listQ->FindObject("Centrality_selected");
  TH1F *hCentR = (TH1F *)listR->FindObject("Centrality_selected");

  hCentQ->Add(hCentR);
  hCentQ->SetName("EventCounter");

  outFile.cd();

  table.Write();
  hCentQ->Write();

  outFile.Close();

  cout << "\nTable for ML selected data generated!\n" << endl;
}