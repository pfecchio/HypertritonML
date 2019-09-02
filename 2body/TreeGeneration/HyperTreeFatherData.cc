#include "AliAnalysisTaskHyperTriton2He3piML.h"
#include "AliPID.h"
#include "TH1D.h"
#include <Riostream.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TDirectoryFile.h>
#include <TFile.h>
#include <TList.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <TROOT.h>
#include <TRandom3.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/Table2.h"

void HyperTreeFatherData(TString name = "HyperTritonTree_18r.root") {

  char *dataDir{nullptr}, *tableDir{nullptr};
  getDirs2(dataDir, tableDir);

  TChain inputChain("_custom/fTreeV0");
  inputChain.AddFile(Form("%s/%s", dataDir, name.Data()));
  inputChain.AddFile(Form("%s/HyperTritonTree_18q.root", dataDir));

  TTreeReader fReader(&inputChain);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderValue<RCollision> RColl            = {fReader, "RCollision"};

  // TFile tfileHist("CentHist.root", "READ");
  // TH1D *fHistCent = (TH1D *)tfileHist.Get("fHistCent");
  // fHistCent->SetDirectory(0);
  // tfileHist.Close();

  // double fMin = (double)fHistCent->GetBinContent(280);
  // cout << fMin << endl;

  TFile tfile(Form("%s/DataTable.root", tableDir), "RECREATE");
  Table2 tree("DataTable", "Data Table");

  TH1D eventCounter{"EventCounter",";Centrality (%);Events",100,0,100};
  while (fReader.Next()) {
    eventCounter.Fill(RColl->fCent);
    for (auto &RHyper : RHyperVec)
      tree.Fill(RHyper, *RColl);
  }

  tfile.cd();
  eventCounter.Write();
  tree.Write();
}
