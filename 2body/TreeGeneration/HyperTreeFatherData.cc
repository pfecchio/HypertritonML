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

#include "../include/Common.h"
#include "../include/Table.h"

void HyperTreeFatherData()
{
    
  char *dataDir{nullptr}, *tableDir{nullptr};
  getDirs(dataDir, tableDir);
  
  TChain inputChain("_custom/fTreeV0");
  inputChain.AddFile(Form("%s/LHC18r.root", dataDir));
  inputChain.AddFile(Form("%s/LHC18q.root", dataDir));

  TTreeReader fReader(&inputChain);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};


  // TFile tfileHist("CentHist.root", "READ");
  // TH1D *fHistCent = (TH1D *)tfileHist.Get("fHistCent");
  // fHistCent->SetDirectory(0);
  // tfileHist.Close();

  // double fMin = (double)fHistCent->GetBinContent(280);
  // cout << fMin << endl;

  TFile tfile(Form("%s/DataTable.root", tableDir), "RECREATE");
  Table tree("DataTable", "Data Table");


  while (fReader.Next())
  {
    if (RColl->fCent > 10.051 && RColl->fCent < 40.05)
      Nev1040++;
    //int bin = (int)(Centrality * 10.);
    //double height = (double)fHistCent->GetBinContent(bin);
    //  if ((gRandom->Rndm() * height) > fMin)
    //    continue;

    for (auto& RHyper : RHyperVec)
      tree.Fill(RHyper, *RColl);
  }


  tfile.cd();
  tree.Write();


}