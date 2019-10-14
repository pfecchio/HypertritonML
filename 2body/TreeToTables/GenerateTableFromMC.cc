#include <iostream>
#include <vector>

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TRandom3.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include "AliAnalysisTaskHyperTriton2He3piML.h"
#include "AliPID.h"

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/GenTable2.h"
#include "../../common/GenerateTable/Table2.h"

void GenerateTableFromMC(bool reject = true) {
  gRandom->SetSeed(1989);

  string hypDataDir  = getenv("HYPERML_DATA_2");
  string hypTableDir = getenv("HYPERML_TABLES_2");
  string hypUtilsDir = getenv("HYPERML_UTILS");

  string inFileName = "HyperTritonTree_19d2.root";
  string inFileArg  = hypDataDir + "/" + inFileName;

  string outFileName = "SignalTable.root";
  string outFileArg  = hypTableDir + "/" + outFileName;

  string absFileName = "absorption.root";
  string absFileArg  = hypUtilsDir + "/" + absFileName;
  TFile absFile(absFileArg.data());
  TH1* hCorrM = (TH1*)absFile.Get("hCorrectionHyp");
  TH1* hCorrA = (TH1*)absFile.Get("hCorrectionAntiHyp");

  string bwFileName = "BlastWaveFits.root";
  string bwFileArg  = hypUtilsDir + "/" + bwFileName;

  // get the bw functions for the pt rejection
  TFile bwFile(bwFileArg.data());

  TF1 *BlastWave{nullptr};
  TF1 *BlastWave0{(TF1 *)bwFile.Get("BlastWave/BlastWave0")};
  TF1 *BlastWave1{(TF1 *)bwFile.Get("BlastWave/BlastWave1")};
  TF1 *BlastWave2{(TF1 *)bwFile.Get("BlastWave/BlastWave2")};

  float max  = 0.0;
  float max0 = BlastWave0->GetMaximum();
  float max1 = BlastWave1->GetMaximum();
  float max2 = BlastWave2->GetMaximum();

  // read the tree
  TFile *inFile = new TFile(inFileArg.data(), "READ");

  TTreeReader fReader("_custom/fTreeV0", inFile);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
  TTreeReaderValue<RCollision> RColl            = {fReader, "RCollision"};

  TH2D *hNSigmaTPCVsPtHe3 =
      new TH2D("nSigmaTPCvsPTHe3", ";#it{p}_{T} (GeV/#it{c});n#sigma_{TPC}", 32, 2, 10, 128, -8, 8);

  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");
  Table2 table("SignalTable", "Signal Table");
  GenTable2 genTable("GenTable", "Generated particle table");

  TH1D genHA("genA", ";ct (cm)",50,0,40);
  TH1D absHA("absA", ";ct (cm)",50,0,40);

  TH1D genHM("genM", ";ct (cm)",50,0,40);
  TH1D absHM("absM", ";ct (cm)",50,0,40);
  while (fReader.Next()) {
    auto cent = RColl->fCent;

    if (cent <= 10) {
      BlastWave = BlastWave0;
      max       = max0;
    }
    if (cent > 10. && cent <= 40.) {
      BlastWave = BlastWave1;
      max       = max1;
    } else {
      BlastWave = BlastWave2;
      max       = max2;
    }
    for (auto &SHyper : SHyperVec) {

      bool matter = SHyper.fPdgCode > 0;

      double pt = std::hypot(SHyper.fPxHe3 + SHyper.fPxPi, SHyper.fPyHe3 + SHyper.fPyPi);

      float BlastWaveNum = BlastWave->Eval(pt) / max;
      if (reject) {
        if (BlastWaveNum < gRandom->Rndm()) continue;
      }
      genTable.Fill(SHyper, *RColl);
      int ind = SHyper.fRecoIndex;
      
      (genTable.IsMatter() ? genHM : genHA).Fill(genTable.GetCt());
      float protonPt = pt / 3.;
      float corrBin = hCorrA->FindBin(protonPt);
      float threshold = (genTable.IsMatter() ? hCorrM : hCorrA)->GetBinContent(corrBin);
      if (gRandom->Rndm() < threshold)
        (genTable.IsMatter() ? absHM : absHA).Fill(genTable.GetCt());

      if (ind >= 0) {
        auto &RHyper = RHyperVec[ind];
        table.Fill(RHyper, *RColl);
        double recpt = std::hypot(RHyper.fPxHe3 + RHyper.fPxPi, RHyper.fPyHe3 + RHyper.fPyPi);
        hNSigmaTPCVsPtHe3->Fill(recpt, RHyper.fTPCnSigmaHe3);
      }
    }
  }

  outFile.cd();

  table.Write();
  genTable.Write();
  hNSigmaTPCVsPtHe3->Write();
  absHM.Sumw2();
  genHM.Sumw2();
  absHM.Divide(&genHM);
  absHM.Write();
  absHA.Sumw2();
  genHA.Sumw2();
  absHA.Divide(&genHA);
  absHA.Write();
  outFile.Close();

  std::cout << "\nDerived tables from MC generated!\n" << std::endl;
}
