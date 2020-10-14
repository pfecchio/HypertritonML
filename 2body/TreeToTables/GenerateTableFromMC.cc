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

void GenerateTableFromMC(bool reject = true, string hypDataDir = "", string hypTableDir = "", string ptShape = "bw")
{
  gRandom->SetSeed(1989);

  if (hypDataDir == "")
    hypDataDir = getenv("HYPERML_DATA_2");
  if (hypTableDir == "")
    hypTableDir = getenv("HYPERML_TABLES_2");
  string hypUtilsDir = getenv("HYPERML_UTILS");

  string mcName = getenv("HYPERML_MC");

  string inFileName = string("HyperTritonTree_") + mcName + ".root";
  string inFileArg = hypDataDir + "/" + inFileName;

  string outFileName = string("SignalTable_") + mcName + ".root";
  string outFileArg = hypTableDir + "/" + outFileName;

  string absFileName = "absorption.root";
  string absFileArg = hypUtilsDir + "/" + absFileName;

  string bwFileName = "BlastWaveFits.root";
  string bwFileArg = hypUtilsDir + "/" + bwFileName;

  TFile absFile(absFileArg.data());
  TH1 *hCorrM = (TH1 *)absFile.Get("hCorrectionHyp");
  TH1 *hCorrA = (TH1 *)absFile.Get("hCorrectionAntiHyp");

  // get the bw functions for the pt rejection
  TFile bwFile(bwFileArg.data());

  TF1 *hypPtShape{nullptr}; TF1 *hypPtShape0; TF1 *hypPtShape1; TF1 *hypPtShape2;
  if (ptShape == "bw")
    {
      hypPtShape0 = (TF1 *)bwFile.Get("BlastWave/BlastWave0");
      hypPtShape1 = (TF1 *)bwFile.Get("BlastWave/BlastWave1");
      hypPtShape2 = (TF1 *)bwFile.Get("BlastWave/BlastWave2");
    }

  else if (ptShape == "bol")
  {
    hypPtShape0 = (TF1 *)bwFile.Get("Boltzmann/Boltzmann0");
    hypPtShape1 = (TF1 *)bwFile.Get("Boltzmann/Boltzmann1");
    hypPtShape2 = (TF1 *)bwFile.Get("Boltzmann/Boltzmann2");
  }

    else if (ptShape == "mtexp")
  {
    hypPtShape0 = (TF1 *)bwFile.Get("Mt-exp/Mt-exp0");
    hypPtShape1 = (TF1 *)bwFile.Get("Mt-exp/Mt-exp1");
    hypPtShape2 = (TF1 *)bwFile.Get("Mt-exp/Mt-exp2");
  }



  float max = 0.0;
  float max0 = hypPtShape0->GetMaximum();
  float max1 = hypPtShape1->GetMaximum();
  float max2 = hypPtShape2->GetMaximum();

  // read the tree
  TFile *inFile = new TFile(inFileArg.data(), "READ");

  TTreeReader fReader("_default/fTreeV0", inFile);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
  TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};

  TH2D *hNSigmaTPCVsPtHe3 =
      new TH2D("nSigmaTPCvsPTHe3", ";#it{p}_{T} (GeV/#it{c});n#sigma_{TPC}", 32, 2, 10, 128, -8, 8);

  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");
  Table2 table("SignalTable", "Signal Table");
  GenTable2 genTable("GenTable", "Generated particle table");

  TH1D genHA("genA", ";ct (cm)", 50, 0, 40);
  TH1D absHA("absA", ";ct (cm)", 50, 0, 40);

  TH1D genHM("genM", ";ct (cm)", 50, 0, 40);
  TH1D absHM("absM", ";ct (cm)", 50, 0, 40);

  while (fReader.Next())
  {
    auto cent = RColl->fCent;

    if (cent <= 10)
    {
      hypPtShape = hypPtShape0;
      max = max0;
    }
    else if (cent <= 40.)
    {
      hypPtShape = hypPtShape1;
      max = max1;
    }
    else if (cent <= 100)
    {
      hypPtShape = hypPtShape2;
      max = max2;
    }
    for (auto &SHyper : SHyperVec)
    {

      bool matter = SHyper.fPdgCode > 0;

      double pt = std::hypot(SHyper.fPxHe3 + SHyper.fPxPi, SHyper.fPyHe3 + SHyper.fPyPi);
      
      if (reject)
      {
        float hypPtShapeNum = hypPtShape->Eval(pt) / max;
        if (hypPtShapeNum < gRandom->Rndm())
          continue;
      }
      genTable.Fill(SHyper, *RColl);
      int ind = SHyper.fRecoIndex;

      (genTable.IsMatter() ? genHM : genHA).Fill(genTable.GetCt());
      float protonPt = pt / 3.;
      float corrBin = hCorrA->FindBin(protonPt);
      float threshold = (genTable.IsMatter() ? hCorrM : hCorrA)->GetBinContent(corrBin);
      if (gRandom->Rndm() < threshold)
        (genTable.IsMatter() ? absHM : absHA).Fill(genTable.GetCt());

      if (ind >= 0)
      {
        auto &RHyper = RHyperVec[ind];
        table.Fill(RHyper, *RColl, false);
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


}
