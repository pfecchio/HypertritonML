#include <iostream>
#include <string>
#include <vector>

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TRandom3.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <TLorentzVector.h>
#include "AliAnalysisTaskHyperTriton2He3piML.h"
#include "AliPID.h"

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/GenTable2.h"
#include "../../common/GenerateTable/Table2.h"

using std::string;
using namespace ROOT::Math;

void GenerateTableFromMC(bool reject = true, string hypDataDir = "", string hypTableDir = "", string ptShape = "bw")
{

  double lamCt = 262.5 * 0.029979245800;
  double rejCt = 210 * 0.029979245800;
  gRandom->SetSeed(1989);
  if (hypDataDir == "")
    hypDataDir = getenv("HYPERML_TREES__2");
  if (hypTableDir == "")
    hypTableDir = getenv("HYPERML_TABLES_2");
  string hypUtilsDir = getenv("HYPERML_UTILS");

  string pass = "3";

  bool useProposeMasses = (pass == "3");

  string otf = "";

  string mcName = "20g7";

  string inFileName = string("HyperTritonTree_") + mcName + otf + ".root";
  string inFileArg = hypDataDir + "/" + inFileName;

  string outFileName = string("SignalTable_B_") + mcName + otf + ".root";
  string outFileArg = hypTableDir + "/" + outFileName;

  string bwFileName = "Anti_fits.root";
  string bwFileArg = hypUtilsDir + "/" + bwFileName;


  // get the bw functions for the pt rejection
  TFile bwFile(bwFileArg.data());

  TF1 *hypPtShape{nullptr};
  TF1 *hypPtShape0{nullptr};
  TF1 *hypPtShape1{nullptr};
  TF1 *hypPtShape2{nullptr};
  TF1 *hypPtShape3{nullptr};
  TF1 *hypPtShape4{nullptr};

  if (ptShape == "bw")
  {
    hypPtShape0 = (TF1 *)bwFile.Get("BGBW/0/BGBW0");
    hypPtShape1 = (TF1 *)bwFile.Get("BGBW/1/BGBW1");
    hypPtShape2 = (TF1 *)bwFile.Get("BGBW/2/BGBW2");
    hypPtShape3 = (TF1 *)bwFile.Get("BGBW/3/BGBW3");
    hypPtShape4 = (TF1 *)bwFile.Get("BGBW/4/BGBW4");
  }

  float max = 0.0;
  float max0 = hypPtShape0->GetMaximum();
  float max1 = hypPtShape1->GetMaximum();
  float max2 = hypPtShape2->GetMaximum();
  float max3 = hypPtShape3->GetMaximum();
  float max4 = hypPtShape4->GetMaximum();

  // read the tree
  TFile *inFile = new TFile(inFileArg.data(), "READ");

  TTreeReader fReader("_default/fTreeV0", inFile);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
  TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};

  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");
  Table2 table("SignalTable", "Signal Table");
  GenTable2 genTable("GenTable", "Generated particle table");

  while (fReader.Next())
  {
    auto cent = RColl->fCent;

    if (cent <= 5)
    {
      hypPtShape = hypPtShape0;
      max = max0;
    }
    if (cent <= 10)
    {
      hypPtShape = hypPtShape1;
      max = max1;
    }
    else if (cent <= 30.)
    {
      hypPtShape = hypPtShape2;
      max = max2;
    }

    else if (cent <= 50)
    {
      hypPtShape = hypPtShape3;
      max = max3;
    }
    else if (cent <= 90)
    {
      hypPtShape = hypPtShape4;
      max = max4;
    }

    for (auto &SHyper : SHyperVec)
    {

      bool matter = SHyper.fPdgCode > 0;

      double pt = std::hypot(SHyper.fPxHe3 + SHyper.fPxPi, SHyper.fPyHe3 + SHyper.fPyPi);
      LorentzVector<PxPyPzM4D<double>> sHe3{SHyper.fPxHe3, SHyper.fPyHe3, SHyper.fPzHe3, AliPID::ParticleMass(AliPID::kHe3)};
      LorentzVector<PxPyPzM4D<double>> sPi{SHyper.fPxPi, SHyper.fPyPi, SHyper.fPzPi, AliPID::ParticleMass(AliPID::kPion)};
      LorentzVector<PxPyPzM4D<double>> sMother = sHe3 + sPi;
      double len = Hypote(SHyper.fDecayX, SHyper.fDecayY, SHyper.fDecayZ);
      auto Ct = len * kHyperMass / sMother.P();

      if (reject)
      {
        float hypPtShapeNum = hypPtShape->Eval(pt) / max;
        if (hypPtShapeNum < gRandom->Rndm())
          continue;
      }
      genTable.Fill(SHyper, *RColl);
      int ind = SHyper.fRecoIndex;


      if (ind >= 0)
      {
        auto &RHyper = RHyperVec[ind];
        table.Fill(RHyper, *RColl, useProposeMasses);
        double recpt = std::hypot(RHyper.fPxHe3 + RHyper.fPxPi, RHyper.fPyHe3 + RHyper.fPyPi);
      }
    }
  }

  outFile.cd();

  table.Write();
  genTable.Write();
  outFile.Close();
}
