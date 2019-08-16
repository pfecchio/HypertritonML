#include <TChain.h>
#include <TF1.h>
#include <TFile.h>
#include <TRandom3.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include "AliPID.h"
#include "AliAnalysisTaskHyperTriton2He3piML.h"

#include <vector>

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/Table.h"
#include "../../common/GenerateTable/GenTable.h"

void HyperTreeFatherMC(bool reject = true,TString name="HyperTritonTree_19d2.root")
{

  char *dataDir{nullptr}, *tableDir{nullptr};
  getDirs(dataDir, tableDir);

  TFile bwFile("../fitsM.root");
  TF1 *BlastWave{nullptr};
  TF1 *BlastWave0{(TF1 *)bwFile.Get("BlastWave/BlastWave0")};
  TF1 *BlastWave1{(TF1 *)bwFile.Get("BlastWave/BlastWave1")};
  TF1 *BlastWave2{(TF1 *)bwFile.Get("BlastWave/BlastWave2")};


  float max;
  float max0 = BlastWave0->GetMaximum();
  float max1 = BlastWave1->GetMaximum();
  float max2 = BlastWave2->GetMaximum();


  TChain mcChain("_default/fTreeV0");
  mcChain.AddFile(Form("%s/%s", dataDir,name.Data()));

  TTreeReader fReader(&mcChain);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
  TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};

  TFile tableFile(Form("%s/SignalTable.root", tableDir), "RECREATE");
  Table outputTable("SignalTable", "Signal table");
  GenTable outputGenTable("GenTable", "Generated particle table");

  while (fReader.Next())
  {
    auto cent = RColl->fCent;

    if (cent <= 10)
    {
      BlastWave = BlastWave0;
      max = max0;
    }
    if (cent > 10. && cent < 40.)
    {
      BlastWave = BlastWave1;
      max = max1;
    }
    else
    {
      BlastWave = BlastWave2;
      max = max2;
    }

    for (auto &SHyper : SHyperVec)
    {

      bool matter = SHyper.fPdgCode > 0;

      double pt = std::hypot(SHyper.fPxHe3 + SHyper.fPxPi, SHyper.fPyHe3 + SHyper.fPyPi);
    
      float BlastWaveNum = BlastWave->Eval(pt) / max;
      if (reject)
      {
        if (BlastWaveNum < gRandom->Rndm())
          continue;
        
      }
      outputGenTable.Fill(SHyper, *RColl);
      int ind = SHyper.fRecoIndex;

      if (ind >= 0)
      {
        auto &RHyper = RHyperVec[ind];
        outputTable.Fill(RHyper, *RColl);
      }
    }
  }

  tableFile.cd();
  outputTable.Write();
  outputGenTable.Write();
}
