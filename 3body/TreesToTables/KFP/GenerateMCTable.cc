#include <iostream>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TRandom3.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

using namespace std;

#include "../../../common/GenerateTable/Common.h"
#include "../../../common/GenerateTable/KFGenTable3.h"
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

int RecoIndex(TTreeReaderArray<int>& vec, int index) {
  for (unsigned int i = 0; i < vec.GetSize(); ++i) {
    if (vec[i] == index) return i;
  }
  return -999;
}

void GenerateMCTable(bool reject = true) {
  gRandom->SetSeed(42);

  string hypDataDir  = getenv("HYPERML_DATA_3");
  string hypTableDir = getenv("HYPERML_TABLES_3");
  string hypUtilsDir = getenv("HYPERML_UTILS");

  string inFileName = "HyperTritonTree_19d2.root";
  string inFileArg  = hypDataDir + "/KFP/" + inFileName;

  string outFileName = "NewSignalTable.root";
  string outFileArg  = hypTableDir + "/KFP/" + outFileName;

  string bwFileName = "BlastWaveFits.root";
  string bwFileArg  = hypUtilsDir + "/" + bwFileName;

  // get the bw functions for the pt rejection
  TFile bwFile(bwFileArg.data());

  TF1* BlastWave{nullptr};
  TF1* BlastWave0{(TF1*)bwFile.Get("BlastWave/BlastWave0")};
  TF1* BlastWave1{(TF1*)bwFile.Get("BlastWave/BlastWave1")};
  TF1* BlastWave2{(TF1*)bwFile.Get("BlastWave/BlastWave2")};

  float max  = 0.0;
  float max0 = BlastWave0->GetMaximum();
  float max1 = BlastWave1->GetMaximum();
  float max2 = BlastWave2->GetMaximum();

  TChain inputChain("Hyp3KF");
  inputChain.AddFile(inFileArg.data());

  TTreeReader fReader(&inputChain);
  TTreeReaderValue<REvent3KF> RColl           = {fReader, "RCollision"};
  TTreeReaderArray<RHyperTriton3KF> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderArray<SHyperTriton3KF> SHyperVec = {fReader, "SHyperTriton"};
  TTreeReaderArray<int> SGenRecMap            = {fReader, "SGenRecMap"};

  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");
  Table3 table("Table", "Table", true);
  GenTable3 genTable("GenTable", "Generated hypertritons table");

  std::cout << "\n\nGenerating derived tables from MC tree..." << std::endl;

  long double counter = 0;
  long long entries   = inputChain.GetEntries();

  while (fReader.Next()) {
    counter++;
    // get centrality for pt rejection
    auto cent = RColl->fCent;
    if (cent > 90.) continue;

    // define the BW to use for the rejection
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

    for (unsigned int i = 0; i < SHyperVec.GetSize(); ++i) {
      const SHyperTriton3KF* SHyper = &SHyperVec[i];
      float pt_gen                  = SHyper->pt;

      if (pt_gen > 10. || pt_gen < 1.) continue;
      float BlastWaveNum = BlastWave->Eval(pt_gen) / max;

      if (reject && BlastWaveNum < gRandom->Rndm()) continue;
      genTable.Fill(*SHyper, *RColl);

      int index = RecoIndex(SGenRecMap, i);
      if (index >= 0) {
        table.Fill(RHyperVec[index], *RColl, SHyper);
      }
    }
    PrintProgress(counter/entries);
  }

  outFile.cd();

  table.Write();
  genTable.Write();

  outFile.Close();

  std::cout << "\nDerived tables from MC generated!\n" << std::endl;
}
