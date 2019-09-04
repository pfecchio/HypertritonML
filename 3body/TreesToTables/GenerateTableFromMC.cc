#include <boost/progress.hpp>
#include <iostream>
#include <vector>

#include <TFile.h>
#include <TH1D.h>
#include <TRandom3.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

using namespace std;

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/GenTable3.h"
#include "../../common/GenerateTable/Table3.h"

void GenerateTableFromMC(bool reject = true) {

  string hypDataDir  = getenv("HYPERML_DATA_3");
  string hypTableDir = getenv("HYPERML_TABLES_3");
  string hypUtilsDir = getenv("HYPERML_UTILS");

  string inFileName = "HyperTritonTree_19d2.root";
  string inFileArg  = hypDataDir + "/" + inFileName;

  string outFileName = "HyperTritonTable_19d2.root";
  string outFileArg  = hypTableDir + "/" + outFileName;

  string bwFileName = "BlastWaveFits.root";
  string bwFileArg  = hypUtilsDir + "/" + bwFileName;

  // get the bw functions for the pt rejection
  TFile bwFile(bwFileArg.data());

  TF1 *BlastWave{nullptr};
  TF1 *BlastWave0{(TF1 *)bwFile.Get("BlastWave/BlastWave0")};
  TF1 *BlastWave1{(TF1 *)bwFile.Get("BlastWave/BlastWave1")};
  TF1 *BlastWave2{(TF1 *)bwFile.Get("BlastWave/BlastWave2")};

  BlastWave0->SetNormalized(1);
  BlastWave1->SetNormalized(1);
  BlastWave2->SetNormalized(1);

  float max  = 0.0;
  float max0 = BlastWave0->GetMaximum();
  float max1 = BlastWave1->GetMaximum();
  float max2 = BlastWave2->GetMaximum();

  // read the tree
  TFile *inFile = new TFile(inFileArg.data(), "READ");

  TTreeReader fReader("fHypertritonTree", inFile);
  TTreeReaderValue<REvent> rEv             = {fReader, "REvent"};
  TTreeReaderArray<SHypertriton3> sHyp3Vec = {fReader, "SHypertriton"};
  TTreeReaderArray<RHypertriton3> rHyp3Vec = {fReader, "RHypertriton"};

  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");
  Table3 table("SignalTable", "SignalTable");
  GenTable3 genTable("GenerateTable", "Generated particle table");

  std::cout << "\n\nGenerating derived tables from MC tree..." << std::endl;

  // info for progress bar
  int n_entries = 33480530;

  // progress bar
  boost::progress_display show_progress(n_entries);

  while (fReader.Next()) {
    // get centrality for pt rejection
    auto cent = rEv->fCent;

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

    for (auto &sHyp3 : sHyp3Vec) {
      ++show_progress;

      bool matter = sHyp3.fPdgCode > 0;

      // compute the 4-vector of the daughter tracks
      const double eDeu = Hypot(sHyp3.fPxDeu, sHyp3.fPyDeu, sHyp3.fPzDeu, kDeuMass);
      const double eP   = Hypot(sHyp3.fPxP, sHyp3.fPyP, sHyp3.fPzP, kPMass);
      const double ePi  = Hypot(sHyp3.fPxPi, sHyp3.fPyPi, sHyp3.fPzPi, kPiMass);

      const TLorentzVector deu4Vector{sHyp3.fPxDeu, sHyp3.fPyDeu, sHyp3.fPzDeu, eDeu};
      const TLorentzVector p4Vector{sHyp3.fPxP, sHyp3.fPyP, sHyp3.fPzP, eP};
      const TLorentzVector pi4Vector{sHyp3.fPxPi, sHyp3.fPyPi, sHyp3.fPzPi, ePi};

      // compute the 4-vector of the generated hypertriton and its pT
      const TLorentzVector hyp4Vector = deu4Vector + p4Vector + pi4Vector;
      float pT                        = hyp4Vector.Pt();

      float BlastWaveNum = BlastWave->Eval(pT) / max;

      if (reject && BlastWaveNum < gRandom->Rndm()) continue;
      genTable.Fill(sHyp3, *rEv);

      // fill the rec table if the hypertriton was actually reconstructed
      int ind = sHyp3.fRecoIndex;
      if (ind >= 0) {
        auto &rHyp3 = rHyp3Vec[ind];
        table.Fill(rHyp3, *rEv);
      }
    }
  }

  inFile->Close();
  outFile.cd();

  table.Write();
  genTable.Write();

  outFile.Close();

  std::cout << "\nDerived tables from MC generated!\n" << std::endl;
}
