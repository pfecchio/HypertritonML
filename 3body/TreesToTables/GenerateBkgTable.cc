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

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/Table3.h"

void GenerateBkgTable() {
  gRandom->SetSeed(42);

  string dataDir  = getenv("HYPERML_DATA_3");
  string tableDir = getenv("HYPERML_TABLES_3");

  string inFileNameQ = "HyperTritonTreeBkg_18q.root";
  string inFileArgQ  = dataDir + "/" + inFileNameQ;

  string inFileNameR = "HyperTritonTreeBkg_18r.root";
  string inFileArgR  = dataDir + "/" + inFileNameR;

  string outFileName = "DataTableBkg.root";
  string outFileArg  = tableDir + "/" + outFileName;

  TChain inputChain("fHypertritonTree");
  inputChain.AddFile(inFileArgQ.data());
  inputChain.AddFile(inFileArgR.data());

  // new flat tree with the features
  TFile outFile(outFileArg.data(), "RECREATE");

  TTreeReader fReader(&inputChain);
  TTreeReaderValue<REvent> rEv             = {fReader, "REvent"};
  TTreeReaderArray<RHypertriton3> rHyp3Vec = {fReader, "RHypertriton"};

  Table3 table("BackgroundTable", "BackgroundTable");

  while (fReader.Next()) {

    for (auto &rHyp3 : rHyp3Vec) {
      using namespace ROOT::Math;
      const LorentzVector<PxPyPzM4D<double>> deu4Vector{rHyp3.fPxDeu, rHyp3.fPyDeu, rHyp3.fPzDeu, kDeuMass};
      const LorentzVector<PxPyPzM4D<double>> p4Vector{rHyp3.fPxP, rHyp3.fPyP, rHyp3.fPzP, kPMass};
      const LorentzVector<PxPyPzM4D<double>> pi4Vector{rHyp3.fPxPi, rHyp3.fPyPi, rHyp3.fPzPi, kPiMass};
      const LorentzVector<PxPyPzM4D<double>> hyper4Vector = deu4Vector + p4Vector + pi4Vector;

      if (hyper4Vector.Pt() > 10. || hyper4Vector.Pt() < 1.) continue;

      const double decayLenght[3]{rHyp3.fDecayVtxX - rEv->fX, rHyp3.fDecayVtxY - rEv->fY, rHyp3.fDecayVtxZ - rEv->fZ};
      const double decayLenghtNorm = Hypot(decayLenght[0], decayLenght[1], decayLenght[2]);

      float ct = decayLenghtNorm / (hyper4Vector.Beta() * hyper4Vector.Gamma());

      if (ct > 1. && ct <= 2.) {
        if (gRandom->Rndm() < 0.026) {
          table.Fill(rHyp3, *rEv);
        }
      } else {
        table.Fill(rHyp3, *rEv);
      }
      
    }
  }

  outFile.cd();
  table.Write();

  outFile.Close();

  cout << "\nTable for background generated!\n" << endl;
}