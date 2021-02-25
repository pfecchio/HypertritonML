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

  string dataDir  = getenv("HYPERML_TREES__3");
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

  int counter[10]{0};

  while (fReader.Next()) {
    if (rEv->fCent > 90.) continue;

    for (auto &rHyp3 : rHyp3Vec) {
      // reject candidates without deu and p tof
      if (std::abs(rHyp3.fNSigmaTOFDeu) < 3.5) continue;
      if (std::abs(rHyp3.fNSigmaTOFP) < 3.5) continue;

      using namespace ROOT::Math;
      const LorentzVector<PxPyPzM4D<double>> deu4Vector{rHyp3.fPxDeu, rHyp3.fPyDeu, rHyp3.fPzDeu, kDeuMass};
      const LorentzVector<PxPyPzM4D<double>> p4Vector{rHyp3.fPxP, rHyp3.fPyP, rHyp3.fPzP, kPMass};
      const LorentzVector<PxPyPzM4D<double>> pi4Vector{rHyp3.fPxPi, rHyp3.fPyPi, rHyp3.fPzPi, kPiMass};
      const LorentzVector<PxPyPzM4D<double>> hyper4Vector = deu4Vector + p4Vector + pi4Vector;

      if (hyper4Vector.Pt() > 10. || hyper4Vector.Pt() < 1.) continue;

      const double decayLenght[3]{rHyp3.fDecayVtxX - rEv->fX, rHyp3.fDecayVtxY - rEv->fY, rHyp3.fDecayVtxZ - rEv->fZ};
      const double decayLenghtNorm = Hypot(decayLenght[0], decayLenght[1], decayLenght[2]);

      float ct = decayLenghtNorm / (hyper4Vector.Beta() * hyper4Vector.Gamma());

      if (ct <= 1.) continue;
      if (ct > 1. && ct <= 2.) {
        if (gRandom->Rndm() < 0.005) {
          table.Fill(rHyp3, *rEv);
          counter[0]++;
        }
      } else {
        table.Fill(rHyp3, *rEv);
      }
      if (ct > 2. && ct <= 4.) {
        counter[1]++;
      }
      if (ct > 4. && ct <= 6.) {
        counter[2]++;
      }
      if (ct > 6. && ct <= 8.) {
        counter[3]++;
      }
      if (ct > 8. && ct <= 10.) {
        counter[4]++;
      }
      if (ct > 10. && ct <= 14.) {
        counter[5]++;
      }
      if (ct > 14. && ct <= 18.) {
        counter[6]++;
      }
      if (ct > 18. && ct <= 23.) {
        counter[7]++;
      }
      if (ct > 23. && ct <= 35.) {
        counter[8]++;
      }
      if (ct > 35.) {
        counter[9]++;
      }
    }
  }

  outFile.cd();
  table.Write();

  outFile.Close();

  cout << "\nTable for background generated!\n" << endl;

  cout << "ct 1-2: " << counter[0] << endl;
  cout << "ct 2-4: " << counter[1] << endl;
  cout << "ct 4-6: " << counter[2] << endl;
  cout << "ct 6-8: " << counter[3] << endl;
  cout << "ct 8-10: " << counter[4] << endl;
  cout << "ct 10-14: " << counter[5] << endl;
  cout << "ct 14-18: " << counter[6] << endl;
  cout << "ct 18-23: " << counter[7] << endl;
  cout << "ct 23-35: " << counter[8] << endl;
  cout << "ct  >35: " << counter[9] << endl;
}