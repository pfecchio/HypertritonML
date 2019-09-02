#ifndef HYPERGENTABLE2_H
#define HYPERGENTABLE2_H

#include "Common.h"

#include <string>

#include <TLorentzVector.h>
#include <TTree.h>
#include <TVector3.h>

#include "AliAnalysisTaskHyperTriton2He3piML.h"
#include "AliPID.h"
#include "Math/LorentzVector.h"

class GenTable2 {
  public: 
  GenTable2(std::string name, std::string title);
  void Fill(const SHyperTritonHe3pi& SHyperVec, const RCollision& RColl);
  void Write() { tree->Write(); }

  private:
  TTree* tree;
  float Pt;
  float Rapidity;
  float Phi;
  float Ct;
  float Centrality;
  bool Matter;    
};

GenTable2::GenTable2(std::string name, std::string title) {
  tree = new TTree(name.data(), title.data());

  tree->Branch("pT", &Pt);
  tree->Branch("rapidity", &Rapidity);
  tree->Branch("phi", &Phi);
  tree->Branch("ct", &Ct);
  tree->Branch("centrality", &Centrality);
  tree->Branch("matter", &Matter);
};

void GenTable2::Fill(const SHyperTritonHe3pi& SHyper, const RCollision& RColl) {
  Centrality = RColl.fCent;
  Matter = SHyper.fPdgCode > 0;
  const double len = Hypot(SHyper.fDecayX, SHyper.fDecayY, SHyper.fDecayZ);

  using namespace ROOT::Math;
  const LorentzVector<PxPyPzM4D<double>> sHe3{SHyper.fPxHe3, SHyper.fPyHe3, SHyper.fPzHe3, AliPID::ParticleMass(AliPID::kHe3)};
  const LorentzVector<PxPyPzM4D<double>> sPi{SHyper.fPxPi, SHyper.fPyPi, SHyper.fPzPi, AliPID::ParticleMass(AliPID::kPion)};
  const LorentzVector<PxPyPzM4D<double>> sMother = sHe3 + sPi;
  Pt = sMother.Pt();
  Phi = sMother.Phi();
  Ct = len * kHyperTritonMass / sMother.P();
  Rapidity = sMother.Rapidity();
  tree->Fill();
}

#endif