#ifndef HYPERGENTABLE3_H
#define HYPERGENTABLE3_H

#include "Common.h"

#include <string>

#include <TLorentzVector.h>
#include <TTree.h>
#include <TVector3.h>

#include "AliAnalysisTaskHypertriton3ML.h"
#include "AliPID.h"
#include "Math/LorentzVector.h"

class GenTable3 {
public:
  GenTable3(std::string name, std::string title);
  void Fill(const SHypertriton3 &sHyp3, const REvent &rEv);
  void Write() { tree->Write(); }

private:
  TTree *tree;

  float fCentrality;
  float fPt;
  float fRapidity;
  float fPhi;
  float fCt;
  bool fMatter;
};

GenTable3::GenTable3(std::string name, std::string title) {
  tree = new TTree(name.data(), title.data());

  tree->Branch("pT", &fPt);
  tree->Branch("rapidity", &fRapidity);
  tree->Branch("phi", &fPhi);
  tree->Branch("ct", &fCt);
  tree->Branch("centrality", &fCentrality);
  tree->Branch("matter", &fMatter);
};

void GenTable3::Fill(const SHypertriton3 &sHyp3, const REvent &rEv) {
  // get event centrality
  fCentrality = rEv.fCent;

  // matter or anti-matter
  fMatter = sHyp3.fPdgCode > 0;

  // compute decay lenght
  const double decayLenght = Hypot(sHyp3.fDecayVtxX, sHyp3.fDecayVtxY, sHyp3.fDecayVtxZ);

  // compute the 4-vector of the daughter tracks
  const double eDeu = Hypot(sHyp3.fPxDeu, sHyp3.fPyDeu, sHyp3.fPzDeu, kDeuMass);
  const double eP   = Hypot(sHyp3.fPxP, sHyp3.fPyP, sHyp3.fPzP, kPMass);
  const double ePi  = Hypot(sHyp3.fPxPi, sHyp3.fPyPi, sHyp3.fPzPi, kPiMass);

  const TLorentzVector deu4Vector{sHyp3.fPxDeu, sHyp3.fPyDeu, sHyp3.fPzDeu, eDeu};
  const TLorentzVector p4Vector{sHyp3.fPxP, sHyp3.fPyP, sHyp3.fPzP, eP};
  const TLorentzVector pi4Vector{sHyp3.fPxPi, sHyp3.fPyPi, sHyp3.fPzPi, ePi};

  // compute the 4-vector of the hypertriton candidate
  const TLorentzVector hyp4Vector = deu4Vector + p4Vector + pi4Vector;

  fPt       = hyp4Vector.Pt();
  fPhi      = hyp4Vector.Phi();
  fCt       = decayLenght * kHyperTritonMass / hyp4Vector.P();
  fRapidity = hyp4Vector.Rapidity();

  // fill the tree
  tree->Fill();
}

#endif