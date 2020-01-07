#ifndef HYPERAPPTABLE3_H
#define HYPERAPPTABLE3_H

#include "Common.h"

#include <string>

#include <TLorentzVector.h>
#include <TTree.h>
#include <TVector3.h>

#include "AliAnalysisTaskHypertriton3ML.h"
#include "AliPID.h"

class AppliedTable3 {
public:
  AppliedTable3(std::string name, std::string title);
  void Fill(const MLSelected &mlSel);
  void Write() { tree->Write(); }

private:
  TTree *tree;

  float fScore;
  float fCentrality;
  float fCt;
  float fInvMass;
  float fHypCandPt;
};

AppliedTable3::AppliedTable3(std::string name, std::string title) {
  tree = new TTree(name.data(), title.data());

  tree->Branch("score", &fScore);
  tree->Branch("centrality", &fCentrality);
  tree->Branch("ct", &fCt);
  tree->Branch("InvMass", &fInvMass);
  tree->Branch("HypCandPt", &fHypCandPt);
};

void AppliedTable3::Fill(const MLSelected &mlSel) {
  // get gentrality of the collision
  fScore      = mlSel.score;
  fCentrality = mlSel.fCentrality;
  fCt         = mlSel.fCt;
  fInvMass    = mlSel.fInvMass;
  fHypCandPt  = mlSel.fCandPt;

  // fill the tree
  tree->Fill();
}

#endif