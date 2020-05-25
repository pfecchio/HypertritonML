#ifndef KFGENTABLE3_H
#define KFGENTABLE3_H

#include "Common.h"

#include <cmath>

#include <TTree.h>

#include "AliAnalysisTaskHyperTriton3KF.h"

class GenTable3 {
public:
  GenTable3(const char* name, const char* title);
  void Fill(const SHyperTriton3KF& SHyperVec, const REvent3KF& RColl);
  void Write() { tree->Write(); }

private:
  TTree* tree;
  float pt;
  float rapidity;
  float phi;
  float ct;
  float centrality;
  bool matter;
};

GenTable3::GenTable3(const char* name, const char* title) {
  tree = new TTree(name, title);

  tree->Branch("pt", &pt);
  tree->Branch("rapidity", &rapidity);
  tree->Branch("phi", &phi);
  tree->Branch("ct", &ct);
  tree->Branch("centrality", &centrality);
  tree->Branch("matter", &matter);
};

void GenTable3::Fill(const SHyperTriton3KF& SHyper, const REvent3KF& RColl) {
  centrality     = RColl.fCent;
  pt             = SHyper.pt;
  const double p = std::hypot(SHyper.pz, pt);
  const double e = std::hypot(p, kHyperTritonMass);
  rapidity       = 0.5 * std::log((e + SHyper.pz) / (e - SHyper.pz + 1.e-12));
  phi            = SHyper.phi;
  ct             = SHyper.l * kHyperTritonMass / p;
  matter         = SHyper.positive;

  tree->Fill();
}

#endif