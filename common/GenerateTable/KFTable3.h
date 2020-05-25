#ifndef KFTABLE3_H
#define KFTABLE3_H

#include "Common.h"

#include <TTree.h>

#include "AliAnalysisTaskHyperTriton3KF.h"

class Table3 {
public:
  Table3(const char* name, const char* title, bool isMC);
  void Fill(const RHyperTriton3KF& RHyperVec, const REvent3KF& RColl, const SHyperTriton3KF* SHyper = nullptr);
  void Write() { tree->Write(); }

private:
  TTree* tree;
  float m;
  float pt;
  float ct;
  float centrality;
  float rapidity;
  float cosPA;
  float chi2_deuprot;
  float chi2_3prongs;
  float chi2_topology;
  float dca_de;
  float dca_pr;
  float dca_pi;
  float tpc_nsig_de;
  float tpc_nsig_pr;
  float tpc_nsig_pi;
  float tof_nsig_de;
  float tof_nsig_pr;
  float tof_nsig_pi;
  float dca_de_pr;
  float dca_de_pi;
  float dca_pr_pi;
  bool has_tof_de;
  bool has_tof_pr;
  bool has_tof_pi;
  UChar_t tpc_nclus_de;
  UChar_t tpc_nclus_pr;
  UChar_t tpc_nclus_pi;

  float mc_pt;
  float mc_ct;
};

Table3::Table3(const char* name, const char* title, bool isMC) {
  tree = new TTree(name, title);

  tree->Branch("m", &m);
  tree->Branch("pt", &pt);
  tree->Branch("ct", &ct);
  tree->Branch("centrality", &centrality);
  tree->Branch("rapidity", &rapidity);
  tree->Branch("cosPA", &cosPA);
  tree->Branch("chi2_deuprot", &chi2_deuprot);
  tree->Branch("chi2_3prongs", &chi2_3prongs);
  tree->Branch("chi2_topology", &chi2_topology);
  tree->Branch("dca_de", &dca_de);
  tree->Branch("dca_pr", &dca_pr);
  tree->Branch("dca_pi", &dca_pi);
  tree->Branch("tpc_nsig_de", &tpc_nsig_de);
  tree->Branch("tpc_nsig_pr", &tpc_nsig_pr);
  tree->Branch("tpc_nsig_pi", &tpc_nsig_pi);
  tree->Branch("tof_nsig_de", &tof_nsig_de);
  tree->Branch("tof_nsig_pr", &tof_nsig_pr);
  tree->Branch("tof_nsig_pi", &tof_nsig_pi);
  tree->Branch("dca_de_pr", &dca_de_pr);
  tree->Branch("dca_de_pi", &dca_de_pi);
  tree->Branch("dca_pr_pi", &dca_pr_pi);
  tree->Branch("has_tof_de", &has_tof_de);
  tree->Branch("has_tof_pr", &has_tof_pr);
  tree->Branch("has_tof_pi", &has_tof_pi);
  tree->Branch("tpc_nclus_de", &tpc_nclus_de);
  tree->Branch("tpc_nclus_pr", &tpc_nclus_pr);
  tree->Branch("tpc_nclus_pi", &tpc_nclus_pi);
  if (isMC) {
    tree->Branch("mc_ct", &mc_ct);
    tree->Branch("mc_pt", &mc_pt);
  }
};

void Table3::Fill(const RHyperTriton3KF& RHyper, const REvent3KF& RColl, const SHyperTriton3KF* SHyper) {
  centrality = RColl.fCent;

  pt             = std::abs(RHyper.pt);
  const double p = std::hypot(pt, RHyper.pz);
  const double e = std::hypot(kHyperTritonMass, p);
  rapidity       = 0.5 * std::log((e + RHyper.pz) / (e - RHyper.pz + 1.e-12));
  ct             = RHyper.l * kHyperTritonMass / p;
  m              = RHyper.m;
  cosPA          = RHyper.cosPA;
  chi2_deuprot   = RHyper.chi2_deuprot;
  chi2_3prongs   = RHyper.chi2_3prongs;
  chi2_topology  = RHyper.chi2_topology;
  dca_de         = RHyper.dca_de;
  dca_pr         = RHyper.dca_pr;
  dca_pi         = RHyper.dca_pi;
  tpc_nsig_de    = RHyper.tpcNsig_de;
  tpc_nsig_pr    = RHyper.tpcNsig_pr;
  tpc_nsig_pi    = RHyper.tpcNsig_pi;
  tof_nsig_de    = RHyper.tofNsig_de;
  tof_nsig_pr    = RHyper.tofNsig_pr;
  tof_nsig_pi    = RHyper.tofNsig_pi;
  dca_de_pr      = RHyper.dca_de_pr;
  dca_de_pi      = RHyper.dca_de_pi;
  dca_pr_pi      = RHyper.dca_pr_pi;
  has_tof_de     = RHyper.hasTOF_de;
  has_tof_pr     = RHyper.hasTOF_pr;
  has_tof_pi     = RHyper.hasTOF_pi;
  tpc_nclus_de   = RHyper.tpcClus_de;
  tpc_nclus_pr   = RHyper.tpcClus_pr;
  tpc_nclus_pi   = RHyper.tpcClus_pi;

  if (SHyper) {
    mc_pt       = SHyper->pt;
    double mc_p = std::hypot(mc_pt, SHyper->pz);
    mc_ct       = SHyper->l * kHyperTritonMass / mc_p;
  }

  if (chi2_deuprot < 0 || chi2_deuprot > 50.) return;
  if (chi2_3prongs < 0 || chi2_3prongs > 50.) return;
  if (chi2_topology < 0 || chi2_topology > 150.) return;

  if (tpc_nsig_de > 3.) return;
  if (tpc_nsig_pr > 3.) return;
  if (tpc_nsig_pi > 3.) return;

  if (!has_tof_de) return;
  if (tof_nsig_de > 4.) return;

  if (dca_de < 0.05) return;
  if (dca_pr < 0.05) return;
  if (dca_pi < 0.05) return;

  if (cosPA < 0.99) return;

  tree->Fill();
}

#endif
