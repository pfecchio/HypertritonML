#ifndef HYPERTABLE_H
#define HYPERTABLE_H

#include <TTree.h>

#include "AliAnalysisTaskHypertriton3.h"


class TableO2
{
public:
  TableO2(bool isMC);
  void Fill(const RHyperTriton3O2 &RHyper);
  void Fill(const SHyperTriton3O2 &RHyper, TF1 *blastWave[3], double max[3]);
  // void SetName(const char *name);
  bool AcceptCandidateBW(float gPt, float centrality, TF1 *blastWave[3], double max[3]);
  void Write() { tree->Write("", TObject::kOverwrite); }

private:
  TTree *tree;
  float centrality;
  float pt;
  float ct;
  float m;
  float cos_pa;
  float dca_de;
  float dca_pr;
  float dca_pi;
  float tpc_nsig_de;
  float tpc_nsig_pr;
  float tpc_nsig_pi;
  float tof_nsig_de;
  float tof_nsig_pr;
  float dca_de_pr;
  float dca_de_pi;
  float dca_pr_pi;
  float dca_de_sv;
  float dca_pr_sv;
  float dca_pi_sv;
  float chi2;
  float tpc_ncls_de;
  float tpc_ncls_pr;
  float tpc_ncls_pi;
  float has_tof_de;
  float has_tof_pr;
  float has_tof_pi;
  bool positive;

  float cos_pa_lambda;
  float mppi_vert;
  float dca_lambda_hyper;

  // mc only variables
  float gPt;
  float gY;
  float gPhi;
  float gCt;
  float gT;
  bool gPositive;
  bool gReconstructed;
  bool bw_accept;
};

TableO2::TableO2(bool isMC)
{
  if (isMC) {
    tree = new TTree("SignalTable", "SignalTable");
  } else {
    tree = new TTree("DataTable", "DataTable");
  }

  tree->Branch("centrality", &centrality);
  tree->Branch("pt", &pt);
  tree->Branch("m", &m);
  tree->Branch("ct", &ct);
  tree->Branch("cos_pa", &cos_pa);
  tree->Branch("dca_de", &dca_de);
  tree->Branch("dca_pr", &dca_pr);
  tree->Branch("dca_pi", &dca_pi);
  tree->Branch("tpc_nsig_de", &tpc_nsig_de);
  tree->Branch("tpc_nsig_pr", &tpc_nsig_pr);
  tree->Branch("tpc_nsig_pi", &tpc_nsig_pi);
  tree->Branch("tof_nsig_de", &tof_nsig_de);
  tree->Branch("tof_nsig_pr", &tof_nsig_pr);
  tree->Branch("dca_de_pr", &dca_de_pr);
  tree->Branch("dca_de_pi", &dca_de_pi);
  tree->Branch("dca_pr_pi", &dca_pr_pi);
  tree->Branch("dca_de_sv",&dca_de_sv);
  tree->Branch("dca_pr_sv",&dca_pr_sv);
  tree->Branch("dca_pi_sv",&dca_pi_sv);
  tree->Branch("chi2",&chi2);
  tree->Branch("tpc_ncls_de", &tpc_ncls_de);
  tree->Branch("tpc_ncls_pr", &tpc_ncls_pr);
  tree->Branch("tpc_ncls_pi", &tpc_ncls_pi);
  tree->Branch("has_tof_de", &has_tof_de);
  tree->Branch("has_tof_pr", &has_tof_pr);
  tree->Branch("has_tof_pi", &has_tof_pi);
  tree->Branch("positive", &positive);

  tree->Branch("cos_pa_lambda", &cos_pa_lambda);
  tree->Branch("mppi_vert", &mppi_vert);
  tree->Branch("dca_lambda_hyper", &dca_lambda_hyper);

  if (isMC)
  {
    tree->Branch("gPt", &gPt);
    tree->Branch("gY", &gY);
    tree->Branch("gPhi", &gPhi);
    tree->Branch("gCt", &gCt);
    tree->Branch("gT", &gT);
    tree->Branch("gPositive", &gPositive);
    tree->Branch("gReconstructed", &gReconstructed);
    tree->Branch("bw_accept", &bw_accept);
  }
};

void TableO2::Fill(const RHyperTriton3O2 &RHyper)
{
  if (RHyper.cosPA < 0) return;

  centrality = RHyper.centrality;
  pt = RHyper.pt;
  positive = RHyper.pt > 0;
  m = RHyper.m;
  ct = RHyper.ct;
  cos_pa = RHyper.cosPA;
  dca_de = RHyper.dca_de;
  dca_pr = RHyper.dca_pr;
  dca_pi = RHyper.dca_pi;
  tpc_nsig_de = RHyper.tpcNsig_de;
  tpc_nsig_pr = RHyper.tpcNsig_pr;
  tpc_nsig_pi = RHyper.tpcNsig_pi;
  tof_nsig_de = RHyper.tofNsig_de;
  tof_nsig_pr = RHyper.tofNsig_pr;
  dca_de_pr = RHyper.dca_de_pr;
  dca_de_pi = RHyper.dca_de_pi;
  dca_pr_pi = RHyper.dca_pr_pi;
  dca_de_sv = RHyper.dca_de_sv;
  dca_pr_sv = RHyper.dca_pr_sv;
  dca_pi_sv = RHyper.dca_pi_sv;
  chi2 = RHyper.chi2;
  tpc_ncls_de = RHyper.tpcClus_de;
  tpc_ncls_pr = RHyper.tpcClus_pr;
  tpc_ncls_pi = RHyper.tpcClus_pi;
  has_tof_de = RHyper.hasTOF_de;
  has_tof_pr = RHyper.hasTOF_pr;
  has_tof_pi = RHyper.hasTOF_pi;

  cos_pa_lambda = RHyper.cosPA_Lambda;
  mppi_vert = RHyper.mppi_vert;
  dca_lambda_hyper = RHyper.dca_lambda_hyper;

  tree->Fill();
}

void TableO2::Fill(const SHyperTriton3O2 &SHyper, TF1 *blastWave[3], double max[3])
{
  gPt = SHyper.gPt;
  gPhi = SHyper.gPhi;
  const double p = std::hypot(SHyper.gPt, gPt);
  const double e = std::hypot(p, kHyperTritonMass);
  gY = 0.5 * std::log((e + SHyper.gPz) / (e - SHyper.gPz + 1.e-12));
  gCt = SHyper.gCt;
  gT = SHyper.gT;
  gPositive = SHyper.gPositive;
  gReconstructed = SHyper.gReconstructed;
  centrality = SHyper.centrality;
  bw_accept = AcceptCandidateBW(gPt, centrality, blastWave, max);

  pt = SHyper.pt;
  positive = SHyper.pt > 0;
  m = SHyper.m;
  ct = SHyper.ct;
  cos_pa = SHyper.cosPA;
  dca_de = SHyper.dca_de;
  dca_pr = SHyper.dca_pr;
  dca_pi = SHyper.dca_pi;
  tpc_nsig_de = SHyper.tpcNsig_de;
  tpc_nsig_pr = SHyper.tpcNsig_pr;
  tpc_nsig_pi = SHyper.tpcNsig_pi;
  tof_nsig_de = SHyper.tofNsig_de;
  tof_nsig_pr = SHyper.tofNsig_pr;
  dca_de_pr = SHyper.dca_de_pr;
  dca_de_pi = SHyper.dca_de_pi;
  dca_pr_pi = SHyper.dca_pr_pi;
  dca_de_sv = SHyper.dca_de_sv;
  dca_pr_sv = SHyper.dca_pr_sv;
  dca_pi_sv = SHyper.dca_pi_sv;
  chi2 = SHyper.chi2;
  tpc_ncls_de = SHyper.tpcClus_de;
  tpc_ncls_pr = SHyper.tpcClus_pr;
  tpc_ncls_pi = SHyper.tpcClus_pi;
  has_tof_de = SHyper.hasTOF_de;
  has_tof_pr = SHyper.hasTOF_pr;
  has_tof_pi = SHyper.hasTOF_pi;

  cos_pa_lambda = SHyper.cosPA_Lambda;
  mppi_vert = SHyper.mppi_vert;
  dca_lambda_hyper = SHyper.dca_lambda_hyper;

  tree->Fill();
}

bool TableO2::AcceptCandidateBW(float gPt, float centrality, TF1 *blastWave[3], double max[3])
{

  TF1 *BlastWave{nullptr};
  double maximum  = 0.0;
  int index = 2;
  if (centrality <= 10) {
    index = 0;
  } else if (centrality <= 40.) {
    index = 1;
  } 

  BlastWave = blastWave[index];
  maximum   = max[index];

  float bwNum = BlastWave->Eval(gPt) / maximum;

  if (bwNum < gRandom->Rndm()) {
    return false;
  }
  return true;
}

#endif