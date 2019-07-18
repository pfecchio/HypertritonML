#ifndef HYPERTABLE_H
#define HYPERTABLE_H

#include "Common.h"

#include <string>

#include <TLorentzVector.h>
#include <TTree.h>
#include <TVector3.h>

#include "AliAnalysisTaskHyperTriton2He3piML.h"

class Table {
  public: 
  Table(std::string name, std::string title);
  void Fill(const RHyperTritonHe3pi& RHyperVec, const RCollision& RColl);
  void Write() { tree->Write(); }

  private:
  TTree* tree;
  float V0pt;
  float TPCnSigmaHe3;
  float DistOverP;
  float InvMass;
  float ArmenterosAlpha;
  float V0CosPA;
  float V0Chi2;
  float PiProngPt;
  float He3ProngPt;
  float ProngsDCA;
  float PiProngPvDCA;
  float He3ProngPvDCA;
  float NpidClustersHe3;
  float TPCnSigmaPi;
  float Lrec;
  float Centrality;
};

Table::Table(std::string name, std::string title) {
  tree = new TTree(name.data(), title.data());

  tree->Branch("V0pt", &V0pt);
  tree->Branch("TPCnSigmaHe3", &TPCnSigmaHe3);
  tree->Branch("DistOverP", &DistOverP);
  tree->Branch("InvMass", &InvMass);
  tree->Branch("ArmenterosAlpha", &ArmenterosAlpha);
  tree->Branch("V0CosPA", &V0CosPA);
  tree->Branch("V0Chi2", &V0Chi2);
  tree->Branch("PiProngPt", &PiProngPt);
  tree->Branch("He3ProngPt", &He3ProngPt);
  tree->Branch("ProngsDCA", &ProngsDCA);
  tree->Branch("He3ProngPvDCA", &He3ProngPvDCA);
  tree->Branch("PiProngPvDCA", &PiProngPvDCA);
  tree->Branch("NpidClustersHe3", &NpidClustersHe3);
  tree->Branch("TPCnSigmaPi", &TPCnSigmaPi);
  tree->Branch("Lrec", &Lrec);
  tree->Branch("Centrality", &Centrality);
};

void Table::Fill(const RHyperTritonHe3pi& RHyper, const RCollision& RColl) {
  Centrality = RColl.fCent;
  double eHe3 = Hypot(RHyper.fPxHe3, RHyper.fPyHe3, RHyper.fPzHe3, AliPID::ParticleMass(AliPID::kHe3));
  double ePi = Hypot(RHyper.fPxPi, RHyper.fPyPi, RHyper.fPzPi, AliPID::ParticleMass(AliPID::kPion));

  TLorentzVector he3Vector, piVector, hyperVector;
  he3Vector.SetPxPyPzE(RHyper.fPxHe3, RHyper.fPyHe3, RHyper.fPzHe3, eHe3);
  piVector.SetPxPyPzE(RHyper.fPxPi, RHyper.fPyPi, RHyper.fPzPi, ePi);
  hyperVector = piVector + he3Vector;

  TVector3 v(RHyper.fDecayX, RHyper.fDecayY, RHyper.fDecayZ);
  float pointAngle = hyperVector.Angle(v);
  float CosPA = std::cos(pointAngle);

  float qP, qN, qT;
  if (RHyper.fMatter == true)
  {
    qP = SProd(hyperVector, he3Vector) / fabs(hyperVector.P());
    qN = SProd(hyperVector, piVector) / fabs(hyperVector.P());
    qT = VProd(hyperVector, he3Vector) / fabs(hyperVector.P());
  }
  else
  {
    qN = SProd(hyperVector, he3Vector) / fabs(hyperVector.P());
    qP = SProd(hyperVector, piVector) / fabs(hyperVector.P());
    qT = VProd(hyperVector, piVector) / fabs(hyperVector.P());
  }

  float alpha = (qP - qN) / (qP + qN);
  DistOverP = Hypot(RHyper.fDecayX, RHyper.fDecayY, RHyper.fDecayZ) / hyperVector.P();
  InvMass = hyperVector.M();
  ArmenterosAlpha = alpha;
  V0CosPA = CosPA;
  V0Chi2 = RHyper.fChi2V0;
  PiProngPt = Hypot(RHyper.fPxPi, RHyper.fPyPi);
  He3ProngPt = Hypot(RHyper.fPxHe3, RHyper.fPyHe3);
  ProngsDCA = RHyper.fDcaV0daughters;
  PiProngPvDCA = RHyper.fDcaPi2PrimaryVertex;
  He3ProngPvDCA = RHyper.fDcaHe32PrimaryVertex;
  Lrec = Hypot(RHyper.fDecayX, RHyper.fDecayY, RHyper.fDecayZ);
  NpidClustersHe3 = RHyper.fNpidClustersHe3;
  TPCnSigmaPi = RHyper.fTPCnSigmaPi;
  TPCnSigmaHe3 = RHyper.fTPCnSigmaHe3;
  V0pt = hyperVector.Pt();
  tree->Fill();
}

#endif