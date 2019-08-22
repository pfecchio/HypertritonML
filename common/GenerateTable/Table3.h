#ifndef HYPERTABLE3_H
#define HYPERTABLE3_H

#include "Common.h"

#include <string>

#include <TLorentzVector.h>
#include <TTree.h>
#include <TVector3.h>

#include "AliAnalysisTaskHypertriton3ML.h"
#include "AliPID.h"

class Table3 {
public:
  Table3(std::string name, std::string title);
  void Fill(const RHypertriton3 &rHyperVec, const REvent &rColl);
  void Write() { tree->Write(); }

private:
  TTree *tree;

  float fCentrality;
  float fHypCandPt;
  float fPtDeu;
  float fPtP;
  float fPtPi;
  float nClsTPCDeu;
  float nClsTPCP;
  float nClsTPCPi;
  float nClsITSDeu;
  float nClsITSP;
  float nClsITSPi;
  float fNSigmaTPCDeu;
  float fNSigmaTPCP;
  float fNSigmaTPCPi;
  float fNSigmaTOFDeu;
  float fNSigmaTOFP;
  float fNSigmaTOFPi;
  float fTrackChi2Deu;
  float fTrackChi2P;
  float fTrackChi2Pi;
  float fDecayVertexChi2NDF;
  float fDCAxyPrimaryVtxDeu;
  float fDCAxyPrimaryVtxP;
  float fDCAxyPrimaryVtxPi;
  float fDCAzPrimaryVtxDeu;
  float fDCAzPrimaryVtxP;
  float fDCAzPrimaryVtxPi;
  float fDCAPrimaryVtxDeu;
  float fDCAPrimaryVtxP;
  float fDCAPrimaryVtxPi;
  float fDCAxyDecayVtxDeu;
  float fDCAxyDecayVtxP;
  float fDCAxyDecayVtxPi;
  float fDCAzDecayVtxDeu;
  float fDCAzDecayVtxP;
  float fDCAzDecayVtxPi;
  float fDCADecayVtxDeu;
  float fDCADecayVtxP;
  float fDCADecayVtxPi;
  float fTrackDistDeuP;
  float fTrackDistPPi;
  float fTrackDistDeuPi;
  float fCosPA;
  float fDistOverP;
  bool fMatter;
};

Table3::Table3(std::string name, std::string title) {
  tree = new TTree(name.data(), title.data());

  tree->Branch("Centrality", &fCentrality);
  tree->Branch("HypCandPt", &fHypCandPt);
  tree->Branch("PtDeu", &fPtDeu);
  tree->Branch("PtP", &fPtP);
  tree->Branch("PtPi", &fPtPi);
  tree->Branch("nClsTPCDeu", &nClsTPCDeu);
  tree->Branch("nClsTPCP", &nClsTPCP);
  tree->Branch("nClsTPCPi", &nClsTPCPi);
  tree->Branch("nClsITSDeu", &nClsITSDeu);
  tree->Branch("nClsITSP", &nClsITSP);
  tree->Branch("nClsITSPi", &nClsITSPi);
  tree->Branch("nSigmaTPCDeu", &fNSigmaTPCDeu);
  tree->Branch("nSigmaTPCP", &fNSigmaTPCP);
  tree->Branch("nSigmaTPCPi", &fNSigmaTPCPi);
  tree->Branch("nSigmaTOFDeu", &fNSigmaTOFDeu);
  tree->Branch("nSigmaTOFP", &fNSigmaTOFP);
  tree->Branch("nSigmaTOFPi", &fNSigmaTOFPi);
  tree->Branch("trackChi2Deu", &fTrackChi2Deu);
  tree->Branch("trackChi2P", &fTrackChi2P);
  tree->Branch("trackChi2Pi", &fTrackChi2Pi);
  tree->Branch("vertexChi2", &fDecayVertexChi2NDF);
  tree->Branch("DCA2xyPrimaryVtxDeu", &fDCAxyPrimaryVtxDeu);
  tree->Branch("DCAxyPrimaryVtxP", &fDCAxyPrimaryVtxP);
  tree->Branch("DCAxyPrimaryVtxPi", &fDCAxyPrimaryVtxPi);
  tree->Branch("DCAzPrimaryVtxDeu", &fDCAzPrimaryVtxDeu);
  tree->Branch("DCAzPrimaryVtxP", &fDCAzPrimaryVtxP);
  tree->Branch("DCAzPrimaryVtxPi", &fDCAzPrimaryVtxPi);
  tree->Branch("DCAPrimaryVtxDeu", &fDCAPrimaryVtxDeu);
  tree->Branch("DCAPrimaryVtxP", &fDCAPrimaryVtxP);
  tree->Branch("DCAPrimaryVtxPi", &fDCAPrimaryVtxPi);
  tree->Branch("DCAxyDecayVtxDeu", &fDCAxyDecayVtxDeu);
  tree->Branch("DCAxyDecayVtxP", &fDCAxyDecayVtxP);
  tree->Branch("DCAxyDecayVtxPi", &fDCAxyDecayVtxPi);
  tree->Branch("DCAzDecayVtxDeu", &fDCAzDecayVtxDeu);
  tree->Branch("DCAzDecayVtxP", &fDCAzDecayVtxP);
  tree->Branch("DCAzDecayVtxPi", &fDCAzDecayVtxPi);
  tree->Branch("DCADecayVtxDeu", &fDCADecayVtxDeu);
  tree->Branch("DCADecayVtxP", &fDCADecayVtxP);
  tree->Branch("DCADecayVtxPi", &fDCADecayVtxPi);
  tree->Branch("TrackDistDeuP", &fTrackDistDeuP);
  tree->Branch("TrackDistPPi", &fTrackDistPPi);
  tree->Branch("TrackDistDeuPi", &fTrackDistDeuPi);
  tree->Branch("CosPA", &fCosPA);
  tree->Branch("DistOverP", &fDistOverP);
  tree->Branch("Matter", &fMatter);
};

void Table3::Fill(const RHypertriton3 &rHyp3, const REvent &rEv) {
  // get gentrality of the collision
  fCentrality = rEv.fCent;

  // get primary vertex position
  const TVector3 primaryVtxPos(rEv.fX, rEv.fY, rEv.fZ);

  // n cluster TPC e clustermap ITS
  nClsTPCDeu = rHyp3.fNClusterTPCDeu;
  nClsTPCP   = rHyp3.fNClusterTPCP;
  nClsTPCPi  = rHyp3.fNClusterTPCPi;

  // n cluster ITS from ITS clustermap
  nClsITSDeu = GetNClsITS(rHyp3.fITSClusterMapDeu);
  nClsITSP   = GetNClsITS(rHyp3.fITSClusterMapP);
  nClsITSPi  = GetNClsITS(rHyp3.fITSClusterMapPi);

  // PID with TPC and TOF
  fNSigmaTPCDeu = rHyp3.fNSigmaTPCDeu;
  fNSigmaTPCP   = rHyp3.fNSigmaTPCP;
  fNSigmaTPCPi  = rHyp3.fNSigmaTPCPi;
  fNSigmaTOFDeu = rHyp3.fNSigmaTOFDeu;
  fNSigmaTOFP   = rHyp3.fNSigmaTOFP;
  fNSigmaTOFPi  = rHyp3.fNSigmaTOFPi;

  // tracks and decay vertex chi2
  fTrackChi2Deu       = rHyp3.fTrackChi2Deu;
  fTrackChi2P         = rHyp3.fTrackChi2P;
  fTrackChi2Pi        = rHyp3.fTrackChi2Pi;
  fDecayVertexChi2NDF = rHyp3.fDecayVertexChi2NDF;

  // daughter's DCA to primary vertex
  fDCAxyPrimaryVtxDeu = rHyp3.fDCAxyDeu;
  fDCAxyPrimaryVtxP   = rHyp3.fDCAxyP;
  fDCAxyPrimaryVtxPi  = rHyp3.fDCAxyPi;
  fDCAzPrimaryVtxDeu  = rHyp3.fDCAzDeu;
  fDCAzPrimaryVtxP    = rHyp3.fDCAzP;
  fDCAzPrimaryVtxPi   = rHyp3.fDCAzPi;
  fDCAPrimaryVtxDeu   = Hypot(rHyp3.fDCAxyDeu, rHyp3.fDCAzDeu);
  fDCAPrimaryVtxP     = Hypot(rHyp3.fDCAxyP, rHyp3.fDCAzP);
  fDCAPrimaryVtxPi    = Hypot(rHyp3.fDCAxyPi, rHyp3.fDCAzPi);

  // vectors of the closest position to the decay vertex of the daughter's tracks
  const TVector3 decayVtxPos(rHyp3.fDecayVtxX, rHyp3.fDecayVtxY, rHyp3.fDecayVtxZ);
  const TVector3 deuPosVector(rHyp3.fPosXDeu, rHyp3.fPosYDeu, rHyp3.fPosZDeu);
  const TVector3 pPosVector(rHyp3.fPosXP, rHyp3.fPosYP, rHyp3.fPosZP);
  const TVector3 piPosVector(rHyp3.fPosXPi, rHyp3.fPosYPi, rHyp3.fPosZPi);

  // DCA to the decay vertex
  fDCAxyDecayVtxDeu = DistanceXY(decayVtxPos, deuPosVector);
  fDCAxyDecayVtxP   = DistanceXY(decayVtxPos, pPosVector);
  fDCAxyDecayVtxPi  = DistanceXY(decayVtxPos, piPosVector);
  fDCAzDecayVtxDeu  = DistanceZ(decayVtxPos, deuPosVector);
  fDCAzDecayVtxP    = DistanceZ(decayVtxPos, pPosVector);
  fDCAzDecayVtxPi   = DistanceZ(decayVtxPos, piPosVector);
  fDCADecayVtxDeu   = Distance3D(decayVtxPos, deuPosVector);
  fDCADecayVtxP     = Distance3D(decayVtxPos, pPosVector);
  fDCADecayVtxPi    = Distance3D(decayVtxPos, piPosVector);

  // compute the 4-vector of the daughter tracks
  const double eDeu = Hypot(rHyp3.fPxDeu, rHyp3.fPyDeu, rHyp3.fPzDeu, kDeuMass);
  const double eP   = Hypot(rHyp3.fPxP, rHyp3.fPyP, rHyp3.fPzP, kPMass);
  const double ePi  = Hypot(rHyp3.fPxPi, rHyp3.fPyPi, rHyp3.fPzPi, kPiMass);

  const TLorentzVector deu4Vector{rHyp3.fPxDeu, rHyp3.fPyDeu, rHyp3.fPzDeu, eDeu};
  const TLorentzVector p4Vector{rHyp3.fPxP, rHyp3.fPyP, rHyp3.fPzP, eP};
  const TLorentzVector pi4Vector{rHyp3.fPxPi, rHyp3.fPyPi, rHyp3.fPzPi, ePi};

  // pT of the daughter particles
  fPtDeu = deu4Vector.Pt();
  fPtP   = p4Vector.Pt();
  fPtPi  = pi4Vector.Pt();

  // compute the distance between the daughter tracks at the decay vertex
  const float deuPos[3] = {rHyp3.fPosXDeu, rHyp3.fPosYDeu, rHyp3.fPosZDeu};
  const float pPos[3]   = {rHyp3.fPosXP, rHyp3.fPosYP, rHyp3.fPosZP};
  const float piPos[3]  = {rHyp3.fPosXPi, rHyp3.fPosYPi, rHyp3.fPosZPi};

  fTrackDistDeuP  = Distance3D(deuPos, pPos);
  fTrackDistPPi   = Distance3D(pPos, piPos);
  fTrackDistDeuPi = Distance3D(deuPos, piPos);

  // compute the 4-vector of the hypertriton candidate
  const TLorentzVector hyper4Vector = deu4Vector + p4Vector + pi4Vector;
  fHypCandPt                        = hyper4Vector.Pt();

  // define the decay lenght vector
  const TVector3 decayLenghtVector = decayVtxPos - primaryVtxPos;

  // compute the cos(theta pointing)
  fCosPA = std::cos(hyper4Vector.Angle(decayLenghtVector));

  // compute dist/P
  fDistOverP = decayLenghtVector.Mag() / hyper4Vector.P();

  // matter or anti-matter
  fMatter = rHyp3.fIsMatter;

  // fill the tree
  tree->Fill();
}

#endif