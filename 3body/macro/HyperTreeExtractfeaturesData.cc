#include <iostream>
#include <vector>

#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include "AliAnalysisTaskHypertriton3ML.h"

template <typename T> double Pot2(T a) { return a * a; }

template <typename T> double Hypot(T a, T b, T c, T d) { return std::sqrt(Pot2(a) + Pot2(b) + Pot2(c) + Pot2(d)); }

template <typename T> double DistanceZ(T v1, T v2) { return std::sqrt(Pot2(v1[2] - v2[2])); }

template <typename T> double DistanceXY(T v1, T v2) { return std::sqrt(Pot2(v1[0] - v2[0]) + Pot2(v1[1] - v2[1])); }

template <typename T> double Distance3D(T v1, T v2) {
  return std::sqrt(Pot2(v1[0] - v2[0]) + Pot2(v1[1] - v2[1]) + Pot2(v1[2] - v2[2]));
}

int GetNClsITS(unsigned char clsMap) {
  int ncls = 0;

  for (int i = 0; i < 6; i++) {
    ncls += (int)(clsMap >> i) & 1;
  }

  return ncls;
}

void HyperTreeExtractfeaturesData() {

  // read the tree
  TFile *inFile = new TFile("~/data/3body_hypertriton_data/train_output/data/LHC18r_pass1/HyperTritonTreeData.root", "read");

  TTreeReader myReader("fHypertritonTree", inFile);
  TTreeReaderValue<REvent> rEv          = {myReader, "REvent"};
  TTreeReaderArray<RHypertriton3> rHyp3 = {myReader, "RHypertriton"};

  // new tree with the features
  TFile outFile("~/data/3body_hypertriton_data/training_set/r/Hyper3BackgroundTree.root", "RECREATE");
  TTree *tree = new TTree("Hyper3Background", "");

  float fCentrality;

  float fPtDeu;
  float fPtP;
  float fPtPi;

  float fDistOverP;

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

  tree->Branch("Centrality", &fCentrality);
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

  while (myReader.Next()) {
    // centrality
    fCentrality = rEv->fCent;
    // primary vertex position
    TVector3 primaryVtxPos(rEv->fX, rEv->fY, rEv->fZ);

    for (auto &rec : rHyp3) {
      // n cluster TPC e clustermap ITS
      nClsTPCDeu = rec.fNClusterTPCDeu;
      nClsTPCP   = rec.fNClusterTPCP;
      nClsTPCPi  = rec.fNClusterTPCPi;
      // n cluster ITS from ITS clustermap
      nClsITSDeu = GetNClsITS(rec.fITSClusterMapDeu);
      nClsITSP   = GetNClsITS(rec.fITSClusterMapP);
      nClsITSPi  = GetNClsITS(rec.fITSClusterMapPi);
      // PID with TPC and TOF
      fNSigmaTPCDeu = rec.fNSigmaTPCDeu;
      fNSigmaTPCP   = rec.fNSigmaTPCP;
      fNSigmaTPCPi  = rec.fNSigmaTPCPi;
      fNSigmaTOFDeu = rec.fNSigmaTOFDeu;
      fNSigmaTOFP   = rec.fNSigmaTOFP;
      fNSigmaTOFPi  = rec.fNSigmaTOFPi;
      // tracks and decay vertex chi2
      fTrackChi2Deu       = rec.fTrackChi2Deu;
      fTrackChi2P         = rec.fTrackChi2P;
      fTrackChi2Pi        = rec.fTrackChi2Pi;
      fDecayVertexChi2NDF = rec.fDecayVertexChi2NDF;
      // daughter's DCA to primary vertex
      fDCAxyPrimaryVtxDeu = rec.fDCAxyDeu;
      fDCAxyPrimaryVtxP   = rec.fDCAxyP;
      fDCAxyPrimaryVtxPi  = rec.fDCAxyPi;
      fDCAzPrimaryVtxDeu  = rec.fDCAzDeu;
      fDCAzPrimaryVtxP    = rec.fDCAzP;
      fDCAzPrimaryVtxPi   = rec.fDCAzPi;
      fDCAPrimaryVtxDeu   = std::hypot(rec.fDCAxyDeu, rec.fDCAzDeu);
      fDCAPrimaryVtxP     = std::hypot(rec.fDCAxyP, rec.fDCAzP);
      fDCAPrimaryVtxPi    = std::hypot(rec.fDCAxyPi, rec.fDCAzPi);

      // vectors of the closest position to the decay vertex of the daughter's tracks
      TVector3 decayVtxPos(rec.fDecayVtxX, rec.fDecayVtxY, rec.fDecayVtxZ);
      TVector3 deuPosVector(rec.fPosXDeu, rec.fPosYDeu, rec.fPosZDeu);
      TVector3 pPosVector(rec.fPosXP, rec.fPosYP, rec.fPosZP);
      TVector3 piPosVector(rec.fPosXPi, rec.fPosYPi, rec.fPosZPi);
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
      double eDeu = Hypot(rec.fPxDeu, rec.fPyDeu, rec.fPzDeu, AliPID::ParticleMass(AliPID::kDeuteron));
      double eP   = Hypot(rec.fPxP, rec.fPyP, rec.fPzP, AliPID::ParticleMass(AliPID::kProton));
      double ePi  = Hypot(rec.fPxPi, rec.fPyPi, rec.fPzPi, AliPID::ParticleMass(AliPID::kPion));

      TLorentzVector deuVector, pVector, piVector, hyperVector;
      deuVector.SetPxPyPzE(rec.fPxDeu, rec.fPyDeu, rec.fPzDeu, eDeu);
      pVector.SetPxPyPzE(rec.fPxP, rec.fPyP, rec.fPzP, eP);
      piVector.SetPxPyPzE(rec.fPxPi, rec.fPyPi, rec.fPzPi, ePi);

      fPtDeu = deuVector.Pt();
      fPtP   = pVector.Pt();
      fPtPi  = piVector.Pt();

      float deuPos[3] = {rec.fPosXDeu, rec.fPosYDeu, rec.fPosZDeu};
      float pPos[3]   = {rec.fPosXP, rec.fPosYP, rec.fPosZP};
      float piPos[3]  = {rec.fPosXPi, rec.fPosYPi, rec.fPosZPi};

      fTrackDistDeuP  = Distance3D(deuPos, pPos);
      fTrackDistPPi   = Distance3D(pPos, piPos);
      fTrackDistDeuPi = Distance3D(deuPos, piPos);

      hyperVector = deuVector + pVector + piVector;

      TVector3 decayLenghtVector = decayVtxPos - primaryVtxPos;

      fCosPA = std::cos(hyperVector.Angle(decayLenghtVector));

      // decay lenght over momentum TODO: che unità di misura sono?
      fDistOverP = decayLenghtVector.Mag() / hyperVector.P();
      // fill the tree
      tree->Fill();
    }
  }

  outFile.Write("", TObject::kOverwrite);
  inFile->Close();
  inFile->Delete();
  outFile.Close();
  outFile.Delete();
}
