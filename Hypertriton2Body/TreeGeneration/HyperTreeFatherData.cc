#include "AliAnalysisTaskHyperTriton2He3piML.h"
#include "AliPID.h"
#include "Math/Vector4D.h"
#include "TH1D.h"
#include <Riostream.h>
#include <TCanvas.h>
#include <TDirectoryFile.h>
#include <TFile.h>
#include <TList.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <TROOT.h>
#include <TRandom3.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

float SProd(TLorentzVector a1, TLorentzVector a2) { return fabs(a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2]); }

float Sq(float a) { return a * a; }

float VProd(TLorentzVector a1, TLorentzVector a2)
{
  float x = a1[1] * a2[2] - a2[1] * a1[2];
  float y = a1[2] * a2[0] - a1[0] * a2[2];
  float z = a1[0] * a2[1] - a1[1] * a2[0];
  return TMath::Sqrt(x * x + y * y + z * z);
}

template <typename F>
double Hypot(F a, F b) { return std::sqrt(a * a + b * b); }

template <typename F>
double Hypot(F a, F b, F c) { return std::sqrt(a * a + b * b + c * c); }

template <typename F>
double Hypot(F a, F b, F c, F d) { return std::sqrt(a * a + b * b + c * c + d * d); }

void HyperTreeFatherData()
{
  TFile *myFile = TFile::Open("~/HypertritonData/HyperTritonTree_18r.root", "r");
  TDirectoryFile *mydir = (TDirectoryFile *)myFile->Get("_custom");
  TTreeReader fReader("fTreeV0", mydir);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};

  // TH1D *fHistCent = new TH1D("fHistCent", "", 1000, 0, 100);
  // fHistCent->SetDirectory(0);

  TFile tfileHist("CentHist.root", "READ");
  TH1D *fHistCent = (TH1D *)tfileHist.Get("fHistCent");
  fHistCent->SetDirectory(0);
  tfileHist.Close();

  double fMin = (double)fHistCent->GetBinContent(280);
  cout << fMin << endl;

  TFile tfile("~/HypertritonData/HyperTree_Data.root", "RECREATE");
  TTree *tree = new TTree("HyperTree_Data", "An example of a ROOT tree");

  int Nev1040 = 0;

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

  while (fReader.Next())
  {
    Centrality = RColl->fCent;
    if (Centrality < 10.051 || Centrality > 40.05)
      continue;
    int bin = (int)(Centrality * 10.);
    double height = (double)fHistCent->GetBinContent(bin);
  //  if ((gRandom->Rndm() * height) > fMin)
  //    continue;

    Nev1040++;
    for (int i = 0; i < (static_cast<int>(RHyperVec.GetSize())); i++)
    {
      auto RHyper = RHyperVec[i];
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
  }

  // TFile tfileHist("~/data/hyper2body_data/CentHist.root", "RECREATE");
  // fHistCent->Write();
  // tfileHist.Close();

  tfile.Write("", TObject::kOverwrite);
  myFile->Close();

  tfile.Close();
  tfile.Delete();

  ofstream n_event_file;
  n_event_file.open("n_evet_1040.txt");
  n_event_file << Nev1040 << "\n";
  n_event_file.close();
}