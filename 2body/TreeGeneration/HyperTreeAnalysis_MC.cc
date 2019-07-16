#include <TROOT.h>
#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <vector>
#include "TH1D.h"
#include <TCanvas.h>
#include <TList.h>
#include <TDirectoryFile.h>
#include "AliAnalysisTaskHyperTriton2He3piML.h"
#include <TMath.h>
#include "Math/Vector4D.h"
#include "AliPID.h"
#include <TLorentzVector.h>
#include <Riostream.h>

float SProd(TLorentzVector a1, TLorentzVector a2)
{

    return fabs(a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2]);
}

float Sq(float a)
{
    return a * a;
}

float VProd(TLorentzVector a1, TLorentzVector a2)
{
    float x = a1[1] * a2[2] - a2[1] * a1[2];
    float y = a1[2] * a2[0] - a1[0] * a2[2];
    float z = a1[0] * a2[1] - a1[1] * a2[0];
    return TMath::Sqrt(x * x + y * y + z * z);
}

template <typename F>
double Hypot(F a, F b, F c)
{
    return std::sqrt(a * a + b * b + c * c);
}

template <typename F>
double Hypot(F a, F b, F c, F d)
{
    return std::sqrt(a * a + b * b + c * c + d * d);
}

void HyperTreeAnalysis_MC()
{

    TFile *myFile = TFile::Open("HyperTritonTree_16h7abc.root ", "r");
    TDirectoryFile *mydir = (TDirectoryFile *)myFile->Get("_default");
    TTreeReader fReader("fTreeV0", mydir);
    TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
    TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
    TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};
    TH1D *fHistSHypPt =
        new TH1D("fHistSHypPt", ";V0 #it{p}_{T} (GeV/#it{c}); Counts", 40, 0., 10.);
    TH1D *fHistRHypPt =
        new TH1D("fHistRHypPt", ";V0 #it{p}_{T} (GeV/#it{c}); Counts", 40, 0., 10.);
    TH2D *fHistSAlpha =
        new TH2D("fHistSAlpha", ";Armenteros Alpha; q ; Counts", 256, -1., 1., 256, 0., 0.25);
    TH2D *fHistRAlpha =
        new TH2D("fHistRAlpha", ";Armenteros Alpha; q ; Counts", 256, -1., 1., 256, 0., 0.25);
    TH1D *fHistInvMassR =
        new TH1D("fHistInvMassR", ";InvMass; Counts", 400, 2.90, 3.5);

    TH1D *fHistInvMassS =
        new TH1D("fHistInvMassS", ";InvMass; Counts", 100, 2.990, 2.992);
    TH2D *fHistInvMassVsPt =
        new TH2D("fHistInvMassVsPt", ";InvMass; pt ; Counts", 200, 1., 10., 200, 2.95, 3.2);

    TH1D *fHistSCosPa =
        new TH1D("fHistSCosPa", ";CosPa; Counts", 100, 0, 1);
    TH1D *fHistRCosPa =
        new TH1D("fHistRCosPa", ";CosPa; Counts", 100, 0, 1);

    while (fReader.Next())
    {

        for (int i = 0; i < (static_cast<int>(SHyperVec.GetSize())); i++)
        {
            auto SHyper = SHyperVec[i];

            double eHe3 = Hypot(SHyper.fPxHe3, SHyper.fPyHe3, SHyper.fPzHe3, AliPID::ParticleMass(AliPID::kHe3));
            double ePi = Hypot(SHyper.fPxPi, SHyper.fPyPi, SHyper.fPzPi, AliPID::ParticleMass(AliPID::kPion));
            TLorentzVector he3Vector, piVector, LVector;
            he3Vector.SetPxPyPzE(SHyper.fPxHe3, SHyper.fPyHe3, SHyper.fPzHe3, eHe3);
            piVector.SetPxPyPzE(SHyper.fPxPi, SHyper.fPyPi, SHyper.fPzPi, ePi);
            auto hyperVector = piVector + he3Vector;
            LVector.SetPxPyPzE(SHyper.fDecayX - RColl->fX, SHyper.fDecayY - RColl->fY, SHyper.fDecayZ - RColl->fZ, 1);
            fHistSCosPa->Fill(SProd(hyperVector, LVector) / (hyperVector.P() * LVector.P()));
            float qP, qN, qT;
            if (SHyper.fPdgCode > 0)
            {
                qP = SProd(hyperVector, he3Vector) / fabs(hyperVector.P());
                qN = SProd(hyperVector, piVector) / fabs(hyperVector.P());
                qT = VProd(hyperVector, he3Vector) / fabs(hyperVector.P());
            }

            else
            {
                qN = SProd(hyperVector, he3Vector) / fabs(hyperVector.P());
                qP = SProd(hyperVector, piVector) / fabs(hyperVector.P());
                qT = fabs(VProd(hyperVector, piVector)) / fabs(hyperVector.P());
            }
            float alpha = (qP - qN) / (qP + qN);
            fHistSAlpha->Fill(alpha, qT);
            fHistSHypPt->Fill(hyperVector.Pt());
            fHistInvMassS->Fill(hyperVector.M());
        }
        for (int i = 0; i < (static_cast<int>(RHyperVec.GetSize())); i++)
        {
            auto RHyper = RHyperVec[i];
            double eHe3 = Hypot(RHyper.fPxHe3, RHyper.fPyHe3, RHyper.fPzHe3, AliPID::ParticleMass(AliPID::kHe3));
            double ePi = Hypot(RHyper.fPxPi, RHyper.fPyPi, RHyper.fPzPi, AliPID::ParticleMass(AliPID::kPion));
            TLorentzVector he3Vector, piVector, LVector;
            he3Vector.SetPxPyPzE(RHyper.fPxHe3, RHyper.fPyHe3, RHyper.fPzHe3, eHe3);
            piVector.SetPxPyPzE(RHyper.fPxPi, RHyper.fPyPi, RHyper.fPzPi, ePi);
            auto hyperVector = piVector + he3Vector;
            LVector.SetPxPyPzE(RHyper.fDecayX, RHyper.fDecayY, RHyper.fDecayZ, 1);
            fHistRCosPa->Fill(SProd(hyperVector, LVector) / (hyperVector.P() * LVector.P()));
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

            fHistRHypPt->Fill(hyperVector.Pt());
            if (fabs(RHyper.fTPCnSigmaHe3) < 1)
            {
                fHistInvMassR->Fill(hyperVector.M());
                fHistRAlpha->Fill(alpha, qT);
                fHistInvMassVsPt->Fill(hyperVector.Pt(), hyperVector.M());
            }
        }
    }

    TFile outputFile("Histos_MC.root", "RECREATE");
    fHistSHypPt->Write("fHistSHypPt");
    fHistRHypPt->Write("fHistRHypPt");
    fHistSAlpha->Write("fHistSAlpha");
    fHistRAlpha->Write("fHistRAlpha");
    fHistInvMassR->Write("fHistInvMassR");
    fHistInvMassS->Write("fHistInvMassS");
    fHistInvMassVsPt->Write("fHistInvMassVsPt");
    fHistSCosPa->Write("fHistSCosPa");
    fHistRCosPa->Write("fHistRCosPa");
    outputFile.Close();
}

void HyperTreeMatching()
{

    TFile *myFile = TFile::Open("HyperTritonTree_16h7abc.root ", "r");
    TDirectoryFile *mydir = (TDirectoryFile *)myFile->Get("_default");
    TTreeReader fReader("fTreeV0", mydir);
    TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
    TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
    TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};
    TH1D *fHistDecX =
        new TH1D("Xrec-Xsim", ";Xrec-Xsim[cm]; Counts", 40, -4., 4.);
    TH1D *fHistDecY =
        new TH1D("Yrec-Ysim", ";Yrec-Ysim[cm]; Counts", 40, -4., 4.);
    TH1D *fHistDecZ =
        new TH1D("Zrec-Zsim", ";Zrec-Zsim[cm]; Counts", 40, -4., 4.);
    TH2D *fHistLAnalysis =
        new TH2D("fHistLAnalysis", ";Lrec-Lsim[cm];Lrec[cm]; Counts", 200, -10., 10., 100, 0, 50);

    TH1D *fHistSPt =
        new TH1D("fHistSPt", ";Pt[Gev/c]; Counts", 40, 0., 10.);

    TH1D *fHistRPt =
        new TH1D("fHistRPt", ";Pt[Gev/c]; Counts", 40, 0., 10.);

    TH2D *fHistAlpha =
        new TH2D("fHistAlpha", ";Armenteros Alpha; q ; Counts", 256, -1., 1., 256, 0., 0.25);

    TH1D *fHistRCosPaSignal =
        new TH1D("fHistRCosPaSignal", ";CosPa; Counts", 100, 0, 1);

    float rec = 0;
    float gen = 0;
    while (fReader.Next())
    {
        for (int i = 0; i < (static_cast<int>(SHyperVec.GetSize())); i++)
        {
            auto SHyper = SHyperVec[i];
            float v0ptS = TMath::Sqrt(Sq(SHyper.fPxHe3 + SHyper.fPxPi) + Sq(SHyper.fPyHe3 + SHyper.fPyPi));
            int reco = SHyper.fRecoIndex;
            fHistSPt->Fill(v0ptS);
            gen = gen + 1;
            if (SHyper.fFake == false)
            {
                rec = rec + 1;
                auto RHyper = RHyperVec[reco];
                double eHe3 = Hypot(RHyper.fPxHe3, RHyper.fPyHe3, RHyper.fPzHe3, AliPID::ParticleMass(AliPID::kHe3));
                double ePi = Hypot(RHyper.fPxPi, RHyper.fPyPi, RHyper.fPzPi, AliPID::ParticleMass(AliPID::kPion));
                TLorentzVector he3Vector, piVector, LVector;
                he3Vector.SetPxPyPzE(RHyper.fPxHe3, RHyper.fPyHe3, RHyper.fPzHe3, eHe3);
                piVector.SetPxPyPzE(RHyper.fPxPi, RHyper.fPyPi, RHyper.fPzPi, ePi);
                auto hyperVector = piVector + he3Vector;
                LVector.SetPxPyPzE(RHyper.fDecayX, RHyper.fDecayY, RHyper.fDecayZ, 1);
                fHistRCosPaSignal->Fill(SProd(hyperVector, LVector) / (hyperVector.P() * LVector.P()));
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
                float v0ptR = hyperVector.Pt();
                fHistRPt->Fill(v0ptR);
                float Xsim = SHyper.fDecayX - RColl->fX;
                float Ysim = SHyper.fDecayY - RColl->fY;
                float Zsim = SHyper.fDecayZ - RColl->fZ;
                fHistDecX->Fill(RHyper.fDecayX - Xsim);
                fHistDecY->Fill(RHyper.fDecayY - Ysim);
                fHistDecZ->Fill(RHyper.fDecayZ - Zsim);
                float Lrec = Hypot(RHyper.fDecayX, RHyper.fDecayY, RHyper.fDecayZ);
                float Lsim = Hypot(Xsim, Ysim, Zsim);
                fHistLAnalysis->Fill(Lrec - Lsim, Lrec);
                fHistAlpha->Fill(alpha, qT);
            }
        }

    }
    cout << rec/gen << endl;
    auto fHistEffvsPt = (TH1D *)fHistRPt->Clone("EffvsPt");
    fHistEffvsPt->Divide(fHistSPt);
    TFile outputFile2("Histos_Sim_Rec_Matching.root", "RECREATE");
    fHistEffvsPt->Write();
    fHistDecX->Write();
    fHistDecY->Write();
    fHistDecZ->Write();
    fHistLAnalysis->Write();
    fHistAlpha->Write();
    fHistRCosPaSignal->Write();
    outputFile2.Close();
}
