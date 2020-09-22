#if !defined(__CINT__) || defined(__MAKECINT__)
#include <TROOT.h>
#include <Riostream.h>

#include "AliMCEvent.h"
#include "AliVParticle.h"
#include "AliMCEventHandler.h"
#endif

Double_t kHe3Mass = 2.809230089;

void He3AbsAnalysisPt(std::string pathToSimulation = "/home/fmazzasc/He3Simulation", bool bwRejection = true)
{
    std::string hypUtilsDir = getenv("HYPERML_UTILS");
    std::string bwFileArg = hypUtilsDir + "/BlastWaveFits.root";
    TFile bwFile(bwFileArg.data());
    std::cout<<bwFileArg.data()<<std::endl;
    TF1 *hypPtShape;
    hypPtShape = (TF1 *)bwFile.Get("BlastWave/BlastWave0");
    std::cout <<hypPtShape.GetMaximum() << std::endl;

    float ptBins[6] = {2, 3, 4, 5, 6, 9};
    TH1D *ptSpectrum = new TH1D("Reconstructed pT spectrum", "pTSpectrum; pT; Counts", 5, ptBins);
    int kNorm = 0;
    for (Int_t dir = 1; dir < 8; dir++)
    {
        AliMCEventHandler mcEvHandler("mcEvHandler", "MC Event Handler");
        std::string pathToDir = pathToSimulation + "/" + Form("00%i", dir);
        mcEvHandler.SetInputPath(pathToDir.data());
        mcEvHandler.Init("");
        Int_t iEvent = 0;

        while (mcEvHandler.LoadEvent(iEvent))
        {
            printf("\n Event %i \n", iEvent++);
            AliMCEvent *mcEv = mcEvHandler.MCEvent();

            for (Int_t i = 0; i < mcEv->GetNumberOfTracks(); ++i)
            {
                AliVParticle *part = mcEv->GetTrack(i);
                if (part->IsPhysicalPrimary() && std::abs(part->PdgCode()) == 1000020030)
                {
                    int counter = 0;
                    for (int c = part->GetDaughterFirst(); c < part->GetDaughterLast(); c++)
                    {
                        AliVParticle *dPart = mcEv->GetTrack(c);
                        int dPartPDG = dPart->PdgCode();
                        if (std::abs(dPartPDG) != 11 && std::abs(dPartPDG) != 22 && part->Pt() < 9)
                        {
                            if (bwRejection)
                            {
                                float hypPtShapeNum = hypPtShape->Eval(part->Pt());
                                // std::cout<<hypPtShapeNum<<std::endl;
                                if (hypPtShapeNum < gRandom->Rndm())
                                    break;
                            }
                            kNorm += 1;
                            // printf("\n PDG Dau Code:  %i \n", dPartPDG);
                            double absCt = ComputeHe3Ct(part, dPart);
                            double decCt = gRandom->Exp(7.6);
                            bool isAbsorbed = absCt < decCt;
                            if (isAbsorbed)
                                ptSpectrum->Fill(part->Pt());
                            break;
                        }
                    }
                }
            }
        }
    }
    TFile fFile("recPtHe3_1.5.root", "recreate");
    ptSpectrum->Scale(1. / kNorm);
    ptSpectrum->Write();
    bwFile.Close();
    fFile.Close();
}

Double_t ComputeHe3Ct(AliVParticle *he3Part, AliVParticle *dauPart)
{
    Double_t primVertex[3];
    Double_t secVertex[3];
    he3Part->XvYvZv(primVertex);
    dauPart->XvYvZv(secVertex);
    Double_t decLength = Dist(secVertex, primVertex);
    // std::cout<<kHe3Mass * decLength / he3Part->P()<<std::endl;
    return kHe3Mass * decLength / he3Part->P();
}

Double_t Dist(Double_t a[3], Double_t b[3]) { return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])); }
