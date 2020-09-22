#if !defined(__CINT__) || defined(__MAKECINT__)
#include <TROOT.h>
#include <Riostream.h>

#include "AliMCEvent.h"
#include "AliVParticle.h"
#include "AliMCEventHandler.h"
#endif

Double_t kHe3Mass = 2.809230089;

void He3AbsAnalysis()
{
    // double ctBins[11] = {0, 1, 2, 4, 6, 8, 10, 14, 18, 23, 35};
    TH1D *ctSpectrum = new TH1D("Reconstructed ct spectrum", "ctSpectrum; ct; Counts", 50, 0, 100);
    for (Int_t dir = 4; dir < 9; dir++)
    {
        AliMCEventHandler mcEvHandler("mcEvHandler", "MC Event Handler");
        mcEvHandler.SetInputPath(Form("./00%i", dir));
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
                        if (std::abs(dPartPDG) != 11 && std::abs(dPartPDG) != 22)
                        {
                            // printf("\n PDG Dau Code:  %i \n", dPartPDG);                         
                            ctSpectrum->Fill(ComputeHe3Ct(part, dPart));
                            break;
                        }
                    }
                }
            }
        }
    }
    TFile fFile("recCtHe3.root", "recreate");
    ctSpectrum->Write();
    fFile.Close();
}

Double_t ComputeHe3Ct(AliVParticle *he3Part, AliVParticle *dauPart)
{
    Double_t primVertex[3];
    Double_t secVertex[3];
    he3Part->XvYvZv(primVertex);
    dauPart->XvYvZv(secVertex);
    Double_t decLength = Dist(secVertex, primVertex);
    std::cout<<decLength<<std::endl;
    return kHe3Mass * decLength / he3Part->P();
}

Double_t Dist(Double_t a[3], Double_t b[3]) { return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])); }
