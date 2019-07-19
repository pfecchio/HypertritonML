#include <TChain.h>
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
#include "AliPID.h"
#include <TRandom3.h>
#include <Riostream.h>
#include <TF1.h>
#include <TSystem.h>
#include "../include/Common.h"
#include "../include/Table.h"

void HyperTreeFatherMC(bool fRejec = true)
{

    char *dataDir{nullptr}, *tableDir{nullptr};
    getDirs(dataDir, tableDir);

    TFile bwFile("../fitsM.root");
    TF1 *BlastWave{nullptr};
    TF1 *BlastWave0{(TF1 *)bwFile.Get("BlastWave/BlastWave0")};
    TF1 *BlastWave1{(TF1 *)bwFile.Get("BlastWave/BlastWave1")};
    TF1 *BlastWave2{(TF1 *)bwFile.Get("BlastWave/BlastWave2")};

    BlastWave0->SetNormalized(1);
    BlastWave1->SetNormalized(1);
    BlastWave2->SetNormalized(1);

    std::vector<TF1 *> v{BlastWave1, BlastWave2};

    TF1 *BlastWave1040 = new TF1("BlastWave1040", SumTF1(v), 0, 10, 0);

    float max;
    float max0 = BlastWave0->GetMaximum();
    float max1 = BlastWave1->GetMaximum();
    float max2 = BlastWave2->GetMaximum();
    float max1040 = BlastWave1040->GetMaximum();

    TChain mcChain("_default/fTreeV0");
    cout<<dataDir<<endl;
    mcChain.AddFile(Form("%s/HyperTritonTree_19d2.root", dataDir));

    TTreeReader fReader(&mcChain);
    TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
    TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
    TTreeReaderValue<RCollision> RColl = {fReader, "RCollision"};

    TFile tableFile(Form("%s/SignalTable.root", tableDir), "RECREATE");
    Table outputTable("SignalTable", "Signal table");

    while (fReader.Next())
    {

        if (RColl->fCent <= 10)
        {
            BlastWave = BlastWave0;
            max = max0;
        }
        // else if (RColl->fCent > 10 && RColl->fCent <= 20)
        // {
        //     BlastWave = BlastWave1;
        //     max = max1;
        // }
        // else
        // {
        //     BlastWave = BlastWave2;
        //     max = max2;
        // }

        if (RColl->fCent > 10.051 && RColl->fCent < 40.05)
        {
            BlastWave = BlastWave1040;
            max = max1040;
        }
        else
        {
            BlastWave = BlastWave2;
            max = max2;
        }

        for (auto &SHyper : SHyperVec)
        {

            float mc_pt = Hypot(SHyper.fPxHe3 + SHyper.fPxPi, SHyper.fPyHe3 + SHyper.fPyPi);
            float BlastWaveNum = (BlastWave->Eval(mc_pt)) / max;
            if (fRejec == true)
            {
                if (BlastWaveNum < gRandom->Rndm())
                    continue;
            }

            int ind = SHyper.fRecoIndex;

            if (SHyper.fFake == false)
            {

                auto &RHyper = RHyperVec[ind];
                outputTable.Fill(RHyper, *RColl);
            }
        }
    }

    tableFile.cd();
    outputTable.Write();
}
