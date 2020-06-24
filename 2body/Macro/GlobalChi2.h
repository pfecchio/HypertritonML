#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"
#include "TH1.h"
#include "TList.h"
#include "Math/WrappedMultiTF1.h"
#include "HFitInterface.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TF1.h"

// definition of shared parameter
// background function
int iparExp1[2] = { 0,      // normalisation
                 1    // tau
};

// signal + background function
int iparExp2[5] = { 0, // normalisation (common)
                  1, // tau (common)
                  2, // delta
};

double ComputeIntegral(TF1 *func, int BinLowEdge, int BinSupEdge){

double expo_inf, expo_sup;

if(func->GetNpar() == 2){

expo_inf = TMath::Exp(-BinLowEdge/(0.029979245800*func->GetParameter(1)));
expo_sup = TMath::Exp(-BinSupEdge/(0.029979245800*func->GetParameter(1)));

}

else
{
expo_inf = TMath::Exp(-BinLowEdge/(0.029979245800*(func->GetParameter(1)+func->GetParameter(2))));
expo_sup = TMath::Exp(-BinSupEdge/(0.029979245800*(func->GetParameter(1)+func->GetParameter(2))));
}

return func->GetParameter(0)*(expo_inf - expo_sup);

}

struct GlobalChi2 {
   GlobalChi2(TH1D* h1, TH1D* h2, TF1 *f1, TF1 *f2) :
      fHist_1(h1), fHist_2(h2), fF1(f1), fF2(f2) {}


   double operator() (const double *par) const {
      double p1[2];
      for (int i = 0; i < 2; ++i) p1[i] = par[iparExp1[i] ];

      double p2[3];
      for (int i = 0; i < 3; ++i) p2[i] = par[iparExp2[i] ];

      fF1->SetParameters(p1);
      fF2->SetParameters(p2);
      double chi2_1=0;
      double chi2_2=0;
      for (int i=1; i<=  fHist_1->GetNbinsX(); i++){
         int BinWidth = fHist_1->GetBinWidth(i);
         int BinLowEdge = fHist_1->GetBinLowEdge(i);

         chi2_1 += TMath::Power(ComputeIntegral(fF1,BinLowEdge, BinLowEdge + BinWidth)/BinWidth - fHist_1->GetBinContent(i), 2)/fHist_1->GetBinContent(i); 
         chi2_2 += TMath::Power(ComputeIntegral(fF2, BinLowEdge, BinLowEdge + BinWidth)/BinWidth - fHist_2->GetBinContent(i), 2)/fHist_2->GetBinContent(i);


      }

      return chi2_1 + chi2_2;
   }

   const  TH1D * fHist_1;
   const  TH1D * fHist_2;
   TF1 * fF1;
   TF1 * fF2;
};


