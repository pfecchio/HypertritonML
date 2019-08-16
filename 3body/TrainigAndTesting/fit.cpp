using namespace std;
void fit()
{
  float Centrality[] = {10,30,50};
  float Yield[] = {3.37,1.28,0.77};
  float YieldAnti[] = {3.38,1.30,0.77};
  float dNde[]={1756,983,415};
  float UncdNde[]={52,25,14};
  float StatUnc[] = {0.77,0.33,0.23};
  float StatUncAnti[] = {0.70,0.36,0.25};
  float SystUnc[] = {0.40,0.17,0.1};
  float SystUncAnti[] = {0.48,0.2,0.12};
  float TotUnc[3];
  float TotUncAnti[3];
  float BinCentrality[] = {5,20,40};
  float UncBinCentrality[] ={5,10,10};

  for(int index=0;index<3;index++)
  {
      TotUnc[index]=SystUnc[index]*SystUnc[index]+StatUnc[index]*StatUnc[index];
      TotUncAnti[index]=SystUnc[index]*SystUnc[index]+StatUnc[index]*StatUnc[index];
  }
  TCanvas* cv= new TCanvas("cv","cv",600,400);
  cv->cd();
  TGraphErrors* YieldVsCen= new TGraphErrors(3,BinCentrality,Yield,UncBinCentrality,TotUnc);
  YieldVsCen->SetTitle("Yield vs Centrality;centrality;yield");
  YieldVsCen->Draw("AP");
  TF1* expo = new TF1("f1","[0]*TMath::Exp(-x/[1])+[2]");
  TF1* expop = new TF1("f2","[0]*TMath::Exp(-x/[1])+[2]");
  TF1* pol1 = new TF1("f3","[0]+x*[1]");
  expo->SetParLimits(1,10,100);
  expo->SetParLimits(0,3,20);
  YieldVsCen->Fit(expo,"L");
  //YieldVsCen->Fit(expop,"M+");
  //YieldVsCen->Fit(pol1,"M+");
  cout<<expo->GetProb()<<endl;
  TCanvas* cv2= new TCanvas("cv2","cv2",600,400);
  cv2->cd();
  TGraphErrors* YieldVsdN= new TGraphErrors(3,dNde,Yield,UncdNde,TotUnc);
  YieldVsdN->SetTitle("Yield vs dN/d#eta;dN/d#eta;yield");
  YieldVsdN->Draw("AP");
  TF1* expo2 = new TF1("f1","[0]*TMath::Exp((x-[3])*[1])");
  YieldVsdN->Fit(expo2,"M+");
  //expo2->SetParameter(0,3);
  //expo2->Draw("SAME");
  cout<<expo->GetProb()<<endl;
}