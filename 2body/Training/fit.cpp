#include <cmath>

#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TString.h>
#include <TPaveText.h>
#include <TStyle.h>
#include <AliVertexingHFUtils.h>
constexpr double kNorm{93302191};
//maybe a better way to put the efficiency can be found
constexpr double kEfficiency[]{0.11856290399597486, 0.15661899102241447, 0.16222183704217277, 0.1648291740454953, 0.16816989503204774, 0.1714054462910652, 0.1760109901835254, 0.1795357879760743, 0.18128415203975598, 0.18128415203975598};//da fixare
constexpr double kEfficiencyBDT[]{0.5345631565462465, 0.6126070991432069, 0.7283126265666904, 0.8428128231644261, 0.7672894302229563, 0.8265470784703719, 0.8218950749464669, 0.8683673469387755, 0.7831830306109488, 0.8498358471713375, 0.8863994518670778, 0.8552803129074316, 0.7823441081166522, 0.8429954578054029, 0.8699920403289997, 0.9178571428571428, 0.8107591754650578, 0.8629856850715747, 0.8575952822976958, 0.9049586776859504, 0.7435292435292435, 0.7951741753165589, 0.8144523604478197, 0.8267605633802817, 0.6663605051664753, 0.8070285254404006, 0.7971966157343099, 0.7725, 0.6598587223587223, 0.7089543196413426, 0.7628309022150189, 0.7, 0.5052379527087698, 0.6204100022527597, 0.567722371967655, 0.7169811320754716};
//it needs the binning because because trouble with pyroot
constexpr double kCtBinning[]{0,2,4,6,8,10,14,18,23,28};

void fit(TString fileName = "../../../HypertritonAnalysis/Trees/results.root", float minCent = 0, float maxCent = 90,char* variable = "ct") {

  TFile inputFile(fileName.Data());
  TH2* inputH2 = (TH2*)inputFile.Get("InvMassVsX");

  TFile output(Form("fit_%s",fileName.Data()),"recreate");
  TF1 fitTpl("fitTpl","expo(0)+pol0(2)+gausn(3)",0,5);\
  fitTpl.SetParNames("B_{exp}","#tau","B_{0}","N_{sig}","#mu","#sigma");
  TF1 bkgTpl("fitTpl","expo(0)+pol0(2)",0,5);
  fitTpl.SetNpx(300);
  fitTpl.SetLineWidth(2);
  fitTpl.SetLineColor(kRed);
  bkgTpl.SetNpx(300);
  bkgTpl.SetLineWidth(2);
  bkgTpl.SetLineStyle(kDashed);
  bkgTpl.SetLineColor(kRed);
  fitTpl.SetParameter(0,2);
  fitTpl.SetParameter(1,-1);
  fitTpl.SetParameter(2,0);
  fitTpl.SetParameter(3,40);
  fitTpl.SetParameter(4,2.99);
  fitTpl.SetParameter(5,0.002);
  fitTpl.SetParLimits(5,0.0001,0.004);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  TH1* rawSpectra = inputH2->ProjectionY("RawSpectra");
  rawSpectra->Reset();  
  rawSpectra->UseCurrentStyle();

  TH1* corSpectra = (TH1*)rawSpectra->Clone("CorSpectra");

  double row_counts[10];
  double cor_counts[10];
  TH1D var_distr("histo",";ct [cm];dN/dct [cm^{-1}]",9,kCtBinning);
  var_distr.SetTitle(";ct [cm];dN/dct [cm^{-1}];");
  for (int iB = 1; iB <= inputH2->GetNbinsY(); ++iB) {

    TCanvas cv(Form("cv%i",iB));
    TH1* input = inputH2->ProjectionX(Form("proj%i",iB),iB,iB);
    input->UseCurrentStyle();
    input->SetLineColor(kBlack);
    input->SetMarkerStyle(20);
    input->SetMarkerColor(kBlack);
    input->SetTitle(";m (^{3}He + #pi) (GeV/#it{c})^{2};Counts");
    input->SetMaximum(1.5 * input->GetMaximum());
    input->Fit(&fitTpl,"RL","",2.96,3.03);
    input->SetDrawOption("e");
    input->GetXaxis()->SetRangeUser(2.96,3.03);
    bkgTpl.SetParameters(fitTpl.GetParameters());
    bkgTpl.Draw("same");

  
    double nsigma = 3;
    double mu = fitTpl.GetParameter(4);
    double sigma = fitTpl.GetParameter(5);
    double signal,errsignal,bkg,errbkg,signif,errsignif;
    signal = fitTpl.GetParameter(3) / input->GetBinWidth(1);
    errsignal = fitTpl.GetParError(3) / input->GetBinWidth(1);
    bkg = bkgTpl.Integral(mu - nsigma * sigma, mu + nsigma * sigma) / input->GetBinWidth(1);
    errbkg = std::sqrt(bkg);
    AliVertexingHFUtils::ComputeSignificance(signal,errsignal,bkg,errbkg,signif,errsignif);
    //compute the signal
    double peak = input->Integral(30*(mu-nsigma*sigma-2.96)/(3.05-2.96),30*(mu+nsigma*sigma-2.96)/(3.05-2.96));
    cout<<"int: "<<input->Integral(30*(mu-nsigma*sigma-2.96)/(3.05-2.96),30*(mu+nsigma*sigma-2.96)/(3.05-2.96))<<endl;
    cout<<"peak: "<<peak<<endl;
    cout<<"bkg: "<<bkg<<endl;
    cout<<"eff: "<<kEfficiency[iB]<<endl;
    cout<<"effBDT: "<<kEfficiencyBDT[iB]<<endl;
    row_counts[iB]=peak-signal;
    cor_counts[iB]=row_counts[iB]/kEfficiency[iB]/kEfficiencyBDT[iB];
    cout<<"sig: "<<cor_counts[iB]<<endl;
    var_distr.SetBinContent(iB,cor_counts[iB]);
    var_distr.SetBinError(iB,TMath::Sqrt(cor_counts[iB]));

    TPaveText *pinfo2=new TPaveText(0.5,0.5,0.91,0.9,"NDC");
    pinfo2->SetBorderSize(0);
    pinfo2->SetFillStyle(0);
    pinfo2->SetTextAlign(kHAlignRight+kVAlignTop);
    pinfo2->SetTextFont(42);
    TString str=Form("ALICE Internal, Pb-Pb 2018 %2.0f-%2.0f%%",minCent,maxCent);
    pinfo2->AddText(str);    
    str=Form("{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi + c.c., %1.1f #leq #it{%s} < %1.1f GeV/#it{c} ",kCtBinning[iB-1],variable,kCtBinning[iB]);
    pinfo2->AddText(str);    
    str=Form("Significance (%.0f#sigma) %.1f #pm %.1f ",nsigma,signif,errsignif);
    pinfo2->AddText(str);
  
    str=Form("S (%.0f#sigma) %.0f #pm %.0f ",nsigma,signal,errsignal);
    pinfo2->AddText(str);
    str=Form("B (%.0f#sigma) %.0f #pm %.0f",nsigma,bkg,errbkg);
    pinfo2->AddText(str);
    if(bkg>0) str=Form("S/B (%.0f#sigma) %.4f ",nsigma,signal/bkg); 
    pinfo2->AddText(str);
    pinfo2->Draw();


    input->Write();
    cv.SaveAs(Form("img/hyper%2.0f_%2.0f_%i.pdf",minCent,maxCent,iB));
    cv.Write();
    //prende la larghezza del bin di pt
    double deltaPt = inputH2->GetYaxis()->GetBinUpEdge(iB) - inputH2->GetYaxis()->GetBinLowEdge(iB);
    //non corretto
    rawSpectra->SetBinContent(iB, signal / kNorm / deltaPt);
    rawSpectra->SetBinError(iB, errsignal / kNorm / deltaPt);
    //correto cper efficiencza
    corSpectra->SetBinContent(iB, signal / kEfficiency[iB-1]/ kEfficiencyBDT[iB-1] / kNorm / deltaPt);
    corSpectra->SetBinError(iB, errsignal / kEfficiency[iB-1]/ kEfficiencyBDT[iB-1] / kNorm / deltaPt);
  }
  cout<<var_distr.GetBinContent(3);
  rawSpectra->Write();
  corSpectra->Write();

  TF1 exp("fitTpl","[0]*TMath::Exp(-x/[1]/0.029979245800)");
  exp.SetParLimits(1,180,240);
  var_distr.Fit(&exp);
  
  var_distr.Write();
  output.Close();

}