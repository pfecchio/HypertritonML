#include "Riostream.h"
#include <TCanvas.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TH2.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPad.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>


  const Int_t kMarkTyp=20; //marker type
  const Int_t kMarkCol=1; //...and color
  const Float_t kTitSize=0.055; //axis title size
  const Float_t kAsize=0.85*kTitSize; //...and label size
  const Float_t kToffset=0.8;
  const Int_t kFont=42;


void CollectionPlotMass(){

  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);

  const int N=5;
  Float_t point[N] = {1,2,3,4,5};
  Float_t bind[N] = {0.41,0.08,-0.16,0.23,0.4};
  Float_t err_x[N]={0,0,0,0,0};
  Float_t err_y_low[N] = {0.12,0.07,0.27,0.08,0.12};
  Float_t err_y_high[N] = {0.12,0.07,0.27,0.08,0.12};
  Float_t errsyst_y[N] = {0.,0.,0.,0.,0.11};
  Float_t erry_sumq_h[N]={0,0,0,0,0};
  Float_t erry_sumq_l[N]={0,0,0,0,0};
  Float_t errsyst_x[N]={0,0,0,0,0.1};

  double w[N]={0,0,0,0,0};
  double v[N]={0,0,0,0,0};
  double s[N]={0,0,0,0,0};
  double sum_w=0;
  double sum_v=0;
  double chi2 =0;

  for(int i=0;i<N;i++){
    cout << "BL: " << bind[i] << endl;
    s[i]=sqrt(err_y_high[i]*err_y_high[i]+errsyst_y[i]*errsyst_y[i]);
    erry_sumq_h[i]=sqrt(err_y_high[i]*err_y_high[i]+errsyst_y[i]*errsyst_y[i]);
    erry_sumq_l[i]=sqrt(err_y_low[i]*err_y_low[i]+errsyst_y[i]*errsyst_y[i]);
    cout << "Value s" << i << " : " << s[i] << endl;
    w[i]=1/(s[i]*s[i]);
    cout << "Value weight" << i << " : " << w[i] << endl;
    v[i]=w[i]*bind[i];
    cout << "Value weigthed" << i << " : " << v[i] << endl;
    sum_w=sum_w+w[i];
    sum_v=sum_v+v[i];
    //chi2=chi2+(w[i]*(206.651-bind[i])*(206.651-bind[i]));
  }
  cout << "Sum of weights after all it : " << sum_w << endl;
  cout << "Sum of weighted measurements after all it : " << sum_v << endl;
  Float_t Average = sum_v/sum_w;
  Float_t Error = 1/sqrt(sum_w);
  cout << "Final value : " << Average << endl;
  cout << "Uncertainty : " << Error << endl;
  //cout << "Chi2 : " << chi2 << endl;
  //cout << "Chi2/(N-1) : " << chi2/(N-1) << endl;

  TGraphAsymmErrors *gSpect;
  gSpect = new TGraphAsymmErrors(N,point,bind,err_x,err_x,err_y_low,err_y_high);

  TGraphAsymmErrors *gSpect2; //syst. err.
  gSpect2 = new TGraphAsymmErrors(N,point,bind,errsyst_x,errsyst_x,errsyst_y,errsyst_y);

  Float_t point_a[1]={6};
  Float_t bind_a[1]={0.049};
  Float_t err_x_a[1]={0};
  Float_t err_y_low_a[1]={0.061};
  Float_t err_y_high_a[1]={0.061};
  Float_t errsyst_y_a[1]={0.046};
  Float_t errsyst_x_a[1]={0.1};
  TGraphAsymmErrors *gSpect_alice = new TGraphAsymmErrors(1,point_a,bind_a,err_x_a,err_x_a,err_y_low_a,err_y_high_a);
  TGraphAsymmErrors *gSpect2_alice = new TGraphAsymmErrors(1,point_a,bind_a,errsyst_x_a,errsyst_x_a,errsyst_y_a,errsyst_y_a);

  TLine *Fcn = new TLine(0.5,0,6.5,0);
  Fcn->SetLineWidth(1);
  Fcn->SetLineStyle(2);
  Fcn->SetLineColor(kBlue);
  //TLine *Fcn1 = new TLine(0.5,Average,6.5,Average);
  //Fcn1->SetLineWidth(1);
  //Fcn1->SetLineStyle(2);
  //Fcn1->SetLineColor(kBlue-3);

  TCanvas *c1=new TCanvas("rawyield","rawyields",20,20,1600,1024);
  //c1->SetTopMargin(0.03); c1->SetBottomMargin(0.145);
  c1->SetLeftMargin(0.15); c1->SetRightMargin(0.1);
  c1->cd(1); //pad->SetLogy();

  TH2D *ho1 = new TH2D("ho1", "ho1", 1000, 0.5, 6.5, 1000, -1, 1); //...just for frame
  //ho1->GetYaxis()->SetTitle("{}^{3}_{#Lambda}H Lifetime (ps)");
  ho1->GetYaxis()->SetTitle("B_{#Lambda} (MeV)");
  ho1->GetXaxis()->SetTitleOffset(1.1);
  ho1->GetXaxis()->SetNdivisions(000);
  ho1->GetYaxis()->SetTitleOffset(1.1);
  ho1->GetYaxis()->SetLabelOffset(.01);
  ho1->SetTitleSize(0.05,"XY");ho1->SetTitleFont(42,"XY");
  ho1->SetLabelSize(0.04,"XY");ho1->SetLabelFont(42,"XY");
  ho1->SetMarkerStyle(kFullCircle);;
  ho1->SetTitle("");
  ho1->Draw("");

  double x[] = {0.5, 6.5};
  double y[] = {Average, Average};
  double ex[] = {0,0};
  double ey[] = {1/sqrt(sum_w),1/sqrt(sum_w)};
  
  /*
  TGraphErrors* ge = new TGraphErrors(2, x, y, ex, ey);
  ge->SetFillColorAlpha(kBlue-3,0.35);
  ge->SetFillStyle(3004);
  ge->Draw("3SAME");

  TLine *Fcn1_up = new TLine(0.5,Average+Error,6.5,Average+Error);
  Fcn1_up->SetLineWidth(1);
  Fcn1_up->SetLineColor(kBlue-3);
  TLine *Fcn1_low = new TLine(0.5,Average-Error,6.5,Average-Error);
  Fcn1_low->SetLineWidth(1);
  Fcn1_low->SetLineColor(kBlue-3);


  TBox *boxFcn1 = new TBox(0,0,0,0);
  boxFcn1->SetFillColorAlpha(kBlue-3,0.35);
  boxFcn1->SetFillStyle(3004);
  boxFcn1->SetLineWidth(1);
  boxFcn1->SetLineStyle(2);
  boxFcn1->SetLineColor(kBlue-3);
  boxFcn1->Draw("same");
  */


  gSpect->SetMarkerStyle(kMarkTyp); gSpect->SetMarkerSize(1.4);
  gSpect->SetMarkerColor(kMarkCol); gSpect->SetLineColor(kMarkCol);
  gSpect->SetLineWidth(1);

  gSpect2->SetMarkerStyle(0);gSpect2->SetMarkerColor(kMarkCol); gSpect2->SetMarkerSize(0.1);
  gSpect2->SetLineStyle(1); gSpect2->SetLineColor(kMarkCol); gSpect2->SetLineWidth(1);
  gSpect2->SetFillColor(0); gSpect2->SetFillStyle(0);

  gSpect_alice->SetMarkerStyle(kFullDiamond); gSpect_alice->SetMarkerSize(2.8);
  gSpect_alice->SetMarkerColor(kRed); gSpect_alice->SetLineColor(kRed);
  gSpect_alice->SetLineWidth(1);

  gSpect2_alice->SetMarkerStyle(0);gSpect2_alice->SetMarkerColor(kRed); gSpect2_alice->SetMarkerSize(0.1);
  gSpect2_alice->SetLineStyle(1); gSpect2_alice->SetLineColor(kRed); gSpect2_alice->SetLineWidth(1);
  gSpect2_alice->SetFillColor(0); gSpect2_alice->SetFillStyle(0);


  Fcn->Draw("SAME");
  //Fcn1->Draw("SAME");
  //Fcn1_up->Draw("SAME");
  //Fcn1_low->Draw("SAME");
  gSpect->Draw("pzsame");
  gSpect2->Draw("spe2");
  gSpect_alice->Draw("pzsame");
  gSpect2_alice->Draw("spe2");

  //TLegend *leg1 = new TLegend(.18,0.75,.53,0.9);
  //leg1->SetFillStyle(0);
  //leg1->SetMargin(0.16); //separation symbol-text
  //leg1->SetBorderSize(0);
  //leg1->SetTextFont(42);
  //leg1->SetTextSize(0.025);
  ////leg1->SetEntrySeparation(0.1);
  //leg1->AddEntry(boxFcn1, "Average B_{#Lambda}", "fl");
  ////leg1->AddEntry(gSpect, "experimental value","p");


  //leg1->Draw();

  TLatex *note=new TLatex();  note->SetNDC(kTRUE);
  note->SetTextColor(1); note->SetTextFont(42); note->SetTextSize(.04);
  note->DrawLatex(0.2, 0.20, Form("Measures recalibrated with the latest mass measurement of:"));
  TLatex *note2=new TLatex();  note2->SetNDC(kTRUE);
  note2->SetTextColor(1); note2->SetTextFont(42); note2->SetTextSize(.04);
  note2->DrawLatex(0.2, 0.15, Form("#pi, p, d,^{3}He"));

  TLatex *lat1=new TLatex();  lat1->SetNDC(kTRUE);
  lat1->SetTextColor(1); lat1->SetTextFont(42); lat1->SetTextSize(.033);
  lat1->DrawLatex(0.23, 0.70, Form("NPB1(1967)"));

  TLatex *lat2=new TLatex();  lat2->SetNDC(kTRUE);
  lat2->SetTextColor(1); lat2->SetTextFont(42); lat2->SetTextSize(.033);
  lat2->DrawLatex(0.29, 0.45, Form("NPB4(1968)"));

  TLatex *lat3=new TLatex();  lat3->SetNDC(kTRUE);
  lat3->SetTextColor(1); lat3->SetTextFont(42); lat3->SetTextSize(.033);
  lat3->DrawLatex(0.42, 0.29, Form("PRD1(1970)"));

  TLatex *lat4=new TLatex();  lat4->SetNDC(kTRUE);
  lat4->SetTextColor(1); lat4->SetTextFont(42); lat4->SetTextSize(.033);
  lat4->DrawLatex(0.54, 0.65, Form("NPB52(1973)"));

  TLatex *lat5=new TLatex();  lat5->SetNDC(kTRUE);
  lat5->SetTextColor(1); lat5->SetTextFont(42); lat5->SetTextSize(.033);
  lat5->DrawLatex(0.67, 0.72, Form("STAR(2019)"));

  TLatex *lat6=new TLatex();  lat6->SetNDC(kTRUE);
  lat6->SetTextColor(2); lat6->SetTextFont(42); lat6->SetTextSize(.033);
  lat6->DrawLatex(0.79, 0.42, Form("This analysis"));

  TPaveText *pavunc = new TPaveText(0.7,0.12,0.85,0.15,"blNDC");
  pavunc->SetTextFont(43);
  pavunc->SetTextSize(20);
  pavunc->SetBorderSize(0);
  pavunc->SetFillStyle(0);
  pavunc->AddText("Uncertainties: stat. (bars), syst. (boxes)");

  //pavunc->Draw();


  TGraphAsymmErrors *gSpect3;
  gSpect3 = new TGraphAsymmErrors(N,point,bind,err_x,err_x,erry_sumq_l,erry_sumq_h);

  TCanvas *cfit = new TCanvas("cfit","cfit");
  cfit->cd();
  gSpect3->Draw("ape");

  c1->SaveAs("Fig7.eps");
  c1->SaveAs("Fig7.png");
  /*c1->SaveAs("Lifetime_hypertriton_newprel.gif");
  c1->SaveAs("Lifetime_hypertriton_newprel.pdf");
  c1->SaveAs("Lifetime_hypertriton_newprel.eps");
  */
}


void collection_CPT_nuclei(){

  //gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);

  
  const int N=4;
  Float_t mass[N] = {2.8,1.875612,2.95,3.};
  Float_t err_mass[N] = {0,0,0,0};
  Float_t ratio[N] = {-1.5,0.9,1.5,0.9};
  Float_t stat_ratio[N] = {2.6,0.8,2.6,0.5};
  Float_t syst_ratio[N] = {1.2,1.4,1.2,0.4};
  Float_t err_ratio[N];
  for(int index=0;index<N;index++){
    err_ratio[index]=TMath::Sqrt(stat_ratio[index]*stat_ratio[index]+syst_ratio[index]*syst_ratio[index]);
    ratio[index]/=10000;
    err_ratio[index]/=10000;
  }

  Float_t mass_a[1] = {mass[N-1]};
  Float_t err_mass_a[1] = {err_mass[N-1]};
  Float_t ratio_a[1] = {ratio[N-1]};
  Float_t err_ratio_a[1]= {err_ratio[N-1]};
  std::cout<<ratio_a[0]<<std::endl;

  TGraphErrors* graph = new TGraphErrors(N-1,ratio,mass,err_ratio,err_mass);
  TGraphErrors* graph_a = new TGraphErrors(1,ratio_a,mass_a,err_ratio_a,err_mass_a);
  //graph->GetXaxis()->SetLabelOffset(10);
  //graph->GetXaxis()->SetLabelSize(0);

  //systgraph->SetErrorX(3);

  graph->SetMarkerStyle(kMarkTyp); graph->SetMarkerSize(1.4);
  graph->SetMarkerColor(kMarkCol); graph->SetLineColor(kMarkCol);
  graph->SetLineWidth(1);


  graph_a->SetMarkerStyle(kFullDiamond); graph_a->SetMarkerSize(2.8);
  graph_a->SetMarkerColor(kRed); graph_a->SetLineColor(kRed);
  graph_a->SetLineWidth(1);
  //systgraph->SetMarkerColorAlpha(kBlue,0.);
  TCanvas cv("","");

  TH2D *ho1 = new TH2D("ho1", "ho1", 1000, -0.0005, 0.0005, 1000, 1.8, 3.2); //...just for frame
  //ho1->GetYaxis()->SetTitle("{}^{3}_{#Lambda}H Lifetime (ps)");
  ho1->GetYaxis()->SetTitle("Mass (GeV/c^{2})");
  ho1->GetXaxis()->SetTitle("#Delta(m/|q|)/(m/|q|)");
  ho1->GetXaxis()->SetTitleOffset(1.1);
  ho1->GetXaxis()->SetNdivisions(9);
  ho1->GetYaxis()->SetTitleOffset(1.1);
  ho1->GetYaxis()->SetLabelOffset(.01);
  //ho1->SetTitleSize(0.05,"XY");ho1->SetTitleFont(42,"XY");
  //ho1->SetLabelSize(0.04,"XY");ho1->SetLabelFont(42,"XY");
  ho1->SetMarkerStyle(kFullCircle);;
  ho1->SetTitle("");
  ho1->Draw("");

  graph->Draw("pzsame");
  graph_a->Draw("pzsame");

  TLatex *lat1=new TLatex();  lat1->SetNDC(kTRUE);
  lat1->SetTextColor(1); lat1->SetTextFont(42); lat1->SetTextSize(.033);
  lat1->DrawLatex(0.25, 0.63, Form("^{3}He-^{3}#bar{He} (STAR 2019)"));

  TLatex *lat2=new TLatex();  lat2->SetNDC(kTRUE);
  lat2->SetTextColor(1); lat2->SetTextFont(42); lat2->SetTextSize(.033);
  lat2->DrawLatex(0.58, 0.2, Form("d-#bar{d} (ALICE 2015)"));
  
  TLatex *lat3=new TLatex();  lat3->SetNDC(kTRUE);
  lat3->SetTextColor(2); lat3->SetTextFont(42); lat3->SetTextSize(.033);
  lat3->DrawLatex(0.65, 0.83, Form("{}^{3}_{#Lambda}H-^{3}_{#bar{#Lambda}}#bar{H} This project"));

  TLatex *lat4=new TLatex();  lat4->SetNDC(kTRUE);
  lat4->SetTextColor(1); lat4->SetTextFont(42); lat4->SetTextSize(.033);
  lat4->DrawLatex(0.21, 0.76, Form("{}^{3}_{#Lambda}H-^{3}_{#bar{#Lambda}}#bar{H} (STAR 2019)"));

  TLine* line = new TLine(0.,1.8,0.,3.2);
  line->SetLineColor(kBlue);
  line->SetLineWidth(2);
  line->SetLineStyle(7);
  line->Draw("same");
  cv.SaveAs("collection_cpt.png");
}