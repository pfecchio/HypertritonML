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

void collection_lifetime_hypertriton() {

  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);

  const Int_t N          = 12; // number of lifetime values
  Float_t point[N]       = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Float_t lifetau[N]     = {90, 232, 285, 128, 264, 246, 182, 183, 181, 142, 242, 234};
  Float_t err_x[N]       = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Float_t err_y_low[N]   = {40, 34, 105, 26, 52, 41, 45, 32, 39, 21, 38, 17};
  Float_t err_y_high[N]  = {220, 45, 127, 35, 84, 62, 89, 42, 54, 24, 34, 17};
  Float_t errsyst_y[N]   = {0, 0, 0, 0, 0, 0, 27, 37, 33, 31, 17, 15};
  Float_t erry_sumq_h[N] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Float_t erry_sumq_l[N] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Float_t errsyst_x[N]   = {0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

  double w[N]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double v[N]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double s[N]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double sum_w = 0;
  double sum_v = 0;
  double chi2  = 0;

  const Int_t kMarkTyp   = 20;              // marker type
  const Int_t kMarkCol   = 1;               //...and color
  const Float_t kTitSize = 0.055;           // axis title size
  const Float_t kAsize   = 0.85 * kTitSize; //...and label size
  const Float_t kToffset = 0.8;
  const Int_t kFont      = 42;

  for (int i = 0; i < N; i++) {
    cout << "Lifetime: " << lifetau[i] << endl;
    s[i]           = sqrt(err_y_high[i] * err_y_high[i] + errsyst_y[i] * errsyst_y[i]);
    erry_sumq_h[i] = sqrt(err_y_high[i] * err_y_high[i] + errsyst_y[i] * errsyst_y[i]);
    erry_sumq_l[i] = sqrt(err_y_low[i] * err_y_low[i] + errsyst_y[i] * errsyst_y[i]);
    cout << "Value s" << i << " : " << s[i] << endl;
    w[i] = 1 / (s[i] * s[i]);
    cout << "Value weight" << i << " : " << w[i] << endl;
    v[i] = w[i] * lifetau[i];
    cout << "Value weigthed" << i << " : " << v[i] << endl;
    sum_w = sum_w + w[i];
    sum_v = sum_v + v[i];
    chi2  = chi2 + (w[i] * (206.651 - lifetau[i]) * (206.651 - lifetau[i]));
  }
  cout << "Sum of weights after all it : " << sum_w << endl;
  cout << "Sum of weighted measurements after all it : " << sum_v << endl;
  cout << "Final value : " << sum_v / sum_w << endl;
  cout << "Uncertainty : " << 1 / sqrt(sum_w) << endl;
  cout << "Chi2 : " << chi2 << endl;
  cout << "Chi2/(N-1) : " << chi2 / (N - 1) << endl;

  TGraphAsymmErrors *gSpect;
  gSpect = new TGraphAsymmErrors(N - 1, point, lifetau, err_x, err_x, err_y_low, err_y_high);

  TGraphAsymmErrors *gSpect2; // syst. err.
  gSpect2 = new TGraphAsymmErrors(N - 1, point, lifetau, errsyst_x, errsyst_x, errsyst_y, errsyst_y);

  Float_t point_a[1]      = {12};
  Float_t lifetau_a[1]    = {261};
  Float_t err_x_a[1]      = {0};
  Float_t err_y_low_a[1]  = {11};
  Float_t err_y_high_a[1] = {11};
  Float_t errsyst_y_a[1]  = {6};
  Float_t errsyst_x_a[1]  = {0.1};

  TGraphAsymmErrors *gSpect_alice =
      new TGraphAsymmErrors(1, point_a, lifetau_a, err_x_a, err_x_a, err_y_low_a, err_y_high_a);
  TGraphAsymmErrors *gSpect2_alice =
      new TGraphAsymmErrors(1, point_a, lifetau_a, errsyst_x_a, errsyst_x_a, errsyst_y_a, errsyst_y_a);

  TLine *Fcn = new TLine(0, 263.2, 13, 263.2);
  Fcn->SetLineWidth(2);
  Fcn->SetLineColor(kBlack);
  // TF1 *Fcn1 = new TF1("Fcn1","187.9",0,12);
  TLine *Fcn1 = new TLine(0, 206.335, 13, 206.335);
  //    TF1 *Fcn1 = new TF1("Fcn1","195.9",0,12);
  Fcn1->SetLineWidth(1);
  Fcn1->SetLineStyle(2);
  Fcn1->SetLineColor(kOrange - 3);

  TF1 *Fcn2 = new TF1("Fcn2", "198.5", 0, 13.1);
  Fcn2->SetLineWidth(1.5);
  Fcn2->SetLineStyle(2);
  Fcn2->SetLineColor(kOrange + 2);

  // 1.97085e+02   1.31749e+01
  // 2.08360e+02   1.42092e+01

  TLine *Fcn3 = new TLine(0, 255.5, 13, 255.5);
  Fcn3->SetLineWidth(2);
  Fcn3->SetLineStyle(5);
  Fcn3->SetLineColor(kBlue);
  TLine *Fcn4 = new TLine(0, 239.3, 13, 239.3);
  Fcn4->SetLineWidth(2);
  Fcn4->SetLineStyle(10);
  Fcn4->SetLineColor(kCyan - 7);
  TLine *Fcn5 = new TLine(0, 213, 13, 213);
  Fcn5->SetLineWidth(2);
  Fcn5->SetLineStyle(7);
  // Fcn5->SetLineColor(kMagenta-7);
  Fcn5->SetLineColor(kMagenta + 2);
  TLine *Fcn6 = new TLine(0, 232, 13, 232);
  Fcn6->SetLineWidth(2);
  Fcn6->SetLineStyle(9);
  Fcn6->SetLineColor(kGreen + 2);

  TCanvas *c1 = new TCanvas("rawyield", "rawyields", 20, 20, 1600, 1024);
  // c1->SetTopMargin(0.03); c1->SetBottomMargin(0.145);
  c1->SetLeftMargin(0.15);
  c1->SetRightMargin(0.1);
  c1->cd(1); // pad->SetLogy();

  TH2D *ho1 = new TH2D("ho1", "ho1", 1000, 0, 13, 1000, 0, 510); //...just for frame
  // ho1->GetYaxis()->SetTitle("{}^{3}_{#Lambda}H Lifetime (ps)");
  ho1->GetYaxis()->SetTitle("Lifetime (ps)");
  ho1->GetXaxis()->SetTitleOffset(1.1);
  ho1->GetXaxis()->SetNdivisions(000);
  ho1->GetYaxis()->SetTitleOffset(1.1);
  ho1->GetYaxis()->SetLabelOffset(.01);
  ho1->SetTitleSize(0.05, "XY");
  ho1->SetTitleFont(42, "XY");
  ho1->SetLabelSize(0.04, "XY");
  ho1->SetLabelFont(42, "XY");
  ho1->SetMarkerStyle(kFullCircle);
  ;
  ho1->SetTitle("");
  ho1->Draw("");

  double x[]       = {0, 13};
  double y[]       = {206.335, 206.335};
  double ex[]      = {0, 0};
  double ey[]      = {13.498, 12.806};
  TGraphErrors *ge = new TGraphErrors(2, x, y, ex, ey);
  ge->SetFillColorAlpha(kOrange - 3, 0.35);
  ge->SetFillStyle(3004);
  ge->Draw("3SAME");

  TLine *Fcn1_up = new TLine(0, 219.83, 13, 219.83);
  Fcn1_up->SetLineWidth(1);
  Fcn1_up->SetLineColor(kOrange - 3);
  TLine *Fcn1_low = new TLine(0, 193.529, 13, 193.529);
  Fcn1_low->SetLineWidth(1);
  Fcn1_low->SetLineColor(kOrange - 3);

  TBox *boxFcn1 = new TBox(0, 0, 0, 0);
  boxFcn1->SetFillColorAlpha(kOrange - 3, 0.35);
  boxFcn1->SetFillStyle(3004);
  boxFcn1->SetLineWidth(1);
  boxFcn1->SetLineStyle(2);
  boxFcn1->SetLineColor(kOrange - 3);
  boxFcn1->Draw("same");

  gSpect->SetMarkerStyle(kMarkTyp);
  gSpect->SetMarkerSize(1.4);
  gSpect->SetMarkerColor(kMarkCol);
  gSpect->SetLineColor(kMarkCol);
  gSpect->SetLineWidth(1.4);

  gSpect2->SetMarkerStyle(0);
  gSpect2->SetMarkerColor(kMarkCol);
  gSpect2->SetMarkerSize(0.1);
  gSpect2->SetLineStyle(1);
  gSpect2->SetLineColor(kMarkCol);
  gSpect2->SetLineWidth(1.4);
  gSpect2->SetFillColor(0);
  gSpect2->SetFillStyle(0);

  gSpect_alice->SetMarkerStyle(kFullDiamond);
  gSpect_alice->SetMarkerSize(2.8);
  gSpect_alice->SetMarkerColor(kRed);
  gSpect_alice->SetLineColor(kRed);
  gSpect_alice->SetLineWidth(1.4);

  gSpect2_alice->SetMarkerStyle(0);
  gSpect2_alice->SetMarkerColor(kRed);
  gSpect2_alice->SetMarkerSize(0.1);
  gSpect2_alice->SetLineStyle(1);
  gSpect2_alice->SetLineColor(kRed);
  gSpect2_alice->SetLineWidth(1.4);
  gSpect2_alice->SetFillColor(0);
  gSpect2_alice->SetFillStyle(0);

  Fcn->Draw("SAME");
  Fcn1->Draw("SAME");
  Fcn1_up->Draw("SAME");
  Fcn1_low->Draw("SAME");
  Fcn3->Draw("SAME");
  Fcn4->Draw("SAME");
  Fcn5->Draw("SAME");
  Fcn6->Draw("SAME");
  gSpect->Draw("pzsame");
  gSpect2->Draw("spe2");
  gSpect_alice->Draw("pzsame");
  gSpect2_alice->Draw("spe2");

  TLegend *leg1 = new TLegend(.18, 0.75, .53, 0.9);
  leg1->SetFillStyle(0);
  leg1->SetMargin(0.16); // separation symbol-text
  leg1->SetBorderSize(0);
  leg1->SetTextFont(42);
  leg1->SetTextSize(0.025);
  // leg1->SetEntrySeparation(0.1);
  leg1->AddEntry(Fcn, "#Lambda lifetime - PDG value", "l");
  leg1->AddEntry(boxFcn1, "{}^{3}_{#Lambda}H average lifetime", "fl");
  // leg1->AddEntry(gSpect, "experimental value","p");

  TLegend *leg2 = new TLegend(.5, 0.65, .9, 0.9);
  leg2->SetFillStyle(0);
  leg2->SetMargin(0.16); // separation symbol-text
  leg2->SetBorderSize(0);
  leg2->SetTextFont(42);
  leg2->SetTextSize(0.022);
  // leg2->SetEntrySeparation(0.05);
  leg2->AddEntry("", "Theoretical prediction", "");
  leg2->AddEntry(Fcn3, "H. Kamada #it{et al.}, PRC 57 (1998) 1595", "l");
  leg2->AddEntry(Fcn4, "R.H. Dalitz, M. Rayet, Nuo. Cim. 46 (1966) 786", "l");
  leg2->AddEntry(Fcn6, "J. G. Congleton, J. Phys G Nucl. Part. Phys. 18 (1992) 339", "l");
  leg2->AddEntry(Fcn5, "A. Gal, H. Garcilazo, PLB 791 (2019) 48-53", "l");

  TLatex *lat = new TLatex();
  lat->SetNDC(kTRUE);
  lat->SetTextColor(1);
  lat->SetTextFont(42);
  lat->SetTextSize(.02);
  lat->DrawLatex(0.16, 0.15, Form("PR 136 (1964) B1803"));
  //  lat->DrawLatex(0.05, 0.2, Form("Phys.Rev. 136 (1964) B1803"));

  TLatex *lat0 = new TLatex();
  lat0->SetNDC(kTRUE);
  lat0->SetTextColor(1);
  lat0->SetTextFont(42);
  lat0->SetTextSize(.02);
  lat0->DrawLatex(0.230, 0.55, Form("PRL 20 (1968) 819"));
  // lat0->DrawLatex(0.1, 0.2, Form("Phys.Rev.Lett. 20 (1968) 819"));

  TLatex *lat1 = new TLatex();
  lat1->SetNDC(kTRUE);
  lat1->SetTextColor(1);
  lat1->SetTextFont(42);
  lat1->SetTextSize(.02);
  lat1->DrawLatex(0.27, 0.35, Form("PR 180 (1969) 1307"));
  // lat1->DrawLatex(0.2, 0.2, Form("Phys.Rev. 180 (1969) 1307"));

  TLatex *lat2 = new TLatex();
  lat2->SetNDC(kTRUE);
  lat2->SetTextColor(1);
  lat2->SetTextFont(42);
  lat2->SetTextSize(.02);
  lat2->DrawLatex(0.345, 0.225, Form("NPB 16 (1970) 46"));
  //  lat2->DrawLatex(0.3, 0.2, Form("Nucl.Phys.B 16 (1970) 46"));

  TLatex *lat3 = new TLatex();
  lat3->SetNDC(kTRUE);
  lat3->SetTextColor(1);
  lat3->SetTextFont(42);
  lat3->SetTextSize(.02);
  lat3->DrawLatex(0.40, 0.66, Form("PRD 1 (1970) 66"));
  //  lat3->DrawLatex(0.4, 0.2, Form("Phys.Rev.D 1 (1970) 66"));

  TLatex *lat4 = new TLatex();
  lat4->SetNDC(kTRUE);
  lat4->SetTextColor(1);
  lat4->SetTextFont(42);
  lat4->SetTextSize(.02);
  lat4->DrawLatex(0.45, 0.35, Form("NPB 67 (1973) 269"));
  //  lat4->DrawLatex(0.5, 0.2, Form("Nucl.Phys.B 67 (1973) 269"));

  TLatex *lat5 = new TLatex();
  lat5->SetNDC(kTRUE);
  lat5->SetTextColor(1);
  lat5->SetTextFont(42);
  lat5->SetTextSize(.02);
  lat5->DrawLatex(0.515, 0.55, Form("Science 328 (2010) 58"));

  TLatex *lat6 = new TLatex();
  lat6->SetNDC(kTRUE);
  lat6->SetTextColor(1);
  lat6->SetTextFont(42);
  lat6->SetTextSize(.02);
  lat6->DrawLatex(0.56, 0.302, Form("NPA 913 (2013) 170"));
  //  lat6->DrawLatex(0.7, 0.2, Form("Nucl.Phys.A 913 (2013) 170"));

  TLatex *lat7 = new TLatex();
  lat7->SetNDC(kTRUE);
  lat7->SetTextColor(1);
  lat7->SetTextFont(42);
  lat7->SetTextSize(.02);
  lat7->DrawLatex(0.62, 0.272, Form("PLB 754 (2016) 360"));
  //  lat7->DrawLatex(0.8, 0.2, Form("Phys.Lett.B 754 (2016) 360"));

  TLatex *lat8 = new TLatex();
  lat8->SetNDC(kTRUE);
  lat8->SetTextColor(1);
  lat8->SetTextFont(42);
  lat8->SetTextSize(.02);
  lat8->DrawLatex(0.69, 0.238, Form("PRC 97 (2018) 054909"));

  TLatex *lat9 = new TLatex();
  lat9->SetNDC(kTRUE);
  lat9->SetTextColor(1);
  lat9->SetTextFont(42);
  lat9->SetTextSize(.02);
  lat9->DrawLatex(0.775, 0.38, Form("ALICE"));
  TLatex *lat9_2 = new TLatex();
  lat9_2->SetNDC(kTRUE);
  lat9_2->SetTextColor(1);
  lat9_2->SetTextFont(42);
  lat9_2->SetTextSize(.02);
  lat9_2->DrawLatex(0.755, 0.355, Form("Pb#minusPb 5.02 TeV"));

  TLatex *lat10 = new TLatex();
  lat10->SetNDC(kTRUE);
  lat10->SetTextColor(2);
  lat10->SetTextFont(42);
  lat10->SetTextSize(.02);
  lat10->DrawLatex(0.820, 0.540, Form("#bf{ALICE Internal}"));

  leg1->Draw();
  leg2->Draw();

  TPaveText *pavunc = new TPaveText(0.7, 0.12, 0.85, 0.15, "blNDC");
  pavunc->SetTextFont(43);
  pavunc->SetTextSize(20);
  pavunc->SetBorderSize(0);
  pavunc->SetFillStyle(0);
  pavunc->AddText("Uncertainties: stat. (bars), syst. (boxes)");

  // pavunc->Draw();

  TGraphAsymmErrors *gSpect3;
  gSpect3 = new TGraphAsymmErrors(N, point, lifetau, err_x, err_x, erry_sumq_l, erry_sumq_h);

  TCanvas *cfit = new TCanvas("cfit", "cfit");
  cfit->cd();
  gSpect3->Draw("ape");

  c1->SaveAs("Fig7.eps");
  c1->SaveAs("Fig7.pdf");
  c1->SaveAs("Fig7.png");

  /*c1->SaveAs("Lifetime_hypertriton_newprel.gif");
  c1->SaveAs("Lifetime_hypertriton_newprel.pdf");
  c1->SaveAs("Lifetime_hypertriton_newprel.eps");
  */
}
