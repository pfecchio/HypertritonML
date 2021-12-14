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

constexpr double kAliceResult{0.077};
constexpr double kAliceResultStat[2]{0.063, 0.063}; // first + then -
constexpr double kAliceResultSyst[2]{0.031, 0.031}; // first + then -

void CollectionPlotMassVert()
{
    const Int_t kLineWidth = 2;

    gStyle->SetOptStat(0);

    constexpr float kOffset = -0.5;
    constexpr int nMeasures{6};
    constexpr int nPublished{5};

    const Int_t N = nPublished; // number of blam values
    Float_t point[N] = {1, 2, 3, 4, 5};
    Float_t bind[N] = {0.41, 0.08, -0.24, 0.27, 0.4};
    Float_t err_y[N] = {0, 0, 0, 0, 0};
    Float_t err_x_low[N] = {0.12, 0.07, 0.22, 0.08, 0.12};
    Float_t err_x_high[N] = {0.12, 0.07, 0.22, 0.08, 0.12};
    Float_t errsyst_x[N] = {0.11, 0., 0., 0., 0.};
    Float_t errsyst_y[N] = {0.11, 0, 0, 0, 0.};

    double w[N] = {0, 0, 0, 0, 0};
    double v[N] = {0, 0, 0, 0, 0};
    double s[N] = {0, 0, 0, 0, 0};
    double sum_w = 0;
    double sum_v = 0;
    double chi2 = 0;

    const Int_t kMarkTyp = 20;              // marker type
    const Int_t kMarkCol = 1;               //...and color
    const Float_t kTitSize = 0.055;         // axis title size
    const Float_t kAsize = 0.85 * kTitSize; //...and label size
    const Float_t kToffset = 0.8;
    const Int_t kFont = 42;

    TCanvas *cfit = new TCanvas("cfit", "cfit");
    cfit->cd();

    // cout << "Sum of weights after all it : " << sum_w << endl;
    // cout << "Sum of weighted measurements after all it : " << sum_v << endl;
    // cout << "Final value : " << sum_v/sum_w << endl;
    // cout << "Uncertainty : " << 1/sqrt(sum_w) << endl;
    // cout << "Chi2 : " << chi2 << endl;
    // cout << "Chi2/(N-1) : " << chi2/(N-1) << endl;

    TCanvas *cv = new TCanvas("cv", "blam collection", 700, 867);
    // cv->SetMargin(0.340961, 0.0514874, 0.17, 0.070162);
    cv->SetMargin(0.300961, 0.0514874, 0.121294, 0.140162);
    TH2D *frame = new TH2D("frame", ";B_{#Lambda} (MeV);", 1000, -0.55, 0.75, nMeasures, kOffset, kOffset + nMeasures);
    std::string names[nMeasures]{"NPB1 (1967) 105", "NPB4 (1968) 511", "PRD1 (1970) 66", "NPB52 (1973) 1", "Nat. Phys 16 (2020)"};

    names[nMeasures - 1] = "ALICE Pb#minusPb 5.02 TeV";

    std::reverse(std::begin(names), std::end(names));
    for (int i{0}; i < nMeasures; ++i)
    {
        if (i == 0)
            frame->GetYaxis()->SetBinLabel(i + 1, Form("#color[%d]{%s}", 2, names[i].data()));
        else
            frame->GetYaxis()->SetBinLabel(i + 1, names[i].data());
    }

    frame->Draw("col");

    TGraphAsymmErrors *gSpect;
    gSpect = new TGraphAsymmErrors(N, bind, point, err_x_low, err_x_high, err_y, err_y);

    TGraphAsymmErrors *gSpect2; // syst. err.
    gSpect2 = new TGraphAsymmErrors(N, bind, point, errsyst_x, errsyst_x, errsyst_y, errsyst_y);

    Float_t point_a[1] = {0};
    Float_t blam_a[1] = {kAliceResult};
    Float_t err_y_a[1] = {0};
    Float_t err_x_low_a[1] = {kAliceResultStat[0]};
    Float_t err_x_high_a[1] = {kAliceResultStat[1]};
    Float_t errsyst_x_a[1] = {kAliceResultSyst[0]};
    Float_t errsyst_y_a[1] = {0.1};
    TGraphAsymmErrors *gSpect_alice = new TGraphAsymmErrors(1, blam_a, point_a, err_x_low_a, err_x_high_a, err_y_a, err_y_a);
    TGraphAsymmErrors *gSpect2_alice = new TGraphAsymmErrors(1, blam_a, point_a, errsyst_x_a, errsyst_x_a, errsyst_y_a, errsyst_y_a);

    const int kBCT = TColor::GetColor("#b8d4ff");

    TBox *boxtheo = new TBox(0.046, kOffset, 0.135, nMeasures + kOffset);
    boxtheo->SetFillColorAlpha(kBCT, 0.7);
    boxtheo->SetFillStyle(1001);
    boxtheo->SetLineColor(kWhite);
    boxtheo->SetLineWidth(0);

    TLine *theo1 = new TLine(0.10, kOffset, 0.10, nMeasures + kOffset);
    theo1->SetLineWidth(kLineWidth);
    theo1->SetLineStyle(2);
    theo1->SetLineColor(kPurpleC);

    TLine *theo2 = new TLine(0.262, kOffset, 0.262, nMeasures + kOffset);
    theo2->SetLineWidth(kLineWidth);
    theo2->SetLineStyle(9);
    theo2->SetLineColor(kOrangeC);

    TLine *theo3 = new TLine(0.23, kOffset, 0.23, nMeasures + kOffset);
    theo3->SetLineWidth(kLineWidth);
    theo3->SetLineStyle(10);
    theo3->SetLineColor(kMagentaC);

    gSpect->SetMarkerStyle(kMarkTyp);
    gSpect->SetMarkerSize(1.6);
    gSpect->SetMarkerColor(kMarkCol);
    gSpect->SetLineColor(kMarkCol);

    gSpect2->SetMarkerStyle(0);
    gSpect2->SetMarkerColor(kMarkCol);
    gSpect2->SetMarkerSize(0.1);
    gSpect2->SetLineStyle(1);
    gSpect2->SetLineColor(kMarkCol);
    gSpect2->SetFillColor(0);
    gSpect2->SetFillStyle(0);

    gSpect_alice->SetMarkerStyle(kFullSquare);
    gSpect_alice->SetMarkerSize(1.);
    gSpect_alice->SetMarkerColor(kRed);
    gSpect_alice->SetLineColor(kRed);

    gSpect2_alice->SetMarkerStyle(0);
    gSpect2_alice->SetMarkerColor(kRed);
    gSpect2_alice->SetMarkerSize(0.1);
    gSpect2_alice->SetLineStyle(1);
    gSpect2_alice->SetLineColor(kRed);
    gSpect2_alice->SetFillColor(0);
    gSpect2_alice->SetFillStyle(0);

    boxtheo->Draw("same");

    theo1->Draw("SAME");
    theo2->Draw("SAME");
    theo3->Draw("SAME");

    gSpect->Draw("pzsame");
    gSpect2->Draw("spe2");

    gSpect_alice->Draw("pzsame");
    gSpect2_alice->Draw("spe2");

    TLegend *leg2 = new TLegend(0.30, 0.87062, 0.949085 -0.1, 0.985175);
    leg2->SetFillStyle(0);
    leg2->SetMargin(0.16); // separation symbol-text
    leg2->SetBorderSize(0);
    leg2->SetTextFont(42);
    leg2->SetTextSize(0.022);
    leg2->SetNColumns(2);
    leg2->SetHeader("Theoretical predictions");
    leg2->AddEntry(theo1, "NPB47 (1972) 109-137", "fl");
    leg2->AddEntry(theo2, "PRC77 (2008) 027001  ", "fl");
    leg2->AddEntry(theo3, "arXiv:1711.07521", "fl");
    leg2->AddEntry(boxtheo, "EPJA(2020) 56", "f");

    leg2->Draw();

    cv->SaveAs("../Results/2Body/CollectionMass.eps");
    cv->SaveAs("../Results/2Body/CollectionMass.pdf");
    cv->SaveAs("../Results/2Body/CollectionMass.png");
}
