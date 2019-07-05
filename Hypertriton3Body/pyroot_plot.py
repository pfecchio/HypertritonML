import collections
import random
import string

from ROOT import TH1, TCanvas, TColor, TGaxis, TPad

# custom colors
kBlueC = TColor.GetColor('#1f78b4')
kBlueCT = TColor.GetColorTransparent(kBlueC, 0.5)
kRedC = TColor.GetColor('#e31a1c')
kRedCT = TColor.GetColorTransparent(kRedC, 0.5)
kPurpleC = TColor.GetColor('#911eb4')
kPurpleCT = TColor.GetColorTransparent(kPurpleC, 0.5)
kOrangeC = TColor.GetColor('#ff7f00')
kOrangeCT = TColor.GetColorTransparent(kOrangeC, 0.5)
kGreenC = TColor.GetColor('#33a02c')
kGreenCT = TColor.GetColorTransparent(kGreenC, 0.5)
kMagentaC = TColor.GetColor('#f032e6')
kMagentaCT = TColor.GetColorTransparent(kMagentaC, 0.5)
kYellowC = TColor.GetColor('#ffe119')
kYellowCT = TColor.GetColorTransparent(kYellowC, 0.5)
kBrownC = TColor.GetColor('#b15928')
kBrownCT = TColor.GetColorTransparent(kBrownC, 0.5)

colors = (kBlueC,  kRedC, kGreenC, kOrangeC, kPurpleC, kMagentaC, kYellowC, kBrownC)


def histo_makeup(
        histo, opt='e', stat=False, color=4, x_title='', y_title='', l_width=2, m_size=0, m_style=20, x_range=(0, 0),
        y_range=(0, 0)):

    histo.SetOption(opt)
    histo.SetStats(stat)

    histo.SetLineColor(color)
    histo.SetLineWidth(l_width)

    histo.SetMarkerColor(color)
    histo.SetMarkerStyle(m_style)
    histo.SetMarkerSize(m_size)

    histo.GetXaxis().SetTitle(x_title)
    histo.GetYaxis().SetTitle(y_title)

    if not isinstance(x_range, collections.abc.Sequence):
        raise TypeError("The variable 'x_range' has a wrong type")
    elif len(x_range) != 2:
        raise ValueError("Wrong length given for list 'x_range'")

    if not isinstance(y_range, collections.abc.Sequence):
        raise TypeError("The variable 'y_range' has a wrong type")
    elif len(x_range) != 2:
        raise ValueError("Wrong length given for list 'y_range'")

    if x_range != (0, 0):
        histo.GetXaxis().SetRangeUser(x_range[0], x_range[1])
    if y_range != (0, 0):
        histo.GetYaxis().SetRangeUser(y_range[0], y_range[1])


def ratio_plot(h1, h2, legend, name='c', dim=(800, 600), mode='', l_range=(0, 0), l_y_title=''):

    if not isinstance(dim, collections.abc.Sequence):
        raise TypeError("The variable 'dim' has a wrong type")
    elif len(dim) != 2:
        raise ValueError("Wrong length given for list 'dim'")

    if not isinstance(l_range, collections.abc.Sequence):
        raise TypeError("The variable 'l_range' has a wrong type")
    elif len(l_range) != 2:
        raise ValueError("Wrong length given for list 'l_range'")

    if name == 'c':
        name = name + '_' + random_string()

    c = TCanvas(name, '', dim[0], dim[1])

    pad1 = TPad('pad1', 'pad1', 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0)  # Upper and lower plot are joined
    pad1.Draw()             # Draw the upper pad: pad1
    pad1.cd()               # pad1 becomes the current pad

    h1.SetStats(0)
    h1.Draw()

    h2opt = h2.GetOption() + 'SAME'
    h2.Draw(h2opt)

    legend.Draw()

    axis = TGaxis(-5, 20, -5, 220, 20, 220, 510, '')
    axis.SetLabelFont(43)  # Absolute font size in pixel (precision 3)
    axis.SetLabelSize(20)
    axis.SetTitleFont(43)
    axis.SetTitleSize(20)
    axis.SetTitleOffset(1.3)
    axis.Draw()
    c.Update()

    # lower plot will be in pad
    c.cd()          # Go back to the main canvas before defining pad2
    pad2 = TPad('pad2', 'pad2', 0, 0.01, 1., 0.30)
    pad2.SetTopMargin(0.0)
    pad2.SetBottomMargin(0.28)
    pad2.SetGridy(1)
    pad2.Draw()
    pad2.cd()       # pad2 becomes the current pad

    # Define the ratio plot
    h3 = h1.Clone('h3')
    h3.SetLineColor(kPurpleC)
    h3.SetMarkerColor(kPurpleC)
    h3.SetMarkerSize(0.5)
    h3.Sumw2()
    h3.SetStats(0)  # No statistics on lower plot

    if (mode == 'ratio'):
        h3.Divide(h2)
    if (mode == 'diff'):
        h3.Add(h2, -1.)

    h3.SetMarkerStyle(21)
    h3.Draw('ep')       # Draw the ratio plot

    # Remove the ratio title
    h3.SetTitle('')

    # Y axis ratio plot settings
    l_title = ''
    if l_y_title == '':
        l_title = mode + '   '
    else:
        l_title = l_y_title

    h3.GetYaxis().SetTitle(l_title)
    h3.GetYaxis().SetNdivisions(505)
    h3.GetYaxis().SetTitleSize(20)
    h3.GetYaxis().SetTitleFont(43)
    h3.GetYaxis().SetTitleOffset(1.2)
    h3.GetYaxis().SetLabelFont(43)
    # Absolute font size in pixel(precision 3)
    h3.GetYaxis().SetLabelSize(16)
    if l_range != (0, 0):
        h3.GetYaxis().SetRangeUser(l_range[0], l_range[1])

    # X axis ratio plot settings
    h3.GetXaxis().SetTitleSize(20)
    h3.GetXaxis().SetTitleFont(43)
    h3.GetXaxis().SetTitleOffset(3.5)
    h3.GetXaxis().SetLabelFont(43)
    # Absolute font size in pixel(precision 3)
    h3.GetXaxis().SetLabelSize(18)

    c.Write()
    c.Close()
    # return c


def make_histo_canvas(dim=(800, 600), title='c', x_title='', y_title='', x_range=(0, 0), y_range=(0, 0)):
    if not isinstance(dim, collections.abc.Sequence):
        raise TypeError("The variable 'x_range' has a wrong type")
    elif len(dim) != 2:
        raise ValueError("Wrong length given for list 'x_range'")
    c = TCanvas(title, '', dim[0], dim[1])

    return c


def random_string(stringLength=3):
    '''Generate a random string of fixed length '''
    letters = string.ascii_lowercase
    return ''.join(random.sample(letters, stringLength))
