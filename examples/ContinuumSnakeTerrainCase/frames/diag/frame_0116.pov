#include "default.inc"
#include "surface.inc"

camera{
    location <12,-3.2,3>
    angle 30
    look_at <0.0,-3.2,0.0>
    sky <-3,0,12>
    right x*image_width/image_height
}
light_source{
    <15.0,10.5,15.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <0.0,0.0,1000>
    color rgb<1,1,1>
}

sphere_sweep {
    linear_spline 51
    ,<0.22136968876117327,-2.386532960790422,-0.33614439395535034>,0.0
    ,<0.23579560316362164,-2.3819578559745294,-0.34133630461718384>,0.001444405933878283
    ,<0.25042877584790924,-2.3780033836064476,-0.34645735209134415>,0.002733688514425582
    ,<0.26512032712127753,-2.3740388831245203,-0.35139828431851666>,0.0037941133653625076
    ,<0.2796997678756387,-2.369383464526193,-0.3560573358523541>,0.0046307451971068355
    ,<0.2938956766454194,-2.3633778056363957,-0.3603397302237908>,0.005283185474353696
    ,<0.30727640176728305,-2.3554798862944986,-0.36415033202973823>,0.005794598874521764
    ,<0.31924097623506426,-2.34536662271824,-0.36740478910794083>,0.00620058003411749
    ,<0.3290770737509104,-2.3330152320208155,-0.3700067808411326>,0.006527801879788091
    ,<0.33609513586865686,-2.318754336631043,-0.37185399408333064>,0.006795619711330263
    ,<0.33977832341712916,-2.3032190384502456,-0.37290535842687916>,0.007018006566011825
    ,<0.3399128605618964,-2.2872219518287595,-0.37318637679029976>,0.007205119848667835
    ,<0.33663874203875666,-2.271565811767311,-0.372791917114137>,0.007364433711532417
    ,<0.3304006872332778,-2.256860727538206,-0.3718799979758543>,0.0075015263935279105
    ,<0.3218285222115096,-2.2434065123341984,-0.37065974634247134>,0.007620622272343326
    ,<0.3116079099216253,-2.2311663140298674,-0.3693842689036104>,0.007724966207910139
    ,<0.3003836227095327,-2.219826363701109,-0.3682730776613807>,0.007817084460335388
    ,<0.28872986795310424,-2.2088971688754304,-0.36748789277532184>,0.007898968749670325
    ,<0.27719120303926253,-2.197810682536594,-0.36716047142647107>,0.007972207813666372
    ,<0.26636538979371666,-2.186014105925747,-0.3672749048863613>,0.008038082702723609
    ,<0.2569208805434316,-2.173094157121123,-0.3677111158402852>,0.008097636716798745
    ,<0.24953964279570504,-2.158904617780607,-0.3683632173444048>,0.008151727381894005
    ,<0.24484645683368392,-2.143621397308745,-0.3691217339619791>,0.008201065543276747
    ,<0.24327896342793232,-2.12771381599894,-0.3698758034915918>,0.008246245102718756
    ,<0.2449670175222746,-2.111816947613081,-0.37052314749383475>,0.00828776588047385
    ,<0.2496896961814253,-2.096539736205826,-0.37098567106471764>,0.008326051367736582
    ,<0.2569338193662385,-2.0822809546871515,-0.37122630381596866>,0.00836146264109268
    ,<0.2660218587666309,-2.06912012739272,-0.3712626076546934>,0.008394309364827233
    ,<0.2762417880470494,-2.056819072435109,-0.3711759600900378>,0.008424858562469344
    ,<0.286906821400844,-2.044907349409093,-0.37111118459204895>,0.00845334166411343
    ,<0.29726548498847516,-2.0327349371940002,-0.37120703317671166>,0.008479960209706025
    ,<0.3064247800356899,-2.0196322803616,-0.37160208222376073>,0.008504890496255251
    ,<0.3134389969930042,-2.0052670225881406,-0.37227123754647484>,0.008528287388947346
    ,<0.3174110414932586,-1.9897907472756877,-0.37305720198016107>,0.008550287465601714
    ,<0.31766957555260356,-1.9738204021340444,-0.37389059812117953>,0.008571011625971648
    ,<0.31398810240036396,-1.9582811144383905,-0.37473651091770266>,0.00859056726871202
    ,<0.3066722970415429,-1.9440877899181854,-0.37557374671950844>,0.008609050116966811
    ,<0.2964428728213121,-1.9318272418812086,-0.37638541397381037>,0.008626545756733304
    ,<0.284185348373545,-1.921592527191273,-0.37715773539821623>,0.008643130939168025
    ,<0.27071721601109616,-1.9130096060180157,-0.3778828494490852>,0.00865887468788217
    ,<0.256681625982487,-1.905386242090882,-0.3785631754967069>,0.008673839244344611
    ,<0.24257630179426115,-1.897891258230194,-0.37921683867014067>,0.008688080878257348
    ,<0.22885564152569818,-1.889716038906889,-0.37988562832975464>,0.008701650584808223
    ,<0.21602469180632017,-1.8802130795485337,-0.380649758123179>,0.008714594686749191
    ,<0.2046651327711323,-1.8690088885237777,-0.38164722108509525>,0.008726955356075762
    ,<0.19538383898569284,-1.8560527789370582,-0.3830021234130252>,0.008738771067525925
    ,<0.18869363615043264,-1.8416096762050234,-0.3846436064884311>,0.008750076994045604
    ,<0.1848954848405865,-1.826165984593845,-0.3864031614028901>,0.008760905352682195
    ,<0.18399330094495828,-1.8102903482392987,-0.38818309119744376>,0.008771285707989934
    ,<0.1856954843377343,-1.794477390575173,-0.3899317992975423>,0.008781245238899917
    ,<0.18947169202849148,-1.779023635727825,-0.39164122168865284>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
