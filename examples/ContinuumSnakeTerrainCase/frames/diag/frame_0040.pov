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
    ,<0.04780631523329383,-2.913782137449128,-0.23355260452043386>,0.0
    ,<0.058504649562597544,-2.9038124567236396,-0.24006954891988766>,0.001444405933878283
    ,<0.06876460247564616,-2.8933473099820377,-0.24653944969901714>,0.002733688514425582
    ,<0.07829958817791838,-2.882149452931122,-0.25288971000768695>,0.0037941133653625076
    ,<0.0867817404742625,-2.8700626188068576,-0.2591023813541933>,0.0046307451971068355
    ,<0.09386469249452585,-2.8570312962863884,-0.2651551211230724>,0.005283185474353696
    ,<0.09922555553442332,-2.8431051818836557,-0.27097825707577405>,0.005794598874521764
    ,<0.1026235764842005,-2.8284555184560514,-0.2764903105928686>,0.00620058003411749
    ,<0.10394314840540393,-2.8133436121912907,-0.28162608124030486>,0.006527801879788091
    ,<0.1032275057250388,-2.798055957681586,-0.28633795861434114>,0.006795619711330263
    ,<0.10067884039473143,-2.782832139672264,-0.2905935766668492>,0.007018006566011825
    ,<0.09663111562379718,-2.7678099854178733,-0.29437081580636315>,0.007205119848667835
    ,<0.09151153904859857,-2.7530020299778637,-0.29765465038049>,0.007364433711532417
    ,<0.08580554966404838,-2.7383071020305962,-0.3004358526119416>,0.0075015263935279105
    ,<0.08003376128975041,-2.723551671642546,-0.30271221432952006>,0.007620622272343326
    ,<0.07473901293732713,-2.7085511869800145,-0.3044935231215156>,0.007724966207910139
    ,<0.07047309835386044,-2.6931785591172566,-0.30580942749197>,0.007817084460335388
    ,<0.0677750120146344,-2.677423940521341,-0.306707056234337>,0.007898968749670325
    ,<0.06713388851395667,-2.6614348627130213,-0.3072367033798247>,0.007972207813666372
    ,<0.0689305504054656,-2.645525754681181,-0.30744466514258456>,0.008038082702723609
    ,<0.07336367671780204,-2.6301399518980393,-0.30738595196553264>,0.008097636716798745
    ,<0.08039546327896793,-2.615756929123768,-0.30714731009789686>,0.008151727381894005
    ,<0.08975145169851831,-2.602766597868959,-0.3068739970248871>,0.008201065543276747
    ,<0.10098664315803153,-2.5913607286768023,-0.3067149565597203>,0.008246245102718756
    ,<0.11357545499240297,-2.581467286318971,-0.30676227827321>,0.00828776588047385
    ,<0.12700174008519355,-2.572747714519858,-0.3070736522103822>,0.008326051367736582
    ,<0.14079910341936988,-2.5646425301796114,-0.30765162495107035>,0.00836146264109268
    ,<0.15453743685401333,-2.5564594018256663,-0.30845009718680694>,0.008394309364827233
    ,<0.16774666288140588,-2.547461469008391,-0.309385776225927>,0.008424858562469344
    ,<0.17983717210469494,-2.537011161250275,-0.3103651653810335>,0.00845334166411343
    ,<0.19008780731978703,-2.524749975168468,-0.31135023769751113>,0.008479960209706025
    ,<0.19772136028013995,-2.5107092875126953,-0.3123511559754907>,0.008504890496255251
    ,<0.20207645820675704,-2.4953377570695037,-0.31342205918130184>,0.008528287388947346
    ,<0.20280192217485368,-2.4793907507123385,-0.31465273616461514>,0.008550287465601714
    ,<0.19997073119753592,-2.4637055263677397,-0.3161568130694793>,0.008571011625971648
    ,<0.1940574949644459,-2.4489551900965925,-0.31806343003328896>,0.00859056726871202
    ,<0.18577861739060186,-2.4354758727613293,-0.320469490337525>,0.008609050116966811
    ,<0.1759239171097222,-2.4232006622086115,-0.3233482537226305>,0.008626545756733304
    ,<0.16525500696434467,-2.4117076021570045,-0.3265454355005471>,0.008643130939168025
    ,<0.15445444377238707,-2.400383874003635,-0.3298963398345916>,0.00865887468788217
    ,<0.144139861823602,-2.38862538263983,-0.33328024174286053>,0.008673839244344611
    ,<0.13492064682280364,-2.3759768253513016,-0.336612967201828>,0.008688080878257348
    ,<0.1273968404662608,-2.3622282386688163,-0.3398478962982687>,0.008701650584808223
    ,<0.12209910252058062,-2.347454399501388,-0.34297327150274715>,0.008714594686749191
    ,<0.11939305463627406,-2.3319772029079773,-0.34601280731522105>,0.008726955356075762
    ,<0.11939704756257484,-2.316260597511719,-0.3490265619366816>,0.008738771067525925
    ,<0.12198505055566952,-2.3007611322482546,-0.35204463281292847>,0.008750076994045604
    ,<0.1268223834516477,-2.28580848166772,-0.3550494115214679>,0.008760905352682195
    ,<0.1334054657723883,-2.27153218729862,-0.35802293953502995>,0.008771285707989934
    ,<0.14113884866819068,-2.25783694606257,-0.36095876134615823>,0.008781245238899917
    ,<0.14938517831071355,-2.244440657328027,-0.3638788686474425>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
