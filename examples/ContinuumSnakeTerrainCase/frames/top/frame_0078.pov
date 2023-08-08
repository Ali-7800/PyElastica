#include "default.inc"
#include "surface.inc"

camera{
    location <0.0,0.0,20>
    angle 30
    look_at <0.0,0.0,0>
    sky <-1,0,0>
    right x*image_width/image_height
}
light_source{
    <0.0,8.0,5.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <0.0,0.0,1000>
    color rgb<1,1,1>
}

sphere_sweep {
    linear_spline 51
    ,<0.11618353868659831,-2.6287806798190916,-0.30956403161765456>,0.0
    ,<0.12908115628436764,-2.619405527640046,-0.3108865172510306>,0.001444405933878283
    ,<0.1419621424581492,-2.6100127488763323,-0.31224950515410127>,0.002733688514425582
    ,<0.15444623080025088,-2.600109440243628,-0.3136952886208202>,0.0037941133653625076
    ,<0.1661054071787967,-2.589262014024998,-0.315247247758968>,0.0046307451971068355
    ,<0.1764301634147638,-2.5771516576008353,-0.3168975295744276>,0.005283185474353696
    ,<0.1848439087499741,-2.5636514278857025,-0.31860231141467227>,0.005794598874521764
    ,<0.1907793530502355,-2.5488943877129047,-0.32030850551030626>,0.00620058003411749
    ,<0.19378713333183964,-2.533269949993995,-0.3219564727981224>,0.006527801879788091
    ,<0.19365019165083383,-2.5173495025663604,-0.3235002281803332>,0.006795619711330263
    ,<0.1904478929597392,-2.5017421487877995,-0.32491554943458634>,0.007018006566011825
    ,<0.18454133351650231,-2.4869341073865123,-0.3262087167095352>,0.007205119848667835
    ,<0.1764896420938017,-2.4731671389347354,-0.3274227084760694>,0.007364433711532417
    ,<0.16694376183525572,-2.460393950555551,-0.3286483633257891>,0.0075015263935279105
    ,<0.15655747488490782,-2.448299899794299,-0.3299578906502569>,0.007620622272343326
    ,<0.145958267683424,-2.4363945126244504,-0.33133298136632133>,0.007724966207910139
    ,<0.13577773872319288,-2.4241343996377798,-0.3327479474205222>,0.007817084460335388
    ,<0.12670431665754925,-2.4110435393009944,-0.334245644413943>,0.007898968749670325
    ,<0.11950188268928742,-2.3968539188841675,-0.33590912983436033>,0.007972207813666372
    ,<0.11496663000259442,-2.381622736511016,-0.33777429831933004>,0.008038082702723609
    ,<0.11380348497838376,-2.3657962890208735,-0.3398271887080136>,0.008097636716798745
    ,<0.11644191252648926,-2.3501709096299757,-0.3420476540680651>,0.008151727381894005
    ,<0.12288342972891163,-2.335716131194415,-0.3444136779443075>,0.008201065543276747
    ,<0.13266631771458676,-2.3233023706902656,-0.3469068229052519>,0.008246245102718756
    ,<0.14499679370724777,-2.3134469065698893,-0.3495221996735805>,0.00828776588047385
    ,<0.1589849237607711,-2.3061721134808506,-0.3522483564332504>,0.008326051367736582
    ,<0.17387383367525425,-2.30102070878388,-0.3550486296981467>,0.00836146264109268
    ,<0.1891516397917929,-2.297187187029646,-0.35787290101306557>,0.008394309364827233
    ,<0.20451342164006608,-2.293671082281953,-0.36065370837517513>,0.008424858562469344
    ,<0.21970712823352886,-2.2894182894535335,-0.3633230327346308>,0.00845334166411343
    ,<0.23434491615677447,-2.2834617142817875,-0.3658302239827663>,0.008479960209706025
    ,<0.2477788677588283,-2.275087009429866,-0.368150932057544>,0.008504890496255251
    ,<0.2591235831244081,-2.264010630908146,-0.3702866634185898>,0.008528287388947346
    ,<0.2674503571505558,-2.2504961347420935,-0.3722671191818676>,0.008550287465601714
    ,<0.27208107216938937,-2.2353019161786154,-0.3741501048267708>,0.008571011625971648
    ,<0.27282727403020846,-2.219436274299602,-0.3760253425335356>,0.00859056726871202
    ,<0.27001965933110583,-2.2038066259686686,-0.3779309791298843>,0.008609050116966811
    ,<0.26437548624438584,-2.1889581419039823,-0.37980572267632845>,0.008626545756733304
    ,<0.25677390721440224,-2.174996679987635,-0.38157398887288224>,0.008643130939168025
    ,<0.24810081223420288,-2.161655192652202,-0.38318688550204194>,0.00865887468788217
    ,<0.23917961770187027,-2.148457819413779,-0.3846203085941877>,0.008673839244344611
    ,<0.23077955050455118,-2.1349051656370723,-0.38587501253528994>,0.008688080878257348
    ,<0.2236363333012555,-2.1206378452662182,-0.38698342634901634>,0.008701650584808223
    ,<0.21843262645443104,-2.1055480335595553,-0.38801396302425384>,0.008714594686749191
    ,<0.21573184303736148,-2.0898132051129945,-0.38902516046486507>,0.008726955356075762
    ,<0.21589278326439224,-2.0738467498520463,-0.39001745678419447>,0.008738771067525925
    ,<0.21899302992650374,-2.0581809323043045,-0.3909739467979794>,0.008750076994045604
    ,<0.224810340966251,-2.0433067467896016,-0.39190841563475187>,0.008760905352682195
    ,<0.23288646599948307,-2.0295285048199876,-0.392862835732202>,0.008771285707989934
    ,<0.24264001667914478,-2.016883869304282,-0.39384858940094564>,0.008781245238899917
    ,<0.2534598025473061,-2.005138410494273,-0.3948357455199806>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
