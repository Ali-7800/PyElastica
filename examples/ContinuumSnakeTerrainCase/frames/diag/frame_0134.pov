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
    ,<0.3083656772434931,-2.237723706182968,-0.36624638948026117>,0.0
    ,<0.3022680261395817,-2.222952603272937,-0.36698850111760095>,0.001444405933878283
    ,<0.29470471369265766,-2.2088768284395375,-0.367739356734629>,0.002733688514425582
    ,<0.2861725078191989,-2.195361463003556,-0.36851202924922066>,0.0037941133653625076
    ,<0.27722881238440655,-2.182107371021798,-0.36924872632011235>,0.0046307451971068355
    ,<0.26846212472184366,-2.1687297078244736,-0.3698392531309363>,0.005283185474353696
    ,<0.26048770097515644,-2.154859417973987,-0.37021435695492916>,0.005794598874521764
    ,<0.25396099069279604,-2.140249552241759,-0.3703330153854977>,0.00620058003411749
    ,<0.2495546320471982,-2.124868459387063,-0.3701826871348223>,0.006527801879788091
    ,<0.24788852171707182,-2.108960825154618,-0.36978243091760143>,0.006795619711330263
    ,<0.249421014636965,-2.093046724389425,-0.3691875004427948>,0.007018006566011825
    ,<0.2543370580139166,-2.0778385233028938,-0.3684927243311195>,0.007205119848667835
    ,<0.2624861070482508,-2.064087414793174,-0.36783202141668425>,0.007364433711532417
    ,<0.2734039635138605,-2.052402480468451,-0.36737075612240105>,0.0075015263935279105
    ,<0.2864264802191459,-2.0431106191030763,-0.3672999446679653>,0.007620622272343326
    ,<0.3008183456860385,-2.036139588967609,-0.3677923944569507>,0.007724966207910139
    ,<0.31591380683438985,-2.0309598492346375,-0.36897457366743036>,0.007817084460335388
    ,<0.3312219478731465,-2.0266884440128625,-0.3708899792421768>,0.007898968749670325
    ,<0.3463900063938558,-2.022285038235853,-0.3735095843499225>,0.007972207813666372
    ,<0.3610390147320391,-2.016724366354366,-0.3767912673376319>,0.008038082702723609
    ,<0.37456674064598616,-2.009110035580677,-0.38068732475576833>,0.008097636716798745
    ,<0.38602137328899816,-1.9988642523294549,-0.3851332276860073>,0.008151727381894005
    ,<0.39415015216426946,-1.9859854091616191,-0.39000721378844944>,0.008201065543276747
    ,<0.39768763774320615,-1.9712524723727392,-0.39509296272344685>,0.008246245102718756
    ,<0.3958279516432605,-1.9561864355380731,-0.4000705065147025>,0.00828776588047385
    ,<0.3886389835171426,-1.9426480636938457,-0.40455758833612443>,0.008326051367736582
    ,<0.37711615825674155,-1.9322035321944653,-0.4081951404901949>,0.00836146264109268
    ,<0.36279127888248186,-1.9256066389145894,-0.4107365334237927>,0.008394309364827233
    ,<0.3471416321069968,-1.9226797904071018,-0.4120929349408489>,0.008424858562469344
    ,<0.3311592369763844,-1.922564409533417,-0.4123236315151322>,0.00845334166411343
    ,<0.31525537930865943,-1.9240722154272998,-0.4115783227780804>,0.008479960209706025
    ,<0.2994408695034239,-1.925884924612976,-0.40998673476654324>,0.008504890496255251
    ,<0.2836340840106395,-1.926690441690834,-0.40766732720541>,0.008528287388947346
    ,<0.26796171057094786,-1.9253235962226816,-0.4047900674029021>,0.008550287465601714
    ,<0.252937915704694,-1.9208882862276984,-0.40159631386671063>,0.008571011625971648
    ,<0.23942883807483487,-1.9129594249126816,-0.39843119238206454>,0.00859056726871202
    ,<0.22839920784768872,-1.901722240759565,-0.3957351740734333>,0.008609050116966811
    ,<0.22056455355005913,-1.8879144147205762,-0.3939849441573895>,0.008626545756733304
    ,<0.21615322738740808,-1.8725624300619206,-0.393566010259015>,0.008643130939168025
    ,<0.2148580880004041,-1.8566543522931749,-0.39456946234616525>,0.00865887468788217
    ,<0.2159389321432125,-1.8408413157240855,-0.39675217311488187>,0.008673839244344611
    ,<0.21845589134307178,-1.8253326562048455,-0.39978110551855156>,0.008688080878257348
    ,<0.22147655215740994,-1.810039363805757,-0.4033910570603707>,0.008701650584808223
    ,<0.22413146837963158,-1.7947724001005096,-0.4073792754968909>,0.008714594686749191
    ,<0.22563302158500326,-1.7794117597931585,-0.4115999015569589>,0.008726955356075762
    ,<0.22530660500720123,-1.7640187699681529,-0.4159523456281498>,0.008738771067525925
    ,<0.22265018490938634,-1.7488730980097842,-0.4203703704651578>,0.008750076994045604
    ,<0.21740452052190115,-1.7344262138784758,-0.42481287532058587>,0.008760905352682195
    ,<0.20959540917365785,-1.7211888559552768,-0.4292565652589986>,0.008771285707989934
    ,<0.199515433349729,-1.7095849791769773,-0.43369247845869685>,0.008781245238899917
    ,<0.18764026580327073,-1.699825012818708,-0.4381286056450164>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
