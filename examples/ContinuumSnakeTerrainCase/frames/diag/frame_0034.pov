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
    ,<0.05305116297855908,-2.942686779480122,-0.22654522559881568>,0.0
    ,<0.05635289750667064,-2.9282021142587853,-0.2324650937239741>,0.001444405933878283
    ,<0.06121863825280112,-2.914162408555698,-0.23838100858231764>,0.002733688514425582
    ,<0.06733511360517096,-2.9006353041306054,-0.24440444677425954>,0.0037941133653625076
    ,<0.07432917681574402,-2.8875824064753917,-0.25056452848917327>,0.0046307451971068355
    ,<0.08186938795359695,-2.874860663857091,-0.2567684128588745>,0.005283185474353696
    ,<0.08960447230625919,-2.8622371625738046,-0.26292918667296894>,0.005794598874521764
    ,<0.09716511228571574,-2.8494362565538007,-0.26893642404913903>,0.00620058003411749
    ,<0.10418623066536482,-2.836221213091695,-0.27469396441903293>,0.006527801879788091
    ,<0.11030485109938727,-2.8224275543908144,-0.28011159946610464>,0.006795619711330263
    ,<0.11518395823885076,-2.807998367731637,-0.285112159377619>,0.007018006566011825
    ,<0.11855037247342558,-2.792992819982359,-0.2896365464810303>,0.007205119848667835
    ,<0.12022927174522496,-2.777564036827765,-0.29364273932703594>,0.007364433711532417
    ,<0.12017227283749737,-2.761916034734255,-0.29710632575635165>,0.0075015263935279105
    ,<0.11846835310102216,-2.7462508864759245,-0.30002111649076435>,0.007620622272343326
    ,<0.11533334443357801,-2.730718647850355,-0.30240107407607253>,0.007724966207910139
    ,<0.11109468636680153,-2.715381089404415,-0.304271228778075>,0.007817084460335388
    ,<0.10616720694607762,-2.7001986976415853,-0.3056602445859709>,0.007898968749670325
    ,<0.10103926813954753,-2.685046733197911,-0.30658053512118294>,0.007972207813666372
    ,<0.09624722565726297,-2.6697643126177164,-0.30704062007132404>,0.008038082702723609
    ,<0.09233251702079719,-2.6542267386566825,-0.307077047967114>,0.008097636716798745
    ,<0.08981693113376592,-2.638404773792841,-0.3067534470925441>,0.008151727381894005
    ,<0.08915716243140982,-2.62240438937408,-0.30616390829227347>,0.008201065543276747
    ,<0.09068189452997372,-2.60646813162302,-0.3054418255748109>,0.008246245102718756
    ,<0.09455294774126238,-2.5909343375607343,-0.30472974566368655>,0.00828776588047385
    ,<0.10073350420856525,-2.5761609853791607,-0.3041639194141047>,0.008326051367736582
    ,<0.10897967487368877,-2.5624254105373816,-0.30388444884676336>,0.00836146264109268
    ,<0.11888729111752433,-2.5498375431323304,-0.3040322978544786>,0.008394309364827233
    ,<0.12994885592652575,-2.5382777248215698,-0.30469633022831855>,0.008424858562469344
    ,<0.14163834390215826,-2.527394449760454,-0.3057975762603319>,0.00845334166411343
    ,<0.15346599670323396,-2.516694969056727,-0.30718346314748585>,0.008479960209706025
    ,<0.16493645179800706,-2.5056368377041647,-0.3087476194568449>,0.008504890496255251
    ,<0.1755198129834407,-2.4937416694279526,-0.3104266460493778>,0.008528287388947346
    ,<0.18464627946942164,-2.480707312159075,-0.3122043699371526>,0.008550287465601714
    ,<0.19175610181881306,-2.46648812735518,-0.31411279709328266>,0.008571011625971648
    ,<0.19640385863117563,-2.451312026380046,-0.3162306624380914>,0.00859056726871202
    ,<0.19835327603169645,-2.435610321462751,-0.31867023775171244>,0.008609050116966811
    ,<0.19750318109483278,-2.4198835639859726,-0.3215153541200643>,0.008626545756733304
    ,<0.19397230097043133,-2.4046110464147916,-0.3247511026249928>,0.008643130939168025
    ,<0.18819019931616085,-2.3901071006459653,-0.3282758195023442>,0.00865887468788217
    ,<0.1807588271744313,-2.376416744654311,-0.33195516651254053>,0.008673839244344611
    ,<0.17229396260770952,-2.3633545798702382,-0.3356811414925983>,0.008688080878257348
    ,<0.16338266985259967,-2.350582837769275,-0.33936973391965874>,0.008701650584808223
    ,<0.15457668190040638,-2.337712396441409,-0.34296522238600796>,0.008714594686749191
    ,<0.1464070777571521,-2.3243965018140367,-0.34643795287455176>,0.008726955356075762
    ,<0.13938711705610954,-2.3104103262838347,-0.3497877301272339>,0.008738771067525925
    ,<0.13401484419047646,-2.295683891756539,-0.3530079140933418>,0.008750076994045604
    ,<0.13072903350943116,-2.280327630799512,-0.356085456796065>,0.008760905352682195
    ,<0.12982907865761265,-2.2646255772015693,-0.3590349335864102>,0.008771285707989934
    ,<0.13142661021395172,-2.248962784574745,-0.3618919068490497>,0.008781245238899917
    ,<0.1354231512277007,-2.23372970615462,-0.3647175779718819>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
