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
    ,<0.11623958182183439,-2.765083787017774,-0.3035486778171356>,0.0
    ,<0.1053157113808686,-2.7534232374757406,-0.3043862526145734>,0.001444405933878283
    ,<0.09579733757779497,-2.740590628419873,-0.3052392420966204>,0.002733688514425582
    ,<0.08830131396523283,-2.7264817493213136,-0.3061106980405551>,0.0037941133653625076
    ,<0.08341729542727372,-2.711268782686204,-0.3069630650745006>,0.0046307451971068355
    ,<0.08160061577225813,-2.6953915449979244,-0.3077568123401625>,0.005283185474353696
    ,<0.08306551615402412,-2.679473816428005,-0.3084663381311441>,0.005794598874521764
    ,<0.08772557087913871,-2.6641786327549464,-0.30908128718561334>,0.00620058003411749
    ,<0.09521392096896014,-2.65004753386371,-0.30961653148508994>,0.006527801879788091
    ,<0.10497354810540858,-2.637376639321624,-0.3101182181360694>,0.006795619711330263
    ,<0.11637704356740823,-2.6261638028460355,-0.3106635522939273>,0.007018006566011825
    ,<0.1288258211452035,-2.616130895145393,-0.3113420348696117>,0.007205119848667835
    ,<0.14179398886286934,-2.606794793550641,-0.3122341197587989>,0.007364433711532417
    ,<0.15481480418555132,-2.59755629518599,-0.3133572084409685>,0.0075015263935279105
    ,<0.16743018554197653,-2.5877965127031426,-0.31467557000424357>,0.007620622272343326
    ,<0.1791130960131118,-2.576958301964119,-0.3161441128689894>,0.007724966207910139
    ,<0.1892252074021843,-2.5646545258376143,-0.3177079074917799>,0.007817084460335388
    ,<0.1970330813681609,-2.5507796991386558,-0.31931022334012993>,0.007898968749670325
    ,<0.20180432501321194,-2.535590132900337,-0.3208892504908714>,0.007972207813666372
    ,<0.20299162912605545,-2.519707748739382,-0.32239316920742883>,0.008038082702723609
    ,<0.2004064647326758,-2.503983006350664,-0.32378008959630383>,0.008097636716798745
    ,<0.1942964852678104,-2.48925352275312,-0.32502440419162365>,0.008151727381894005
    ,<0.18527709034644282,-2.476091952899027,-0.3261377249448367>,0.008201065543276747
    ,<0.174153516770398,-2.4646547896212163,-0.3272385422731772>,0.008246245102718756
    ,<0.16173596925421016,-2.4546394535115397,-0.3283577079292953>,0.00828776588047385
    ,<0.148735353335803,-2.4453841931697147,-0.32941480571144893>,0.008326051367736582
    ,<0.13578263932336185,-2.4360573020045444,-0.3304200473011867>,0.00836146264109268
    ,<0.12352992501606415,-2.425830577730167,-0.3314474437358428>,0.008394309364827233
    ,<0.11275001495361309,-2.4140737614130874,-0.3326137132903631>,0.008424858562469344
    ,<0.10434179542451809,-2.400540055089191,-0.3340340477419668>,0.00845334166411343
    ,<0.09920432747385462,-2.3854847345144687,-0.3357404953567555>,0.008479960209706025
    ,<0.09800071521568737,-2.369654366125863,-0.3377227410677368>,0.008504890496255251
    ,<0.10091453533007597,-2.3540861554067183,-0.3399863767146037>,0.008528287388947346
    ,<0.10758094764517004,-2.3397603808064127,-0.34249961093536774>,0.008550287465601714
    ,<0.11722668556099067,-2.327285358529905,-0.3452052275203695>,0.008571011625971648
    ,<0.1289280085002996,-2.316750807305823,-0.348049109800315>,0.00859056726871202
    ,<0.14183979285617163,-2.3077684749052043,-0.3509799096527194>,0.008609050116966811
    ,<0.15529176402208797,-2.2996332665891375,-0.3539557918853623>,0.008626545756733304
    ,<0.16875131127032902,-2.2915100290379944,-0.3569353020376975>,0.008643130939168025
    ,<0.1817148114444631,-2.282598376869532,-0.3598636520120867>,0.00865887468788217
    ,<0.19361222968100383,-2.2722752639940396,-0.36267677913610724>,0.008673839244344611
    ,<0.20379229913940125,-2.2602179762774606,-0.3653217253479188>,0.008688080878257348
    ,<0.21160624763764949,-2.2464728210858627,-0.3677704530047297>,0.008701650584808223
    ,<0.2165546961404457,-2.231425463860832,-0.37001867313695863>,0.008714594686749191
    ,<0.21842298731972526,-2.2156719826785483,-0.3720871863559933>,0.008726955356075762
    ,<0.21733991202637043,-2.19982716704866,-0.37400802137324357>,0.008738771067525925
    ,<0.21374015193001344,-2.1843438229041596,-0.375803836792452>,0.008750076994045604
    ,<0.20826021014625618,-2.1694083697626865,-0.37748257840114274>,0.008760905352682195
    ,<0.20163300405410772,-2.154933327521622,-0.37905218872479457>,0.008771285707989934
    ,<0.19462572602862693,-2.1406300910766167,-0.38054510302315386>,0.008781245238899917
    ,<0.18803227999549743,-2.1261285559386813,-0.382024083194896>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
