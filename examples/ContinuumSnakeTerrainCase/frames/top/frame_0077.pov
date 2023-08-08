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
    ,<0.11321569143404428,-2.6381959853009316,-0.3099533673357409>,0.0
    ,<0.12513321792789556,-2.6275741257807663,-0.3110209634679395>,0.001444405933878283
    ,<0.1373568982365609,-2.617309395432944,-0.31212615344713873>,0.002733688514425582
    ,<0.14948335097630636,-2.6069403938296425,-0.31332925830332775>,0.0037941133653625076
    ,<0.16109499153293597,-2.596013008788752,-0.31466653057086236>,0.0046307451971068355
    ,<0.17173975777779552,-2.5841589500466617,-0.3161534630284552>,0.005283185474353696
    ,<0.18091605885163878,-2.571160277389806,-0.31780985797302774>,0.005794598874521764
    ,<0.18805185671063804,-2.556957533173422,-0.31957797219130135>,0.00620058003411749
    ,<0.19258516520283248,-2.5417242078955504,-0.3213431995882342>,0.006527801879788091
    ,<0.19412539288007485,-2.525898314414118,-0.32302844585457213>,0.006795619711330263
    ,<0.1925510342856781,-2.510063796735367,-0.3245845590456668>,0.007018006566011825
    ,<0.1880452074153477,-2.4947885749005527,-0.32599030233277615>,0.007205119848667835
    ,<0.18105017678431387,-2.4804681957287507,-0.3272536005833102>,0.007364433711532417
    ,<0.17216576976938358,-2.467225215280494,-0.32841379075058635>,0.0075015263935279105
    ,<0.16204705863305469,-2.454893015550787,-0.32954299270088044>,0.007620622272343326
    ,<0.15133926892431238,-2.443069033719657,-0.3307002222763057>,0.007724966207910139
    ,<0.14066417817864074,-2.4312195727489065,-0.3319099131737093>,0.007817084460335388
    ,<0.13066363400771291,-2.4188031326505732,-0.33321342360124334>,0.007898968749670325
    ,<0.12205278756906701,-2.405399566744659,-0.3346749711139332>,0.007972207813666372
    ,<0.11562311327085906,-2.3908445294599003,-0.33634082752598127>,0.008038082702723609
    ,<0.1121637446548635,-2.375338321646107,-0.33822984327429884>,0.008097636716798745
    ,<0.11230409491361261,-2.359478623157099,-0.3403388158955198>,0.008151727381894005
    ,<0.11633086616038657,-2.3441647570077246,-0.3426341684514494>,0.008201065543276747
    ,<0.12406723460513858,-2.330375704134854,-0.34508090324194657>,0.008246245102718756
    ,<0.13489774878062905,-2.3188875915721487,-0.3476664763077197>,0.00828776588047385
    ,<0.1479505433200703,-2.310045531958603,-0.35038529099244015>,0.008326051367736582
    ,<0.16235302523272108,-2.303676929558271,-0.35320902183694897>,0.00836146264109268
    ,<0.177430971872935,-2.299168946162248,-0.35609845044609956>,0.008394309364827233
    ,<0.1927605333838844,-2.2956204401416938,-0.35900743301723803>,0.008424858562469344
    ,<0.20807999057416787,-2.2919899104941974,-0.36186651774456663>,0.00845334166411343
    ,<0.22310935450997207,-2.287235407733871,-0.3646077351440196>,0.008479960209706025
    ,<0.23737330526360056,-2.2804626523297347,-0.36718163939331944>,0.008504890496255251
    ,<0.25012477471358274,-2.271100746331693,-0.36956237662227054>,0.008528287388947346
    ,<0.2604351828964944,-2.2590653546498682,-0.3717448373782386>,0.008550287465601714
    ,<0.2674406896984787,-2.244822275822433,-0.37374517741635427>,0.008571011625971648
    ,<0.27064661070727913,-2.229268545330228,-0.37563325480881826>,0.00859056726871202
    ,<0.2700813398894348,-2.2133992134451748,-0.3774926051317609>,0.008609050116966811
    ,<0.26625854165742935,-2.197979212470269,-0.3793449024233598>,0.008626545756733304
    ,<0.2599761851428571,-2.183374224904596,-0.3811343310435778>,0.008643130939168025
    ,<0.2521121995342892,-2.1695396925036765,-0.38277995526431396>,0.00865887468788217
    ,<0.2435170309659043,-2.1561265907539497,-0.3842398305401768>,0.008673839244344611
    ,<0.23498080846120392,-2.142656720558446,-0.3855074275066045>,0.008688080878257348
    ,<0.22725063725038003,-2.128694499207742,-0.38660668648185165>,0.008701650584808223
    ,<0.22103623749599874,-2.1139866338885835,-0.3875990509606169>,0.008714594686749191
    ,<0.2169739829023645,-2.098541556002986,-0.38855911694096557>,0.008726955356075762
    ,<0.21555663015293514,-2.0826326339665746,-0.3895072760798099>,0.008738771067525925
    ,<0.2170438260415323,-2.0667285843046947,-0.39042712586655726>,0.008750076994045604
    ,<0.2214017126758845,-2.0513595232256248,-0.3913191702867915>,0.008760905352682195
    ,<0.2283190705000212,-2.0369594020081867,-0.3922036707285841>,0.008771285707989934
    ,<0.23728879237977935,-2.0237405311397736,-0.39310631694976966>,0.008781245238899917
    ,<0.24771449910946244,-2.0116378712463203,-0.3940212155701199>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
