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
    ,<0.11114787108081886,-2.6478317146970434,-0.3103105785369424>,0.0
    ,<0.12176425344994701,-2.6358919923583204,-0.31114684878062127>,0.001444405933878283
    ,<0.13308830218716994,-2.6246248445160343,-0.3120272096591006>,0.002733688514425582
    ,<0.14469271447021206,-2.6136544445683794,-0.3130036704547861>,0.0037941133653625076
    ,<0.15614435076760452,-2.6025362290853358,-0.31411201457134985>,0.0046307451971068355
    ,<0.16699085119619478,-2.5908428930459206,-0.31537743450762146>,0.005283185474353696
    ,<0.17672726154431287,-2.578226255569533,-0.3167859418263949>,0.005794598874521764
    ,<0.18479372229299068,-2.5644943379330423,-0.3182980416169691>,0.00620058003411749
    ,<0.19062385188781025,-2.5496816361781045,-0.31986577803283284>,0.006527801879788091
    ,<0.19373328204404883,-2.5340695734149983,-0.3214280627662856>,0.006795619711330263
    ,<0.19383382555051606,-2.518146282126176,-0.3229338969214275>,0.007018006566011825
    ,<0.19091244142736066,-2.5024847093198925,-0.3243504387938898>,0.007205119848667835
    ,<0.18524196210443872,-2.487587430649414,-0.32567286669106127>,0.007364433711532417
    ,<0.1773182236491467,-2.473749783566398,-0.32693091476310726>,0.0075015263935279105
    ,<0.16775868394864987,-2.460990098205698,-0.3281995542192152>,0.007620622272343326
    ,<0.15720324231121705,-2.449046179058821,-0.32953075715567437>,0.007724966207910139
    ,<0.14626972299123758,-2.4374469321824144,-0.33089511730970295>,0.007817084460335388
    ,<0.1355755500284621,-2.4256278584088284,-0.3322655401639977>,0.007898968749670325
    ,<0.12578942926959363,-2.413051921491467,-0.3336801346772346>,0.007972207813666372
    ,<0.11766662275861743,-2.399354725460455,-0.33522078827665674>,0.008038082702723609
    ,<0.11203041099188628,-2.384478737212553,-0.3369368550578094>,0.008097636716798745
    ,<0.10966598474951027,-2.3687683010812357,-0.3388363464709215>,0.008151727381894005
    ,<0.11112704129984627,-2.352971228084332,-0.3409202587060126>,0.008201065543276747
    ,<0.11654612716631194,-2.3380871026473855,-0.3431813874721577>,0.008246245102718756
    ,<0.1255544407220037,-2.325089139598537,-0.34560975353171364>,0.00828776588047385
    ,<0.1373828313221762,-2.3146283861556225,-0.34818865448074815>,0.008326051367736582
    ,<0.15110056884299503,-2.3068510240388536,-0.3508947223650498>,0.00836146264109268
    ,<0.16587629995962747,-2.301383929617698,-0.3536857884827589>,0.008394309364827233
    ,<0.18112875311132465,-2.2974625381543716,-0.35652111512639095>,0.008424858562469344
    ,<0.19651508675624796,-2.294095132460016,-0.3593525568919827>,0.00845334166411343
    ,<0.21179212682290063,-2.2902081661152858,-0.362107589634903>,0.008479960209706025
    ,<0.22662031174705155,-2.284792578968076,-0.36472553041385924>,0.008504890496255251
    ,<0.24041595706969074,-2.2770641369467186,-0.3671696875723455>,0.008528287388947346
    ,<0.2523430091973606,-2.2666411312845756,-0.3694265311402206>,0.008550287465601714
    ,<0.2614798938683929,-2.2536751858413626,-0.37151220136762564>,0.008571011625971648
    ,<0.2671023562733406,-2.2388279846008547,-0.37347765220102275>,0.00859056726871202
    ,<0.2689325944516578,-2.22305792399162,-0.375415964423319>,0.008609050116966811
    ,<0.2671835231537654,-2.207282739430886,-0.3773715006388067>,0.008626545756733304
    ,<0.2624842690119017,-2.1921168929368786,-0.37929870455327436>,0.008643130939168025
    ,<0.2556717495548535,-2.177763307601384,-0.3811341528328339>,0.00865887468788217
    ,<0.24761968860155545,-2.164047774455833,-0.38282292434364507>,0.008673839244344611
    ,<0.23915436271099863,-2.150562263777741,-0.3843330255438395>,0.008688080878257348
    ,<0.2310473205324463,-2.1368390911801303,-0.3856573711175471>,0.008701650584808223
    ,<0.22403392876882253,-2.1225118646893795,-0.38681698403931686>,0.008714594686749191
    ,<0.2188022452192928,-2.1074336942098637,-0.3878688643272495>,0.008726955356075762
    ,<0.21593894320168408,-2.091727834222514,-0.3888739628671125>,0.008738771067525925
    ,<0.2158524707759845,-2.0757593162839703,-0.38984307908048893>,0.008750076994045604
    ,<0.21869277665965836,-2.060040684013191,-0.39076255042431474>,0.008760905352682195
    ,<0.2243144911440727,-2.045086609340287,-0.3916406066336013>,0.008771285707989934
    ,<0.2323222285832441,-2.031261491531061,-0.39249933213175375>,0.008781245238899917
    ,<0.2421685671705523,-2.0186786997725488,-0.39335230206650107>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
