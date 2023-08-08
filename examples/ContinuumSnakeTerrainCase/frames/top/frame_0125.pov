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
    ,<0.26670409670219836,-2.3413992504614605,-0.3681629772276616>,0.0
    ,<0.28031159677541495,-2.33298560669089,-0.3682706706397541>,0.001444405933878283
    ,<0.29210887247094297,-2.3221804360981335,-0.3683303591751855>,0.002733688514425582
    ,<0.3013857299697458,-2.3091486804133137,-0.3682642783612746>,0.0037941133653625076
    ,<0.3075551824489753,-2.2943922131808048,-0.36803088368785625>,0.0046307451971068355
    ,<0.31030229180073315,-2.2786386340134595,-0.36763639076380406>,0.005283185474353696
    ,<0.3096583357390638,-2.2626625571304024,-0.36712916658444406>,0.005794598874521764
    ,<0.30597743911420744,-2.247103805078612,-0.366597751731306>,0.00620058003411749
    ,<0.29982875269588694,-2.2323423573075303,-0.3661451839686484>,0.006527801879788091
    ,<0.2918706284028339,-2.2184656777133607,-0.36586478915483506>,0.006795619711330263
    ,<0.2827818576963442,-2.205292679822997,-0.36582905899820845>,0.007018006566011825
    ,<0.2732496375440435,-2.1924336698358955,-0.36602554449699054>,0.007205119848667835
    ,<0.26394425107391406,-2.179413902236212,-0.3663833121106028>,0.007364433711532417
    ,<0.25552312556310924,-2.1658105595447172,-0.3668428568768135>,0.0075015263935279105
    ,<0.24865833710408927,-2.1513617881373754,-0.3673433930927208>,0.007620622272343326
    ,<0.24401388991518538,-2.1360530353718348,-0.3678222497909242>,0.007724966207910139
    ,<0.24216660097733614,-2.1201596949162473,-0.36821846984836076>,0.007817084460335388
    ,<0.2434913451168897,-2.1042131826559674,-0.3684805630521603>,0.007898968749670325
    ,<0.24806440848031996,-2.088878734962751,-0.36857725076177317>,0.007972207813666372
    ,<0.2556346081999744,-2.074782896085229,-0.3685096127495223>,0.008038082702723609
    ,<0.2656894573642482,-2.0623408408733286,-0.36832302316346593>,0.008097636716798745
    ,<0.27759206658154434,-2.05165892270861,-0.3681202833334937>,0.008151727381894005
    ,<0.29071161947842955,-2.042521335780367,-0.3680597003761097>,0.008201065543276747
    ,<0.30440559495846137,-2.0342759176731784,-0.3682567982036874>,0.008246245102718756
    ,<0.3180253809066895,-2.025914148605832,-0.36879359902250836>,0.00828776588047385
    ,<0.33087673123050787,-2.0164342195291427,-0.36972465790003706>,0.008326051367736582
    ,<0.3420922302550776,-2.0050990073354953,-0.37099144387828575>,0.00836146264109268
    ,<0.3505997984946176,-1.991642133531044,-0.372493259480141>,0.008394309364827233
    ,<0.3552700545593753,-1.9764421527026244,-0.3741370137296246>,0.008424858562469344
    ,<0.35526445309368265,-1.9605483726090338,-0.375821284321545>,0.00845334166411343
    ,<0.3503908677567323,-1.9454149902508955,-0.3774386514290283>,0.008479960209706025
    ,<0.3412262763983962,-1.9324043177484296,-0.37888850228945126>,0.008504890496255251
    ,<0.3288901282185203,-1.9223162686813704,-0.38009387226081753>,0.008528287388947346
    ,<0.31461221578686177,-1.9151911451444346,-0.3810137883157708>,0.008550287465601714
    ,<0.2993682717208466,-1.9104197231155267,-0.3816472847504216>,0.008571011625971648
    ,<0.28375429034945404,-1.9070055810043827,-0.3820324828871968>,0.00859056726871202
    ,<0.2680891677353392,-1.903814141697111,-0.3822466580521253>,0.008609050116966811
    ,<0.2526266812472889,-1.8997515339713562,-0.3824101110742333>,0.008626545756733304
    ,<0.2377495722999325,-1.8939061116802747,-0.3826917673190331>,0.008643130939168025
    ,<0.2240518797421863,-1.8856898347738866,-0.38331014425504983>,0.00865887468788217
    ,<0.21226847706495222,-1.8749563445260278,-0.38453015724509354>,0.008673839244344611
    ,<0.2030802645025452,-1.8620285719691256,-0.38660503437033655>,0.008688080878257348
    ,<0.19686498659172239,-1.8475735983188988,-0.38953017335371>,0.008701650584808223
    ,<0.19359183676642605,-1.8323220527468722,-0.3931102613424704>,0.008714594686749191
    ,<0.1928775289347403,-1.8168603063605282,-0.3971796927673103>,0.008726955356075762
    ,<0.19409507305904355,-1.8015237372001616,-0.4015877526067615>,0.008738771067525925
    ,<0.19649017990385886,-1.7863901820690673,-0.4062077035351861>,0.008750076994045604
    ,<0.1992611316835413,-1.7713573288886935,-0.41094281426710755>,0.008760905352682195
    ,<0.20159498957969801,-1.7562670078365743,-0.4157275161743864>,0.008771285707989934
    ,<0.20268550216225667,-1.74104090279104,-0.4205246965738065>,0.008781245238899917
    ,<0.20176850205148575,-1.7258038226618857,-0.42531934210028016>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
