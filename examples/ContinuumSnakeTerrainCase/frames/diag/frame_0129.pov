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
    ,<0.29801192053806136,-2.3057720387598444,-0.37328956523882106>,0.0
    ,<0.30453457266337525,-2.291207046756337,-0.37217970801720673>,0.001444405933878283
    ,<0.3080759141652416,-2.2756448683348527,-0.37107650665890596>,0.002733688514425582
    ,<0.3085346237419561,-2.2596875050083165,-0.37002093233354755>,0.0037941133653625076
    ,<0.30612824378339226,-2.2438982405335457,-0.3691048724014337>,0.0046307451971068355
    ,<0.3012833294130062,-2.228670675886781,-0.3684170083284978>,0.005283185474353696
    ,<0.2945577013621893,-2.214164893258201,-0.36798595064384404>,0.005794598874521764
    ,<0.2865940633792484,-2.2002874877365683,-0.3678163957041843>,0.00620058003411749
    ,<0.27806145557899514,-2.186746738493054,-0.3678460155642646>,0.006527801879788091
    ,<0.2696221273178412,-2.1731478309474337,-0.36796925848058626>,0.006795619711330263
    ,<0.26192471832913045,-2.1591158424772727,-0.36810946137141426>,0.007018006566011825
    ,<0.25562623440141197,-2.1444016437519964,-0.3682047492413782>,0.007205119848667835
    ,<0.25137682301716824,-2.1289701126981413,-0.36820709776735344>,0.007364433711532417
    ,<0.24975669661267083,-2.113048724279105,-0.36808624256368866>,0.0075015263935279105
    ,<0.2511811678134821,-2.097112686980157,-0.36783714344978535>,0.007620622272343326
    ,<0.25580789419219485,-2.0818011544908903,-0.3674890157572107>,0.007724966207910139
    ,<0.26349043807908296,-2.067775114461316,-0.36711329968382383>,0.007817084460335388
    ,<0.2738019645784016,-2.055550480439005,-0.3668236262642257>,0.007898968749670325
    ,<0.28613754245054174,-2.0453718735963977,-0.36677673426520796>,0.007972207813666372
    ,<0.29980841360994764,-2.0370815030071445,-0.36712376865409474>,0.008038082702723609
    ,<0.3141451890277375,-2.030040690238314,-0.36796868834229185>,0.008097636716798745
    ,<0.32857629836896785,-2.0232751401775193,-0.3693503029651286>,0.008151727381894005
    ,<0.3425583552971826,-2.015729153307506,-0.37122590658615945>,0.008201065543276747
    ,<0.35540143960872245,-2.006476332479816,-0.37352387096815576>,0.008246245102718756
    ,<0.3661326545631644,-1.9949214788638014,-0.37616701932350866>,0.00828776588047385
    ,<0.3735429548913526,-1.9810578784822173,-0.3790530617792489>,0.008326051367736582
    ,<0.37646849548618394,-1.9656391851434103,-0.38204173758438864>,0.00836146264109268
    ,<0.37422768416057034,-1.9500966726112,-0.3849531181431225>,0.008394309364827233
    ,<0.3669664318849889,-1.936120240893953,-0.3875865416659815>,0.008424858562469344
    ,<0.35564884197361474,-1.9250641090415797,-0.3897570040452424>,0.00845334166411343
    ,<0.3416649108946053,-1.9175058429770184,-0.39133149363164443>,0.008479960209706025
    ,<0.3263111006366722,-1.9131760306333454,-0.3922473064095749>,0.008504890496255251
    ,<0.3104537112125492,-1.9111923121778704,-0.3925093852263764>,0.008528287388947346
    ,<0.29449262905935447,-1.9103695012132098,-0.3921789963401006>,0.008550287465601714
    ,<0.2785533489113146,-1.9094515610921647,-0.39136730648341>,0.008571011625971648
    ,<0.26276013993740954,-1.907255698968196,-0.39024112046285536>,0.00859056726871202
    ,<0.24745884897684525,-1.9028003723217357,-0.3890370878846349>,0.008609050116966811
    ,<0.23329665257113985,-1.8954655919068772,-0.38807181309807043>,0.008626545756733304
    ,<0.22110648965370935,-1.8851467589478985,-0.3877358485252952>,0.008643130939168025
    ,<0.21165746091985613,-1.8722812094120456,-0.38843417635130206>,0.00865887468788217
    ,<0.20540436576743615,-1.8577014392535578,-0.390463003680896>,0.008673839244344611
    ,<0.2022951380886657,-1.8423401069037213,-0.39371719813619743>,0.008688080878257348
    ,<0.20181530270797576,-1.8268795862961478,-0.39783935250785046>,0.008701650584808223
    ,<0.20322252761719517,-1.8116440336066075,-0.40254523877380816>,0.008714594686749191
    ,<0.2056923343433582,-1.7966615498383391,-0.4076090516545824>,0.008726955356075762
    ,<0.20839774938021324,-1.7817875368847602,-0.4128645025313702>,0.008738771067525925
    ,<0.21053822625399823,-1.7668506117480327,-0.4181977652213729>,0.008750076994045604
    ,<0.21135550803315334,-1.7517884251354463,-0.4235411213999541>,0.008760905352682195
    ,<0.21017470581073408,-1.7367447306860624,-0.42886349826748604>,0.008771285707989934
    ,<0.20648105953460275,-1.7221061863572849,-0.4341593749911603>,0.008781245238899917
    ,<0.20001264963155926,-1.708460532451164,-0.4394433986817007>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
