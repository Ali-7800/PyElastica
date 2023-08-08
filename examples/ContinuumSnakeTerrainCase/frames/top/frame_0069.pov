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
    ,<0.1163795218080421,-2.727312807952548,-0.3043005264108718>,0.0
    ,<0.11184507970445741,-2.712035444900225,-0.30572457947488496>,0.001444405933878283
    ,<0.11037309670834251,-2.6961643736553245,-0.3071295915201787>,0.002733688514425582
    ,<0.11199909079432097,-2.6803022875070166,-0.3084732422105343>,0.0037941133653625076
    ,<0.11646956532824077,-2.664988032190629,-0.30971224633348476>,0.0046307451971068355
    ,<0.12330906492214974,-2.650566175286286,-0.3108375989745648>,0.005283185474353696
    ,<0.13192414291301038,-2.6371226219603097,-0.31187334821355406>,0.005794598874521764
    ,<0.14169830042735498,-2.6244939667311904,-0.31287828284970226>,0.00620058003411749
    ,<0.15204654970813355,-2.6123339225103646,-0.31391812478712383>,0.006527801879788091
    ,<0.16242576813186554,-2.600209103300694,-0.3150538905757092>,0.006795619711330263
    ,<0.1723185291707647,-2.587701606369781,-0.316357042288469>,0.007018006566011825
    ,<0.18119823372604277,-2.5744826298403236,-0.31786244883856857>,0.007205119848667835
    ,<0.18842888644551228,-2.560315102558299,-0.3195096378564592>,0.007364433711532417
    ,<0.19332518300546012,-2.545187617734846,-0.3211957432997263>,0.0075015263935279105
    ,<0.19533959191382935,-2.5294139968362463,-0.32283932074102656>,0.007620622272343326
    ,<0.194173961499709,-2.513545418693686,-0.32437155112638705>,0.007724966207910139
    ,<0.18986029467683743,-2.498215380388483,-0.32574484178348645>,0.007817084460335388
    ,<0.1827552632756026,-2.483947197728054,-0.3269428529405311>,0.007898968749670325
    ,<0.17344724108099013,-2.470994304159658,-0.32798909243092134>,0.007972207813666372
    ,<0.16262430718052162,-2.459268972409267,-0.32895270288127065>,0.008038082702723609
    ,<0.15096795222833806,-2.448366338334875,-0.329900888048069>,0.008097636716798745
    ,<0.139120119880905,-2.437668617601278,-0.3308510626502341>,0.008151727381894005
    ,<0.12772680326626432,-2.426490056379128,-0.33182643957711955>,0.008201065543276747
    ,<0.11751319949054984,-2.414231709099416,-0.3328937531609645>,0.008246245102718756
    ,<0.10931590923594298,-2.4005570005505534,-0.33414700660660623>,0.00828776588047385
    ,<0.10402191447455256,-2.3855405626797146,-0.33565872095753924>,0.008326051367736582
    ,<0.10239713879256762,-2.3697312247578655,-0.33745734990605797>,0.00836146264109268
    ,<0.10485076526438034,-2.3540655093708547,-0.3395514695992366>,0.008394309364827233
    ,<0.11127176834958495,-2.3396091366702763,-0.3419198536356663>,0.008424858562469344
    ,<0.12104859194195916,-2.327219914400853,-0.3445229062344196>,0.00845334166411343
    ,<0.13327263555719024,-2.3172892006430557,-0.3473214690591632>,0.008479960209706025
    ,<0.14701734868085317,-2.3096605066570293,-0.3502843822018388>,0.008504890496255251
    ,<0.16154434000381634,-2.3037177893778042,-0.3533764661757861>,0.008528287388947346
    ,<0.17635025231106066,-2.2985638652480014,-0.35656474585058606>,0.008550287465601714
    ,<0.19106892772063352,-2.2931957557387506,-0.35980801798957895>,0.008571011625971648
    ,<0.2053101424311523,-2.2866594408966523,-0.36303710006614287>,0.00859056726871202
    ,<0.21852407686990016,-2.2782037374933934,-0.3661704853374723>,0.008609050116966811
    ,<0.2299767885868719,-2.267433857991019,-0.3691251031975193>,0.008626545756733304
    ,<0.23887305066684328,-2.2544195214018647,-0.37182946307486275>,0.008643130939168025
    ,<0.24457780056426245,-2.239673410019257,-0.374234775679513>,0.00865887468788217
    ,<0.2468199502654634,-2.2239780821164383,-0.37632767457793903>,0.008673839244344611
    ,<0.24577013552755259,-2.208123572664662,-0.3781392975471777>,0.008688080878257348
    ,<0.24196287332072428,-2.1926705934375774,-0.37971374446832157>,0.008701650584808223
    ,<0.23614041147412157,-2.1778361888813698,-0.38106777470705244>,0.008714594686749191
    ,<0.2291098749858884,-2.163514500024838,-0.38219507320808505>,0.008726955356075762
    ,<0.2216670327340782,-2.149386385003175,-0.3831005788909933>,0.008738771067525925
    ,<0.21458369432488117,-2.1350642326843534,-0.38382528585184034>,0.008750076994045604
    ,<0.20861879062809008,-2.1202366818628064,-0.38444872109499273>,0.008760905352682195
    ,<0.20451395758433247,-2.1047883049134075,-0.38508072183497827>,0.008771285707989934
    ,<0.2029454183413965,-2.0888803890461376,-0.38575752519598944>,0.008781245238899917
    ,<0.20442940731632417,-2.0729647113746954,-0.38644243430111724>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
