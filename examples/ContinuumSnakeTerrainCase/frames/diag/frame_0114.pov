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
    ,<0.2212188271878807,-2.3983144113000097,-0.32752261266723914>,0.0
    ,<0.2343764776673433,-2.3915377886447406,-0.3335981583685589>,0.001444405933878283
    ,<0.24817854157855762,-2.386125113031243,-0.33961342002833456>,0.002733688514425582
    ,<0.26235569461183134,-2.381589148312597,-0.3454845084756451>,0.0037941133653625076
    ,<0.2767094203171447,-2.3772923958078733,-0.3511087245050422>,0.0046307451971068355
    ,<0.29104340216564745,-2.3725385506270915,-0.35641119926803255>,0.005283185474353696
    ,<0.3050862255885448,-2.3666383021921247,-0.3613222235974606>,0.005794598874521764
    ,<0.3184255025786645,-2.358993737439496,-0.36577097418288784>,0.00620058003411749
    ,<0.33048549119789894,-2.349208689931141,-0.3696470865013356>,0.006527801879788091
    ,<0.34057946440272285,-2.337194303946391,-0.3728022222896896>,0.006795619711330263
    ,<0.3480025142893852,-2.3232045878910266,-0.37511284194545774>,0.007018006566011825
    ,<0.3521779802019083,-2.3078179446975216,-0.37650343453010165>,0.007205119848667835
    ,<0.352796387861741,-2.2918338205003774,-0.37696792407108054>,0.007364433711532417
    ,<0.349889503350596,-2.276102339072937,-0.37658162883059526>,0.0075015263935279105
    ,<0.34380811937041544,-2.261339174224867,-0.3755039214303795>,0.007620622272343326
    ,<0.33513980517783853,-2.2479768637890647,-0.37395491342505005>,0.007724966207910139
    ,<0.32458178493458495,-2.236081682708279,-0.3721749631837574>,0.007817084460335388
    ,<0.31279985879635785,-2.2253906438888524,-0.37043910817109904>,0.007898968749670325
    ,<0.3003935222686131,-2.2153857518490785,-0.36899874038937486>,0.007972207813666372
    ,<0.28790529917801755,-2.2054206879298657,-0.3680247963798153>,0.008038082702723609
    ,<0.2758809524404949,-2.1948570642342022,-0.3675937664601679>,0.008097636716798745
    ,<0.2649400274009717,-2.183161000148787,-0.36766450556421737>,0.008151727381894005
    ,<0.25577179026295316,-2.1700323085670905,-0.36813627338585414>,0.008201065543276747
    ,<0.24906625035257704,-2.155504172435043,-0.3688793888900372>,0.008246245102718756
    ,<0.24538006927529407,-2.1399436560987715,-0.3697451533359817>,0.00828776588047385
    ,<0.24499254690840613,-2.1239582863058266,-0.3705833367884018>,0.008326051367736582
    ,<0.2478169269244103,-2.108216138526126,-0.3712619084922592>,0.00836146264109268
    ,<0.2534220836927932,-2.0932316668721995,-0.3716889682005828>,0.008394309364827233
    ,<0.26115073295876295,-2.079221398516,-0.37183323639740207>,0.008424858562469344
    ,<0.2702706323879398,-2.0660756650179497,-0.3717418936080857>,0.00845334166411343
    ,<0.2800516325757384,-2.053415976706778,-0.3715345240323662>,0.008479960209706025
    ,<0.28979603885336014,-2.040738650463582,-0.3714071434641466>,0.008504890496255251
    ,<0.29865280977631087,-2.027433943169743,-0.3715290801390276>,0.008528287388947346
    ,<0.30562545282841735,-2.013051048010447,-0.3719634248523819>,0.008550287465601714
    ,<0.309824739450649,-1.9976372568377718,-0.3726811151324719>,0.008571011625971648
    ,<0.3105975856220366,-1.98169305957945,-0.37357815860606175>,0.00859056726871202
    ,<0.30768734356686,-1.9660039331386925,-0.37455852084947155>,0.008609050116966811
    ,<0.3013047604642383,-1.9513796416087845,-0.37554381691780375>,0.008626545756733304
    ,<0.2920449932310623,-1.9383828977210653,-0.37648012185529384>,0.008643130939168025
    ,<0.2806908852905668,-1.927164454662448,-0.3773347951698544>,0.00865887468788217
    ,<0.26801558008094273,-1.9174577490941365,-0.37809104787940323>,0.008673839244344611
    ,<0.25467257499818086,-1.9086873847934942,-0.3787460022293855>,0.008688080878257348
    ,<0.241195798206925,-1.900121342320883,-0.37931245159923005>,0.008701650584808223
    ,<0.22807317800195284,-1.8910224066086654,-0.37982475558821643>,0.008714594686749191
    ,<0.2158211684010448,-1.8807815709274178,-0.3803508379478899>,0.008726955356075762
    ,<0.20501415785914784,-1.8690232625841434,-0.3810086538009444>,0.008738771067525925
    ,<0.1962646814291255,-1.8556636524366297,-0.38191282366911866>,0.008750076994045604
    ,<0.1901162910633425,-1.8409313898934527,-0.38301131935302496>,0.008760905352682195
    ,<0.18691910785745633,-1.8252960334337185,-0.3841671059492384>,0.008771285707989934
    ,<0.18673291345195855,-1.8093378054584182,-0.3853113059802753>,0.008781245238899917
    ,<0.18931161236084776,-1.7935862759914412,-0.3864231576518805>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
