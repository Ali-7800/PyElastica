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
    ,<-1.382647246011779e-13,-3.2,-0.16332211435702096>,0.0
    ,<2.6253082375089737e-14,-3.184,-0.16332211435702096>,0.001444405933878283
    ,<1.1586571957367551e-13,-3.168,-0.16332211435702096>,0.002733688514425582
    ,<1.8913046487523203e-13,-3.152,-0.16332211435702096>,0.0037941133653625076
    ,<2.4174024065087476e-13,-3.136,-0.16332211435702096>,0.0046307451971068355
    ,<2.712654465749766e-13,-3.12,-0.16332211435702096>,0.005283185474353696
    ,<2.796100658297404e-13,-3.104,-0.16332211435702096>,0.005794598874521764
    ,<2.728133400200789e-13,-3.088,-0.16332211435702096>,0.00620058003411749
    ,<2.4335646990717273e-13,-3.072,-0.16332211435702096>,0.006527801879788091
    ,<1.9235709146189634e-13,-3.056,-0.16332211435702096>,0.006795619711330263
    ,<1.2307764379579757e-13,-3.04,-0.16332211435702096>,0.007018006566011825
    ,<4.003430850454461e-14,-3.024,-0.16332211435702096>,0.007205119848667835
    ,<-5.142887665654581e-14,-3.008,-0.16332211435702096>,0.007364433711532417
    ,<-1.4547371516889116e-13,-2.992,-0.16332211435702096>,0.0075015263935279105
    ,<-2.3606952287164146e-13,-2.9760000000000004,-0.16332211435702096>,0.007620622272343326
    ,<-3.172806073504503e-13,-2.9600000000000004,-0.16332211435702096>,0.007724966207910139
    ,<-3.835560476418548e-13,-2.9440000000000004,-0.16332211435702096>,0.007817084460335388
    ,<-4.3006006049374337e-13,-2.9280000000000004,-0.16332211435702096>,0.007898968749670325
    ,<-4.530751148222914e-13,-2.9120000000000004,-0.16332211435702096>,0.007972207813666372
    ,<-4.5341784940200216e-13,-2.8960000000000004,-0.16332211435702096>,0.008038082702723609
    ,<-4.340331651791718e-13,-2.8800000000000003,-0.16332211435702096>,0.008097636716798745
    ,<-3.851099852610476e-13,-2.8640000000000003,-0.16332211435702096>,0.008151727381894005
    ,<-3.066158818255401e-13,-2.8480000000000003,-0.16332211435702096>,0.008201065543276747
    ,<-2.0180162403141562e-13,-2.8320000000000003,-0.16332211435702096>,0.008246245102718756
    ,<-7.65677760794242e-14,-2.8160000000000003,-0.16332211435702096>,0.00828776588047385
    ,<6.095823909016577e-14,-2.8000000000000003,-0.16332211435702096>,0.008326051367736582
    ,<2.009924914636298e-13,-2.7840000000000003,-0.16332211435702096>,0.00836146264109268
    ,<3.3285506338688164e-13,-2.7680000000000002,-0.16332211435702096>,0.008394309364827233
    ,<4.45895768765723e-13,-2.7520000000000002,-0.16332211435702096>,0.008424858562469344
    ,<5.305469401418655e-13,-2.736,-0.16332211435702096>,0.00845334166411343
    ,<5.795536913762304e-13,-2.72,-0.16332211435702096>,0.008479960209706025
    ,<5.907175513935476e-13,-2.704,-0.16332211435702096>,0.008504890496255251
    ,<5.734339219759885e-13,-2.688,-0.16332211435702096>,0.008528287388947346
    ,<5.167156699564605e-13,-2.672,-0.16332211435702096>,0.008550287465601714
    ,<4.24036035471004e-13,-2.656,-0.16332211435702096>,0.008571011625971648
    ,<3.0374323016297777e-13,-2.64,-0.16332211435702096>,0.00859056726871202
    ,<1.668463874839573e-13,-2.6240000000000006,-0.16332211435702096>,0.008609050116966811
    ,<2.5353165868161998e-14,-2.6080000000000005,-0.16332211435702096>,0.008626545756733304
    ,<-1.0897120448985397e-13,-2.5920000000000005,-0.16332211435702096>,0.008643130939168025
    ,<-2.2567442609225814e-13,-2.5760000000000005,-0.16332211435702096>,0.00865887468788217
    ,<-3.1646510102052243e-13,-2.5600000000000005,-0.16332211435702096>,0.008673839244344611
    ,<-3.75819310589382e-13,-2.5440000000000005,-0.16332211435702096>,0.008688080878257348
    ,<-4.0147052856510614e-13,-2.5280000000000005,-0.16332211435702096>,0.008701650584808223
    ,<-3.9633465653750837e-13,-2.5120000000000005,-0.16332211435702096>,0.008714594686749191
    ,<-3.711355659464504e-13,-2.4960000000000004,-0.16332211435702096>,0.008726955356075762
    ,<-3.188098873072863e-13,-2.4800000000000004,-0.16332211435702096>,0.008738771067525925
    ,<-2.425964311805984e-13,-2.4640000000000004,-0.16332211435702096>,0.008750076994045604
    ,<-1.47507366836947e-13,-2.4480000000000004,-0.16332211435702096>,0.008760905352682195
    ,<-3.899750011009617e-14,-2.4320000000000004,-0.16332211435702096>,0.008771285707989934
    ,<7.760708172397049e-14,-2.4160000000000004,-0.16332211435702096>,0.008781245238899917
    ,<1.252909118619519e-13,-2.4000000000000004,-0.16332211435702096>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
