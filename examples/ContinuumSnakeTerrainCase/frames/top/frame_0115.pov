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
    ,<0.221208078759902,-2.3930197291632376,-0.3321446993473791>,0.0
    ,<0.23509081369330573,-2.387386205623694,-0.33775910753925537>,0.001444405933878283
    ,<0.2493695398191813,-2.382784758675926,-0.34332897558053194>,0.002733688514425582
    ,<0.2638404566051465,-2.3786379945478635,-0.34876556069545506>,0.0037941133653625076
    ,<0.27833901327204497,-2.3742689700231097,-0.35394539952318493>,0.0046307451971068355
    ,<0.292651686910583,-2.3689855239459763,-0.35877393598161916>,0.005283185474353696
    ,<0.30643910194512425,-2.3621514629641553,-0.3631646109893993>,0.005794598874521764
    ,<0.31919521329674544,-2.3532885694515033,-0.36702596022480133>,0.00620058003411749
    ,<0.3302609484696237,-2.3421795756270543,-0.37024743237343044>,0.006527801879788091
    ,<0.3389189836602945,-2.328942749827791,-0.37270477937433893>,0.006795619711330263
    ,<0.3445312075970096,-2.3140402732340886,-0.3743159749826862>,0.007018006566011825
    ,<0.3466839146776559,-2.298199142219076,-0.3750621473184497>,0.007205119848667835
    ,<0.34529246365629246,-2.282256981544457,-0.37499621709857617>,0.007364433711532417
    ,<0.34061031285658555,-2.2669724149122907,-0.3742445724177345>,0.0075015263935279105
    ,<0.333148174468438,-2.2528669345723715,-0.37300045025253>,0.007620622272343326
    ,<0.3235684099284638,-2.240133025091251,-0.371492606088711>,0.007724966207910139
    ,<0.3125473404801409,-2.2286263748707538,-0.36999001266014786>,0.007817084460335388
    ,<0.3007135891820187,-2.2179279100338496,-0.3687243610477368>,0.007898968749670325
    ,<0.28863823769760966,-2.2074599055831547,-0.3678523419264>,0.007972207813666372
    ,<0.2768685847678476,-2.196614301350818,-0.3674824951561468>,0.008038082702723609
    ,<0.26600953033160407,-2.184850361724065,-0.3675893732890842>,0.008097636716798745
    ,<0.2567304914317389,-2.171813721841909,-0.36805660271413015>,0.008151727381894005
    ,<0.24971199983443879,-2.1574421641129407,-0.36875832283196197>,0.008201065543276747
    ,<0.2455442043437792,-2.1420066737529626,-0.3695662160918701>,0.008246245102718756
    ,<0.2445850852467811,-2.1260497274990553,-0.3703536576829758>,0.00828776588047385
    ,<0.2468525884410979,-2.1102233898896285,-0.37100638564011135>,0.008326051367736582
    ,<0.2520159383766982,-2.0950870464525737,-0.3714380244797336>,0.00836146264109268
    ,<0.2594873518066336,-2.0809439621229298,-0.37160944413296393>,0.008394309364827233
    ,<0.2685682645438246,-2.0677776811941313,-0.371549408218391>,0.008424858562469344
    ,<0.2785524471613335,-2.0552849431065843,-0.3713569550893457>,0.00845334166411343
    ,<0.28878040818471457,-2.0430026770282352,-0.3712090369975293>,0.008479960209706025
    ,<0.2984428733536455,-2.030280452434968,-0.3712636768652634>,0.008504890496255251
    ,<0.3065411523042082,-2.0164987427266103,-0.37165823536435594>,0.008528287388947346
    ,<0.3121386319254064,-2.0015231553297785,-0.3723633734465561>,0.008550287465601714
    ,<0.3144625425239868,-1.9857139846531873,-0.37318867298672415>,0.008571011625971648
    ,<0.3130517218339677,-1.9698011365693564,-0.3740365379709818>,0.00859056726871202
    ,<0.3079125006431452,-1.9546754939794717,-0.37485218332453574>,0.008609050116966811
    ,<0.299517366380937,-1.9410827887558812,-0.37560723383719963>,0.008626545756733304
    ,<0.28863575848471495,-1.9293823630798757,-0.37628927692618>,0.008643130939168025
    ,<0.27610110673803645,-1.9194698662062495,-0.3768968029540724>,0.00865887468788217
    ,<0.26264022238930407,-1.9108537787862427,-0.3774382904389428>,0.008673839244344611
    ,<0.24882838465007812,-1.9028107466285107,-0.37793397770650555>,0.008688080878257348
    ,<0.23514703039905696,-1.8945483230205413,-0.3784194475055069>,0.008701650584808223
    ,<0.222078983032359,-1.8853506629002923,-0.37895028553684423>,0.008714594686749191
    ,<0.21017351142340138,-1.8746965414671874,-0.3796088294094801>,0.008726955356075762
    ,<0.20004266171056534,-1.8623510771179541,-0.38048985674932706>,0.008738771067525925
    ,<0.19226802436725188,-1.84841216723988,-0.381587675402303>,0.008750076994045604
    ,<0.18728086519443896,-1.8332584961469018,-0.38281905373548936>,0.008760905352682195
    ,<0.18526130234402188,-1.8174389085363727,-0.38412697276575014>,0.008771285707989934
    ,<0.1860878858758337,-1.8015161549357932,-0.38546779765975747>,0.008781245238899917
    ,<0.1893604911009105,-1.7859123476791565,-0.38681391279478194>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
