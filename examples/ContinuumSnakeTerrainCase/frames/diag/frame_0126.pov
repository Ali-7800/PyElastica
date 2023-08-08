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
    ,<0.2741729247577531,-2.3340450828964157,-0.36963785373453373>,0.0
    ,<0.2866641951397316,-2.3240502836545316,-0.3694542072831632>,0.001444405933878283
    ,<0.29695493890950575,-2.3118032859373305,-0.36921179148529676>,0.002733688514425582
    ,<0.30440419657184375,-2.297649794053246,-0.36884841241009775>,0.0037941133653625076
    ,<0.3085791887588761,-2.2822138102644653,-0.3683495414583424>,0.0046307451971068355
    ,<0.3093643593900225,-2.2662462888580195,-0.36774554266704845>,0.005283185474353696
    ,<0.3069815270820653,-2.2504397123334816,-0.3671063478718387>,0.005794598874521764
    ,<0.3019185370355116,-2.235275395706845,-0.3665377346731546>,0.00620058003411749
    ,<0.29479631131606343,-2.2209558721851983,-0.36614758630054844>,0.006527801879788091
    ,<0.2862866452942929,-2.2074045960368385,-0.36601317930573907>,0.006795619711330263
    ,<0.2770874579647183,-2.19430708445804,-0.3661431437023708>,0.007018006566011825
    ,<0.2678764936044102,-2.181221209144146,-0.3664782339476658>,0.007205119848667835
    ,<0.25931369650475766,-2.167707049184457,-0.366942707150124>,0.007364433711532417
    ,<0.25207197470321674,-2.1534445197498178,-0.3674544750066241>,0.0075015263935279105
    ,<0.24682704825212598,-2.1383341945011853,-0.3679345649701392>,0.007620622272343326
    ,<0.2441957766965369,-2.12255724073805,-0.3683153242372473>,0.007724966207910139
    ,<0.24463242792897105,-2.1065681856694343,-0.3685476385411917>,0.007817084460335388
    ,<0.24832228704422574,-2.091005913013973,-0.3686090118498431>,0.007898968749670325
    ,<0.25512552041477504,-2.0765337450450283,-0.36851192018600015>,0.007972207813666372
    ,<0.2646083561864722,-2.063660597219522,-0.3683110312927003>,0.008038082702723609
    ,<0.2761526424080602,-2.052599146427125,-0.3681054099431466>,0.008097636716798745
    ,<0.2891047890106419,-2.043226079913444,-0.36804559415115856>,0.008151727381894005
    ,<0.3028195880066796,-2.0350133397620156,-0.36826962336748115>,0.008201065543276747
    ,<0.3166722631240591,-2.0270465954182266,-0.36888252874037075>,0.008246245102718756
    ,<0.3300227854216973,-2.018295606203122,-0.3699493881097611>,0.00828776588047385
    ,<0.34210053286703845,-2.007904093097029,-0.3714079460857373>,0.008326051367736582
    ,<0.35191804188590303,-1.9953964841589225,-0.37314040817982397>,0.00836146264109268
    ,<0.3583224453329123,-1.9808714956898947,-0.3750504657067822>,0.008394309364827233
    ,<0.3602857663812517,-1.9651338348333987,-0.37703478046900835>,0.008424858562469344
    ,<0.3573003650819489,-1.9495567168909078,-0.3789774905631096>,0.00845334166411343
    ,<0.34963797827070786,-1.9356503076472622,-0.3807590374531105>,0.008479960209706025
    ,<0.3382677319974131,-1.9245276363792663,-0.3822747606020724>,0.008504890496255251
    ,<0.32446568183453284,-1.916560308255483,-0.38345324901468003>,0.008528287388947346
    ,<0.3093712106717279,-1.911369683245965,-0.3842663214835483>,0.008550287465601714
    ,<0.29373969985838994,-1.9080583685130312,-0.3847304000196499>,0.008571011625971648
    ,<0.2779622399423865,-1.9054883768709148,-0.3849056196654746>,0.00859056726871202
    ,<0.2622596800751505,-1.9024913408337678,-0.384899104654075>,0.008609050116966811
    ,<0.24691410525763724,-1.8980185917490981,-0.38487313252327043>,0.008626545756733304
    ,<0.23242178398232205,-1.8912853492009905,-0.3850524285789488>,0.008643130939168025
    ,<0.21949340141032517,-1.8819193202939388,-0.3857248959405097>,0.00865887468788217
    ,<0.2088956831163395,-1.870050576742188,-0.38719351889922693>,0.008673839244344611
    ,<0.20120888142025967,-1.8562552719415748,-0.3897228414702608>,0.008688080878257348
    ,<0.19659140459671792,-1.8413381810639315,-0.3932470013423252>,0.008701650584808223
    ,<0.19477733740219313,-1.826000701166318,-0.3974572771242837>,0.008714594686749191
    ,<0.19520277658519897,-1.810692605088891,-0.4021183662418827>,0.008726955356075762
    ,<0.19714161772328578,-1.7955865164720233,-0.4070442353169488>,0.008738771067525925
    ,<0.19980399666785303,-1.7806353743070504,-0.4120999217670332>,0.008750076994045604
    ,<0.20238658625481148,-1.7656842347545145,-0.41719268443545354>,0.008760905352682195
    ,<0.20409153720978013,-1.750602667402005,-0.42226514369816726>,0.008771285707989934
    ,<0.20414911495886195,-1.7354129555299,-0.4272947053028635>,0.008781245238899917
    ,<0.2018756396786107,-1.7203837052152426,-0.4322886559944477>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
