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
    ,<0.30388660804205003,-2.2004689065436036,-0.3767529761998741>,0.0
    ,<0.29384441432405584,-2.1880168618795106,-0.37708122808192224>,0.001444405933878283
    ,<0.2837078025104303,-2.175638152879218,-0.37730624027436777>,0.002733688514425582
    ,<0.27400993857449774,-2.162909356837805,-0.377348811258182>,0.0037941133653625076
    ,<0.26533556877966086,-2.14946229276639,-0.3771481085748959>,0.0046307451971068355
    ,<0.25832452393244415,-2.1350837851599844,-0.3766630074012285>,0.005283185474353696
    ,<0.25363610976057027,-2.1198017923273746,-0.37587738243036967>,0.005794598874521764
    ,<0.2518674405036905,-2.1039317053202455,-0.3748085457884607>,0.00620058003411749
    ,<0.2534441520514086,-2.088058895324734,-0.3735173602758078>,0.006527801879788091
    ,<0.2585224018649129,-2.0729492993950167,-0.3721180211115105>,0.006795619711330263
    ,<0.266941822792241,-2.0594099865244404,-0.3707850529284724>,0.007018006566011825
    ,<0.2782469097050628,-2.0481448621834093,-0.3697595723365143>,0.007205119848667835
    ,<0.29171543789351134,-2.039546499206084,-0.3692579532873944>,0.007364433711532417
    ,<0.3064925726509604,-2.0334375689440867,-0.36940059235855977>,0.0075015263935279105
    ,<0.3218815778941863,-2.029137592823387,-0.37026996696070075>,0.007620622272343326
    ,<0.3374423540698371,-2.0257270455760965,-0.3718767759634053>,0.007724966207910139
    ,<0.3529001412680915,-2.022223552574058,-0.3741515291823287>,0.007817084460335388
    ,<0.3679696582913155,-2.017648709247186,-0.37704044412115395>,0.007898968749670325
    ,<0.38215558408610384,-2.0111008953904834,-0.3805235795935397>,0.007972207813666372
    ,<0.39461631641107764,-2.0019209395585764,-0.3845864786936166>,0.008038082702723609
    ,<0.4041627795704303,-1.9899360897285823,-0.38917284631300797>,0.008097636716798745
    ,<0.40947553841045237,-1.9756993070523463,-0.3941340609125845>,0.008151727381894005
    ,<0.40954030793832885,-1.9605458470943764,-0.39919990882499073>,0.008201065543276747
    ,<0.4041129235036219,-1.9463099289270014,-0.4039979250545404>,0.008246245102718756
    ,<0.3939145396421971,-1.9347353997642749,-0.40813163600661895>,0.00828776588047385
    ,<0.3803711031296995,-1.926879149245499,-0.41128564907459964>,0.008326051367736582
    ,<0.3650436040751023,-1.9228526055536934,-0.4133002498625261>,0.00836146264109268
    ,<0.3491128139451499,-1.9219758376689329,-0.41418417676026364>,0.008394309364827233
    ,<0.3331702668408417,-1.9231350154881879,-0.41408459557395655>,0.008424858562469344
    ,<0.3173185166211545,-1.925073690799182,-0.41321250533192044>,0.00845334166411343
    ,<0.3014559400173421,-1.9265084494630826,-0.4117075094851963>,0.008479960209706025
    ,<0.28559211388603295,-1.9262218493687193,-0.40966320818753965>,0.008504890496255251
    ,<0.27007630357933676,-1.9231968743322774,-0.4072251061562706>,0.008528287388947346
    ,<0.25566479654987534,-1.9167909856275285,-0.4045903964301841>,0.008550287465601714
    ,<0.24335153099436593,-1.9069218604784324,-0.4020393562161322>,0.008571011625971648
    ,<0.2340387163808165,-1.8941096900372127,-0.39991044079963733>,0.00859056726871202
    ,<0.2282112016142102,-1.8792949887958927,-0.39852850104142845>,0.008609050116966811
    ,<0.22578341651168715,-1.863500239903367,-0.39817655518437356>,0.008626545756733304
    ,<0.22614258343989424,-1.8475292120440279,-0.3989001126177994>,0.008643130939168025
    ,<0.22836521492894546,-1.831775075593322,-0.4005280308344594>,0.00865887468788217
    ,<0.2314586670600757,-1.8162596145978862,-0.4028785030395134>,0.008673839244344611
    ,<0.23447218660349015,-1.8008194498303531,-0.4057703857867253>,0.008688080878257348
    ,<0.23654093731842352,-1.785299638369675,-0.40904227777584434>,0.008701650584808223
    ,<0.2369199686503694,-1.7697007689594453,-0.41256050073476747>,0.008714594686749191
    ,<0.23504268587483587,-1.7542434598622505,-0.41621970045728773>,0.008726955356075762
    ,<0.23059320229366737,-1.7393375117494165,-0.4199402468720982>,0.008738771067525925
    ,<0.22355178788896213,-1.7254680741394588,-0.4236658617795844>,0.008750076994045604
    ,<0.21417735870858642,-1.7130470882195148,-0.4273619670410438>,0.008760905352682195
    ,<0.20292366033631645,-1.702284411821384,-0.43101565856716834>,0.008771285707989934
    ,<0.19032379883076955,-1.6931211263108303,-0.4346374580804492>,0.008781245238899917
    ,<0.17688903723531701,-1.6852294950589106,-0.43826453223321354>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
