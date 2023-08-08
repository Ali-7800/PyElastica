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
    ,<0.2777251210534306,-2.0691361244314397,-0.3493447772973561>,0.0
    ,<0.28180306940490785,-2.0551853366497888,-0.35603256487105317>,0.001444405933878283
    ,<0.2846318132016284,-2.0409551234316714,-0.3627702750327082>,0.002733688514425582
    ,<0.2866001116265778,-2.026598596024727,-0.3695425062327365>,0.0037941133653625076
    ,<0.2881523740696559,-2.0121578240189626,-0.37624539103719856>,0.0046307451971068355
    ,<0.2898005139386342,-1.997645660417998,-0.38276830831389025>,0.005283185474353696
    ,<0.2920833139823462,-1.9830897170569315,-0.3890014034696781>,0.005794598874521764
    ,<0.29551244106805463,-1.9686004432034594,-0.3948577432107216>,0.00620058003411749
    ,<0.30051524688668446,-1.9544069502374741,-0.4002893725704961>,0.006527801879788091
    ,<0.3073741951560056,-1.9408483206078386,-0.4052996752236846>,0.006795619711330263
    ,<0.3161777013935842,-1.928322837950131,-0.40994650844019315>,0.007018006566011825
    ,<0.32680210433387,-1.917200201695349,-0.4143458679434238>,0.007205119848667835
    ,<0.33895940737957936,-1.90769782101056,-0.4185649939763497>,0.007364433711532417
    ,<0.3522735138311663,-1.899806599597705,-0.4226067557809212>,0.0075015263935279105
    ,<0.3663573002334187,-1.8932943806686184,-0.4264944281390463>,0.007620622272343326
    ,<0.3808759118437564,-1.8877239070608225,-0.4302377548771642>,0.007724966207910139
    ,<0.3955593582290121,-1.8824978059816502,-0.43382891584234323>,0.007817084460335388
    ,<0.4101499240015281,-1.8769224581040878,-0.4372555177137007>,0.007898968749670325
    ,<0.42432790992388675,-1.8702750821615122,-0.44049645966464174>,0.007972207813666372
    ,<0.4376167730038082,-1.8618984026165175,-0.4435187511447096>,0.008038082702723609
    ,<0.44931390167929,-1.851336195440964,-0.4462749203307091>,0.008097636716798745
    ,<0.45852695089189965,-1.838485771547125,-0.4487089540326866>,0.008151727381894005
    ,<0.4643248610668249,-1.8237203876719192,-0.4507657448675927>,0.008201065543276747
    ,<0.4659810203333859,-1.8078957538921048,-0.4524049756112627>,0.008246245102718756
    ,<0.46321397195001474,-1.7921901775816522,-0.4536161418451203>,0.00828776588047385
    ,<0.45630611503516705,-1.777786177832509,-0.45441107598723157>,0.008326051367736582
    ,<0.44601596584497333,-1.7655430083560608,-0.4548080465222579>,0.00836146264109268
    ,<0.43333517438168656,-1.755786712967207,-0.45484290301614777>,0.008394309364827233
    ,<0.41920794237323694,-1.748279046518945,-0.4545728633775297>,0.008424858562469344
    ,<0.404354868339408,-1.742347353184708,-0.45405767727331414>,0.00845334166411343
    ,<0.3892619315136965,-1.7370736227340258,-0.45335119744450336>,0.008479960209706025
    ,<0.37429132743173943,-1.7314757207325435,-0.4525182443765104>,0.008504890496255251
    ,<0.3598431136294135,-1.7246504780624112,-0.4516030563008191>,0.008528287388947346
    ,<0.3464682787098496,-1.7159145539536207,-0.4506334993322183>,0.008550287465601714
    ,<0.334866842322671,-1.7049372522031452,-0.4496407970053282>,0.008571011625971648
    ,<0.32575265441845147,-1.6918224979286942,-0.4486705774608065>,0.00859056726871202
    ,<0.3196378897603656,-1.6770641657511476,-0.4478146895690997>,0.008609050116966811
    ,<0.3166685040871736,-1.6613588323235278,-0.4471959653460612>,0.008626545756733304
    ,<0.31657799637524203,-1.6453676869401006,-0.4469161639067465>,0.008643130939168025
    ,<0.3187662067700774,-1.629524664653437,-0.44702266553549613>,0.00865887468788217
    ,<0.32244111540879944,-1.6139660104450568,-0.4475181672638133>,0.008673839244344611
    ,<0.32674537332665277,-1.5985841079045775,-0.44835922828529157>,0.008688080878257348
    ,<0.3308261877198399,-1.5831579548224772,-0.4494756059218824>,0.008701650584808223
    ,<0.33386405297059274,-1.5675069318729054,-0.4507838415238866>,0.008714594686749191
    ,<0.33510992464716244,-1.5516217882136576,-0.45220377104158327>,0.008726955356075762
    ,<0.3339555926900342,-1.535734184433274,-0.4536746650189784>,0.008738771067525925
    ,<0.330020093860861,-1.5202992442601506,-0.4551563375114337>,0.008750076994045604
    ,<0.3232242195538424,-1.5058927896918481,-0.4566401403742055>,0.008760905352682195
    ,<0.3137925798620301,-1.4930571621036812,-0.45813470113156723>,0.008771285707989934
    ,<0.3021844822072546,-1.482152172354352,-0.4596531935601399>,0.008781245238899917
    ,<0.28897187919264594,-1.4732604545216448,-0.4611892267048749>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
