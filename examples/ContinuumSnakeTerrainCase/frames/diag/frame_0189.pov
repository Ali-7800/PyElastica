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
    ,<0.2939331945793556,-2.045160500014117,-0.35869522492574474>,0.0
    ,<0.2994802506420267,-2.0315367183283968,-0.3649851456194132>,0.001444405933878283
    ,<0.3074484454265671,-2.0191632898253102,-0.3712729058658618>,0.002733688514425582
    ,<0.3174558805257564,-2.008327919886988,-0.37750048717869567>,0.0037941133653625076
    ,<0.3290230526915734,-1.999061654606472,-0.38355555035502376>,0.0046307451971068355
    ,<0.34167181895657395,-1.9911556418517538,-0.38937052826730156>,0.005283185474353696
    ,<0.3549825949738787,-1.9841978879040671,-0.3949113381734636>,0.005794598874521764
    ,<0.36860225249262923,-1.9776360057247626,-0.40017531489056785>,0.00620058003411749
    ,<0.38220645941600667,-1.9708503730075246,-0.4051876320907171>,0.006527801879788091
    ,<0.39543757803478957,-1.963231747643117,-0.4099960250397509>,0.006795619711330263
    ,<0.40784822292886824,-1.9542655941601101,-0.41466308144232705>,0.007018006566011825
    ,<0.4188812429620233,-1.9436226668035566,-0.4192628908141383>,0.007205119848667835
    ,<0.4279085628955446,-1.9312139839058533,-0.4238085376866022>,0.007364433711532417
    ,<0.4343275191081748,-1.917236840742054,-0.42823145414297514>,0.0075015263935279105
    ,<0.43767526782649396,-1.9021606068705337,-0.43242608266249477>,0.007620622272343326
    ,<0.43774326665357677,-1.8866272541273335,-0.43626688241458744>,0.007724966207910139
    ,<0.4346398513121897,-1.871298672082566,-0.4396432890417898>,0.007817084460335388
    ,<0.42876613886623705,-1.8566883889661392,-0.442471777444041>,0.007898968749670325
    ,<0.4207237421368493,-1.8430409698405004,-0.44470902897764336>,0.007972207813666372
    ,<0.4112010591389635,-1.8302929653656055,-0.44636158767701545>,0.008038082702723609
    ,<0.400888245745351,-1.8181155117361367,-0.4474923906228499>,0.008097636716798745
    ,<0.39044908178557436,-1.8060147044938475,-0.44821860093187427>,0.008151727381894005
    ,<0.3805406987133636,-1.793462225033676,-0.44870821163935326>,0.008201065543276747
    ,<0.37186577852853153,-1.7800228582329913,-0.4491130030358009>,0.008246245102718756
    ,<0.3651808424729371,-1.7654908286153685,-0.4495367476248471>,0.00828776588047385
    ,<0.36119812830623377,-1.7500022841449125,-0.450095356519114>,0.008326051367736582
    ,<0.36043737538970394,-1.7340364383033295,-0.4508789109462247>,0.00836146264109268
    ,<0.3630803563499382,-1.7182856810094105,-0.451893927981238>,0.008394309364827233
    ,<0.3689002847322909,-1.7034289828210747,-0.4531145518684709>,0.008424858562469344
    ,<0.37732791426552664,-1.6898990065171546,-0.4545231188267054>,0.00845334166411343
    ,<0.3876125055172834,-1.677741215796943,-0.45609658499879596>,0.008479960209706025
    ,<0.39898162820220295,-1.666612048382375,-0.4578117387967724>,0.008504890496255251
    ,<0.4107156876140414,-1.6558870439179614,-0.45964521221040194>,0.008528287388947346
    ,<0.4221216794824183,-1.6448235753368514,-0.4615438109751514>,0.008550287465601714
    ,<0.4324564528091833,-1.6327497055909665,-0.46341963471986475>,0.008571011625971648
    ,<0.4408877581343766,-1.6192621434661363,-0.4651787465250945>,0.00859056726871202
    ,<0.44655867433649055,-1.6043787606035496,-0.4667320282399753>,0.008609050116966811
    ,<0.44874944141770057,-1.5885768400361775,-0.46798355971436806>,0.008626545756733304
    ,<0.4470749594223648,-1.5726857647304342,-0.46882202995808286>,0.008643130939168025
    ,<0.4416208555485322,-1.5576478363835526,-0.46915800151459774>,0.00865887468788217
    ,<0.4329150365837645,-1.5442260901934444,-0.46896052168154834>,0.008673839244344611
    ,<0.4217463488598196,-1.5327927449496415,-0.46825383236390994>,0.008688080878257348
    ,<0.4089384059623228,-1.5232739279913996,-0.46711033452573486>,0.008701650584808223
    ,<0.39518828064738,-1.5152279724435642,-0.4656396736531816>,0.008714594686749191
    ,<0.38101873422129795,-1.5079857433507007,-0.4639803930468509>,0.008726955356075762
    ,<0.3668284499441462,-1.5007883835904297,-0.4622942168759923>,0.008738771067525925
    ,<0.35301017971821025,-1.4928742994726414,-0.460724821622567>,0.008750076994045604
    ,<0.3400562071937091,-1.483585505025812,-0.45933606652284015>,0.008760905352682195
    ,<0.3285607214090906,-1.472527154036488,-0.45809383002776904>,0.008771285707989934
    ,<0.3191788875795438,-1.45962068088882,-0.45692239989237593>,0.008781245238899917
    ,<0.3125226297228994,-1.4451180558846024,-0.455765101207301>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
