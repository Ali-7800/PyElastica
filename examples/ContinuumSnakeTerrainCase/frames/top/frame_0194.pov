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
    ,<0.3122713015134267,-2.032980403247953,-0.360077833059795>,0.0
    ,<0.3243946134388161,-2.0248324561029145,-0.3666150575842233>,0.001444405933878283
    ,<0.3371327877659416,-2.0176645677873237,-0.3731515874368012>,0.002733688514425582
    ,<0.35018776377144645,-2.011026860451755,-0.3796319690701322>,0.0037941133653625076
    ,<0.36332493238539093,-2.004408763330174,-0.38596164071156075>,0.0046307451971068355
    ,<0.37628671711616973,-1.997268915068937,-0.39207942717926425>,0.005283185474353696
    ,<0.3887476163753662,-1.9891045684252626,-0.39794902793320164>,0.005794598874521764
    ,<0.4002845571710538,-1.979523900920426,-0.4035583854052874>,0.00620058003411749
    ,<0.4103858913432331,-1.968317824978086,-0.4089175246082638>,0.006527801879788091
    ,<0.41850775058510536,-1.955516093581829,-0.4140588713058671>,0.006795619711330263
    ,<0.4241594357042996,-1.9413720171664133,-0.41898049639617974>,0.007018006566011825
    ,<0.4270111069316246,-1.9263222652685414,-0.42362439620898507>,0.007205119848667835
    ,<0.4269762215473917,-1.910904263207013,-0.42792065224545384>,0.007364433711532417
    ,<0.4242291367431966,-1.895620687456843,-0.43179267558436984>,0.0075015263935279105
    ,<0.41916653887274674,-1.8808222989292174,-0.43517787046061746>,0.007620622272343326
    ,<0.41233496867715747,-1.8666398721799393,-0.4380473322335071>,0.007724966207910139
    ,<0.40435263596809967,-1.8529752645065392,-0.44040983096668346>,0.007817084460335388
    ,<0.3958585971360641,-1.8395503689255122,-0.4423147874899688>,0.007898968749670325
    ,<0.3874980228108535,-1.8259955165942723,-0.44385439568707596>,0.007972207813666372
    ,<0.37992696744476184,-1.811960969615062,-0.4451706245337945>,0.008038082702723609
    ,<0.3738312984386473,-1.7972169499582822,-0.44640355078493565>,0.008097636716798745
    ,<0.36990727867244977,-1.7817504233746757,-0.44763288010775176>,0.008151727381894005
    ,<0.3687679223795305,-1.765833569078977,-0.44885619182927844>,0.008201065543276747
    ,<0.37080481217087097,-1.7500046741090807,-0.45004108186123737>,0.008246245102718756
    ,<0.3760631597185111,-1.7349341262064093,-0.451181331244192>,0.00828776588047385
    ,<0.38421961538138816,-1.7212116111273645,-0.45227720950058026>,0.008326051367736582
    ,<0.39467563537202455,-1.7091470912905338,-0.4533395336092109>,0.00836146264109268
    ,<0.4067183884542763,-1.6986659721600306,-0.45439050391920427>,0.008394309364827233
    ,<0.41966291663488015,-1.6893247700197742,-0.45546500474207957>,0.008424858562469344
    ,<0.4329036182363582,-1.6804171044282723,-0.45660985116456976>,0.00845334166411343
    ,<0.44586523375793624,-1.6711214836052941,-0.45786737956702933>,0.008479960209706025
    ,<0.45789977907954615,-1.6606691163546625,-0.459257421205837>,0.008504890496255251
    ,<0.4682146997272011,-1.6485320738012763,-0.4607820448877998>,0.008528287388947346
    ,<0.47590866830980183,-1.6345979592186353,-0.4624189848612844>,0.008550287465601714
    ,<0.48014047136308124,-1.619259779160852,-0.4641124495885762>,0.008571011625971648
    ,<0.48036811889053316,-1.6033474374336232,-0.46577432519414197>,0.00859056726871202
    ,<0.47652118505913976,-1.5878900154892477,-0.46727003904445263>,0.008609050116966811
    ,<0.4690275148647869,-1.5738055881384796,-0.4684485383060939>,0.008626545756733304
    ,<0.45867298963483016,-1.5616357070911073,-0.46919644257356957>,0.008643130939168025
    ,<0.4463562392007408,-1.551433049692568,-0.4694564819972892>,0.00865887468788217
    ,<0.43288042472333255,-1.5428184708485921,-0.46922408707259133>,0.008673839244344611
    ,<0.4188681427663784,-1.5351335499179208,-0.46853897179900267>,0.008688080878257348
    ,<0.4047928376302558,-1.5276072490608108,-0.46747626991230096>,0.008701650584808223
    ,<0.39106893472088444,-1.51949707159045,-0.4661406643116448>,0.008714594686749191
    ,<0.37813116608738107,-1.51020236712842,-0.46466749750230224>,0.008726955356075762
    ,<0.36648512127894706,-1.499331654032964,-0.46318956992692234>,0.008738771067525925
    ,<0.3566880803957605,-1.4867545265125122,-0.461827501501807>,0.008750076994045604
    ,<0.3492543787803798,-1.4726343788416631,-0.46066167164274247>,0.008760905352682195
    ,<0.34450819443228603,-1.4573884574038118,-0.45965871940122444>,0.008771285707989934
    ,<0.3424875720573339,-1.4415443907482748,-0.4587314262866169>,0.008781245238899917
    ,<0.3429452621211361,-1.4255770311980256,-0.4578236456941266>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
