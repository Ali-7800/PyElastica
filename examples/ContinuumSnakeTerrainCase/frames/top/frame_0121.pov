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
    ,<0.24164599534135825,-2.3610514931822744,-0.36978548012474033>,0.0
    ,<0.25715488417700355,-2.357213094274064,-0.3706483902696318>,0.001444405933878283
    ,<0.27215257967379536,-2.3516987629235806,-0.37147701326028015>,0.002733688514425582
    ,<0.28616737494218325,-2.3440045008041377,-0.3721417260714139>,0.0037941133653625076
    ,<0.2985550496365435,-2.3338786674697936,-0.37252403804855505>,0.0046307451971068355
    ,<0.30858425363227443,-2.3214055731083856,-0.3725472130239734>,0.005283185474353696
    ,<0.31559241121446235,-2.307021712639903,-0.37218322183557867>,0.005794598874521764
    ,<0.31914938975854973,-2.291434386714759,-0.37145537346884944>,0.00620058003411749
    ,<0.319164341682395,-2.2754619156907325,-0.370442193193549>,0.006527801879788091
    ,<0.3158984345247466,-2.2598376591450835,-0.3692732719333962>,0.006795619711330263
    ,<0.30988591052060654,-2.245051877646086,-0.36813598180600815>,0.007018006566011825
    ,<0.30177514730236393,-2.2312960071896604,-0.3671812345708834>,0.007205119848667835
    ,<0.29221679495387637,-2.21848607132763,-0.3664901946240247>,0.007364433711532417
    ,<0.2818468567679873,-2.2063020304101966,-0.36613665909746035>,0.0075015263935279105
    ,<0.271301454639757,-2.194258808005149,-0.36611468344303766>,0.007620622272343326
    ,<0.2612162768709809,-2.1818309865011414,-0.3663526888855121>,0.007724966207910139
    ,<0.25223379159518716,-2.1685908594380683,-0.36678591897061114>,0.007817084460335388
    ,<0.24502655739753462,-2.1543127125805253,-0.3673441881651526>,0.007898968749670325
    ,<0.24025546906757017,-2.139050230120584,-0.36795077780700847>,0.007972207813666372
    ,<0.23846873320646636,-2.1231597682702565,-0.36852486580261995>,0.008038082702723609
    ,<0.2399735494789633,-2.1072382015420126,-0.3689897974047262>,0.008097636716798745
    ,<0.24474272985388323,-2.091970529262841,-0.36928665067716127>,0.008151727381894005
    ,<0.2524117870521642,-2.077931958262922,-0.36938984746625103>,0.008201065543276747
    ,<0.2623760807538861,-2.065417890033385,-0.3693209106478453>,0.008246245102718756
    ,<0.27393548259410944,-2.0543613168951387,-0.3691546639435768>,0.00828776588047385
    ,<0.28641986824469834,-2.044363369922893,-0.3690263047419063>,0.008326051367736582
    ,<0.29918421394082595,-2.0347286821430464,-0.36907830858116686>,0.00836146264109268
    ,<0.3115215145449925,-2.024556062319075,-0.3694393192192483>,0.008394309364827233
    ,<0.3226039328138306,-2.0130411223778175,-0.3701284588334191>,0.008424858562469344
    ,<0.331444711344999,-1.9997466051077988,-0.3710639512201935>,0.00845334166411343
    ,<0.3369831268501945,-1.9847886628242943,-0.3721825977153862>,0.008479960209706025
    ,<0.3383449622699796,-1.9689081538446258,-0.37341371980943744>,0.008504890496255251
    ,<0.3351604091681552,-1.9532970894205768,-0.37468196345482346>,0.008528287388947346
    ,<0.32773320706551856,-1.9392013638127297,-0.37591776226412754>,0.008550287465601714
    ,<0.31692680147485297,-1.9274856250054593,-0.37706745404049047>,0.008571011625971648
    ,<0.30383329784626634,-1.9183825540456665,-0.3780988738456085>,0.00859056726871202
    ,<0.28943366344826527,-1.9115095976523995,-0.3790030651029307>,0.008609050116966811
    ,<0.27442874228234554,-1.9060638618564252,-0.37979508403484435>,0.008626545756733304
    ,<0.25927029458566353,-1.901050386692802,-0.3805169102453762>,0.008643130939168025
    ,<0.24431231236974357,-1.8954698129490166,-0.3812436389989651>,0.00865887468788217
    ,<0.22997223673618933,-1.8884691243468954,-0.3820912856613023>,0.008673839244344611
    ,<0.21681132292241947,-1.8794773894327903,-0.38322471928041>,0.008688080878257348
    ,<0.20549044265564811,-1.8683147226759094,-0.3848412943679842>,0.008701650584808223
    ,<0.19661270691229893,-1.855211321603955,-0.38713774114605354>,0.008714594686749191
    ,<0.19053764750220986,-1.8406946237525674,-0.39005638386258695>,0.008726955356075762
    ,<0.1873105944267909,-1.8253692011974514,-0.39335102323810217>,0.008738771067525925
    ,<0.18665359301319662,-1.8097666298318542,-0.3968477076695527>,0.008750076994045604
    ,<0.1880396948726094,-1.7942308097992632,-0.40042175634783406>,0.008760905352682195
    ,<0.1907838691853242,-1.778875924511776,-0.4039904912045433>,0.008771285707989934
    ,<0.19411341895529738,-1.7636254645309024,-0.40751219277968287>,0.008781245238899917
    ,<0.1972004557553149,-1.7483141984452046,-0.41098810803725533>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
