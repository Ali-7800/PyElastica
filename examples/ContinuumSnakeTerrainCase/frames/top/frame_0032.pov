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
    ,<0.05666516207841037,-2.9462349309855336,-0.22758049793706278>,0.0
    ,<0.057376459806038414,-2.9312854339600576,-0.23321708720489315>,0.001444405933878283
    ,<0.06009683363246817,-2.9165846566802855,-0.23889204317265927>,0.002733688514425582
    ,<0.06461515575683018,-2.9023733803774476,-0.24472930754797192>,0.0037941133653625076
    ,<0.07058418356968599,-2.8887762658495673,-0.25076640281280693>,0.0046307451971068355
    ,<0.07765174530429168,-2.8757751794714674,-0.2569234966528697>,0.005283185474353696
    ,<0.08543778422822582,-2.86321362468412,-0.26312204172837816>,0.005794598874521764
    ,<0.09354478688388121,-2.8508244297912393,-0.269252457254949>,0.00620058003411749
    ,<0.10158422790728563,-2.8383051705068114,-0.27520193873412696>,0.006527801879788091
    ,<0.1091708970150671,-2.825374029171448,-0.2808551004796513>,0.006795619711330263
    ,<0.11592169435448968,-2.811829423555233,-0.2861143727899076>,0.007018006566011825
    ,<0.12147695342116488,-2.797586114489917,-0.2909014922679735>,0.007205119848667835
    ,<0.12552998124002177,-2.7826839232550533,-0.295155739462272>,0.007364433711532417
    ,<0.127866167553925,-2.7672706683568595,-0.29883418295534303>,0.0075015263935279105
    ,<0.12839720226451948,-2.7515624206800453,-0.30191256709631137>,0.007620622272343326
    ,<0.1271791952707044,-2.7357872740844353,-0.30438696368765156>,0.007724966207910139
    ,<0.12440774882374342,-2.720129690948557,-0.3062765494218526>,0.007817084460335388
    ,<0.12040018412342146,-2.7046846344476783,-0.3076166709238489>,0.007898968749670325
    ,<0.11557532812312754,-2.689437954128287,-0.30842776910512903>,0.007972207813666372
    ,<0.11042164375421114,-2.674280918799658,-0.3087285068498883>,0.008038082702723609
    ,<0.10545810358806629,-2.659057200801069,-0.30856726541689455>,0.008097636716798745
    ,<0.10121749579845926,-2.643623162580672,-0.3080194679834918>,0.008151727381894005
    ,<0.09822168980702654,-2.6279108296393034,-0.3071893788807252>,0.008201065543276747
    ,<0.09694014993893153,-2.6119741940596373,-0.30621241448247005>,0.008246245102718756
    ,<0.09773117126711607,-2.59600356915566,-0.30526059074824224>,0.00828776588047385
    ,<0.10079467798970508,-2.5802980585209547,-0.30451566579639117>,0.008326051367736582
    ,<0.10613784825536714,-2.5652018565704218,-0.30412313315461625>,0.00836146264109268
    ,<0.11355770381491545,-2.5510094987262324,-0.3042147953882141>,0.008394309364827233
    ,<0.12267444318373748,-2.537860744377321,-0.30481080703699187>,0.008424858562469344
    ,<0.1330104945916622,-2.525677851664009,-0.3058290706526733>,0.00845334166411343
    ,<0.14405457505258623,-2.5141682919709463,-0.30716975983417666>,0.008479960209706025
    ,<0.1553065833424701,-2.5028902001623763,-0.3087261961409574>,0.008504890496255251
    ,<0.16625950934599268,-2.491342506056232,-0.3104319441075011>,0.008528287388947346
    ,<0.17638216242491325,-2.4790775724525056,-0.31226040355741436>,0.008550287465601714
    ,<0.18512087082232423,-2.4658099532559157,-0.31422762453832187>,0.008571011625971648
    ,<0.19195113162307847,-2.451493935293594,-0.3163942287991013>,0.00859056726871202
    ,<0.1964680396948661,-2.4363357569736155,-0.3188631350024891>,0.008609050116966811
    ,<0.19837588346699736,-2.42070499310523,-0.3217321085205776>,0.008626545756733304
    ,<0.19757072233589903,-2.4050595302247673,-0.32501213946306173>,0.008643130939168025
    ,<0.19422765403984366,-2.389827257655244,-0.32861338134563467>,0.00865887468788217
    ,<0.18874630430077646,-2.375273597568345,-0.3323920711951328>,0.008673839244344611
    ,<0.18165375738632555,-2.3614491845423617,-0.33622383699737124>,0.008688080878257348
    ,<0.17351609390767186,-2.348199910692292,-0.3400074741143169>,0.008701650584808223
    ,<0.16489080240385426,-2.3352280774099574,-0.3436677853750005>,0.008714594686749191
    ,<0.15631324247559067,-2.322178392405119,-0.3471588017614657>,0.008726955356075762
    ,<0.1483023927939003,-2.308727174502269,-0.35046677192852926>,0.008738771067525925
    ,<0.14137359410047665,-2.2946464179401307,-0.35359356043576506>,0.008750076994045604
    ,<0.1360336814882001,-2.2798524957916184,-0.35653859346519495>,0.008760905352682195
    ,<0.13273670945563565,-2.2644429157351924,-0.3593149804792439>,0.008771285707989934
    ,<0.13181471631930924,-2.248690553660435,-0.3619668169477588>,0.008781245238899917
    ,<0.13342520255208404,-2.2329861959006228,-0.3645705563476536>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
