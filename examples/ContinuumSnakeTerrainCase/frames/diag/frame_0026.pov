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
    ,<0.07425774070532369,-2.9519549186939416,-0.21383989526010924>,0.0
    ,<0.06930948262105077,-2.938408778046099,-0.22076785554992642>,0.001444405933878283
    ,<0.06600903224336395,-2.924383802885481,-0.2277210913205976>,0.002733688514425582
    ,<0.06457384853889755,-2.9100767293752097,-0.2347338575898506>,0.0037941133653625076
    ,<0.06508731409360956,-2.895735990019604,-0.24183493643741694>,0.0046307451971068355
    ,<0.06745279315130003,-2.881579451807795,-0.24895699861795947>,0.005283185474353696
    ,<0.0714593111256062,-2.8677410823056153,-0.2559677364437066>,0.005794598874521764
    ,<0.07683944848452566,-2.8542673779846894,-0.26276349420425493>,0.00620058003411749
    ,<0.08329161432931437,-2.84111922178366,-0.2692542257658424>,0.006527801879788091
    ,<0.09049792331688848,-2.8281828669733016,-0.27536309280332494>,0.006795619711330263
    ,<0.0981338193462756,-2.8152908618440415,-0.2810247755295954>,0.007018006566011825
    ,<0.10587108898089258,-2.8022508549867955,-0.2861849874291151>,0.007205119848667835
    ,<0.11337839987738303,-2.7888777422013193,-0.2908018453176064>,0.007364433711532417
    ,<0.12032451664607281,-2.775025824983886,-0.2948467362393296>,0.0075015263935279105
    ,<0.1263899176108103,-2.760613157923494,-0.2983067907250896>,0.007620622272343326
    ,<0.13128610397137677,-2.7456369344947795,-0.3011867041379691>,0.007724966207910139
    ,<0.13478115431008322,-2.7301783242235773,-0.3035120384965505>,0.007817084460335388
    ,<0.13671526228905004,-2.7143799827487642,-0.30531234362341253>,0.007898968749670325
    ,<0.13702199368738963,-2.698418648898606,-0.3066156949865538>,0.007972207813666372
    ,<0.13575976066369094,-2.6824726992900523,-0.30744270197033663>,0.008038082702723609
    ,<0.13311348128458886,-2.666681340473299,-0.30781421734384407>,0.008097636716798745
    ,<0.12936686378626622,-2.651109790727829,-0.3077794759997821>,0.008151727381894005
    ,<0.12487607505696237,-2.6357380730977025,-0.30741427169307317>,0.008201065543276747
    ,<0.12004286080691288,-2.6204765301331516,-0.30682298825862975>,0.008246245102718756
    ,<0.11528888549293892,-2.6051936530322846,-0.30614219101409584>,0.00828776588047385
    ,<0.11103787666863528,-2.5897598620204256,-0.3055371093789988>,0.008326051367736582
    ,<0.10768963135651885,-2.574096344989036,-0.3052054461442494>,0.00836146264109268
    ,<0.10558779176422631,-2.5582141656332804,-0.30537419560676243>,0.008394309364827233
    ,<0.10498326190004177,-2.542229706561354,-0.30627893600855816>,0.008424858562469344
    ,<0.10603625638039638,-2.526344477389021,-0.3080522820622141>,0.00845334166411343
    ,<0.10879063108619065,-2.510774749330041,-0.3106067438419199>,0.008479960209706025
    ,<0.11314077276018988,-2.495681789710609,-0.31373280583551255>,0.008504890496255251
    ,<0.11883554784387104,-2.481126575207665,-0.317223875037601>,0.008528287388947346
    ,<0.1255177761464199,-2.467046368638497,-0.3209024683867869>,0.008550287465601714
    ,<0.13277036577867074,-2.453262825616044,-0.32462079796856025>,0.008571011625971648
    ,<0.14014561211384313,-2.439524862459983,-0.3282622852869909>,0.00859056726871202
    ,<0.1471822297414137,-2.4255674399378955,-0.33173499570910964>,0.008609050116966811
    ,<0.1534216641230123,-2.4111808755154716,-0.33497010199234034>,0.008626545756733304
    ,<0.1584379966911826,-2.396265969862538,-0.33792939878386535>,0.008643130939168025
    ,<0.16187699010763013,-2.3808584117873055,-0.3406044168334037>,0.00865887468788217
    ,<0.1635030503976946,-2.3651126946660406,-0.34301646735608066>,0.008673839244344611
    ,<0.16323298814914775,-2.349254925237846,-0.3452183120858474>,0.008688080878257348
    ,<0.16114734906471542,-2.333516283957888,-0.34729769989184534>,0.008701650584808223
    ,<0.1574713582741973,-2.318070325574253,-0.3493461027946251>,0.008714594686749191
    ,<0.15251570503104106,-2.302985311983135,-0.3513651959761001>,0.008726955356075762
    ,<0.14665159029876426,-2.2882182916483464,-0.3532902759523406>,0.008738771067525925
    ,<0.14029459143882056,-2.273639763295939,-0.3550799671709048>,0.008750076994045604
    ,<0.13388516202213405,-2.259067473896691,-0.35672246506845295>,0.008760905352682195
    ,<0.12787926201254243,-2.244310968666902,-0.35823682843595916>,0.008771285707989934
    ,<0.12274333538860052,-2.229223386025958,-0.35967625294535965>,0.008781245238899917
    ,<0.11894997445933018,-2.2137441484621925,-0.3611020495514117>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
