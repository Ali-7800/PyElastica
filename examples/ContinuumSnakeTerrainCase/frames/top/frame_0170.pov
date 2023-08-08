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
    ,<0.2511066228236849,-2.0865270847711828,-0.34851900738449004>,0.0
    ,<0.262561740566773,-2.0769393034069528,-0.35424382418717915>,0.001444405933878283
    ,<0.2721086658412312,-2.0654702625662393,-0.3600079093529735>,0.002733688514425582
    ,<0.27957888018152616,-2.052593132388041,-0.3658605221711687>,0.0037941133653625076
    ,<0.2850523048950229,-2.03881256309447,-0.37185419218622345>,0.0046307451971068355
    ,<0.2888193065965511,-2.0245316456935547,-0.37801528829373787>,0.005283185474353696
    ,<0.29134886618522565,-2.0099955093499093,-0.3842442261159229>,0.005794598874521764
    ,<0.2932080897960204,-1.995312760874164,-0.390360716325557>,0.00620058003411749
    ,<0.29499196269803896,-1.9805232453403703,-0.3962350280029835>,0.006527801879788091
    ,<0.29728745440911347,-1.9656759506132104,-0.40177128254280425>,0.006795619711330263
    ,<0.3006321733730103,-1.950887205256981,-0.40691282197997264>,0.007018006566011825
    ,<0.30546436013324524,-1.9363782266855765,-0.4116490823250888>,0.007205119848667835
    ,<0.31206670340200193,-1.922468511142595,-0.41603058663919923>,0.007364433711532417
    ,<0.32052966056403714,-1.9095129800649469,-0.4201264009489872>,0.0075015263935279105
    ,<0.33074318044605966,-1.8978167696032464,-0.424018937799075>,0.007620622272343326
    ,<0.3424410412913048,-1.887539228649389,-0.42773419123705825>,0.007724966207910139
    ,<0.3552738222946347,-1.8786341520610645,-0.43123787356005705>,0.007817084460335388
    ,<0.368867237159412,-1.8708498266023805,-0.43453178692468586>,0.007898968749670325
    ,<0.3828695449620804,-1.8637401783056158,-0.43763011467398283>,0.007972207813666372
    ,<0.3969500639724956,-1.8567107827028664,-0.4405476215046089>,0.008038082702723609
    ,<0.4107535626216035,-1.8490869958866287,-0.4432906885612978>,0.008097636716798745
    ,<0.42382184848096077,-1.8402082990512039,-0.44584966074830895>,0.008151727381894005
    ,<0.4355292965395905,-1.8295516363249227,-0.44819955183056726>,0.008201065543276747
    ,<0.445076026132072,-1.8168807402630576,-0.45030543104769416>,0.008246245102718756
    ,<0.4515870419820293,-1.8023759300731133,-0.4521306431337663>,0.00828776588047385
    ,<0.4543112453419589,-1.7866799894475138,-0.453652118385432>,0.008326051367736582
    ,<0.4528468454661779,-1.7707898468286876,-0.45485560703412026>,0.00836146264109268
    ,<0.44727889230936424,-1.7558127224604483,-0.45571694743745794>,0.008394309364827233
    ,<0.4381674100862121,-1.7426688968697055,-0.4562195729372592>,0.008424858562469344
    ,<0.42638365428445024,-1.7318460696873677,-0.4563617097896668>,0.00845334166411343
    ,<0.41286069414597953,-1.723296125242534,-0.45614533453048073>,0.008479960209706025
    ,<0.3983837793846721,-1.7165037494694388,-0.45558300854178224>,0.008504890496255251
    ,<0.3835123379741005,-1.710661106189982,-0.45471493434467597>,0.008528287388947346
    ,<0.3686468311608957,-1.7048441071710727,-0.4535954633492165>,0.008550287465601714
    ,<0.3541672762974937,-1.698160481440581,-0.45227105877315676>,0.008571011625971648
    ,<0.34053896441462195,-1.689901168936881,-0.45081290184864625>,0.00859056726871202
    ,<0.32832699176269803,-1.679665971377538,-0.44933971072122825>,0.008609050116966811
    ,<0.31810666296252893,-1.6674244704906378,-0.448017622769456>,0.008626545756733304
    ,<0.3103279382609117,-1.6534766831005188,-0.4470287570906562>,0.008643130939168025
    ,<0.3052205968695061,-1.638323709800273,-0.4464974621273919>,0.00865887468788217
    ,<0.302697547923508,-1.6225253949014111,-0.4464788185288247>,0.008673839244344611
    ,<0.30234471570944116,-1.6065382268864277,-0.44696069347210526>,0.008688080878257348
    ,<0.3035229936082091,-1.5906092509740501,-0.4478726919874304>,0.008701650584808223
    ,<0.30546770828906467,-1.5747773149903264,-0.4491131583126876>,0.008714594686749191
    ,<0.30737080260316624,-1.5589588718184635,-0.4505738412882216>,0.008726955356075762
    ,<0.30843903014389934,-1.5430745625870508,-0.4521666198865921>,0.008738771067525925
    ,<0.3079344582239383,-1.5271691195133013,-0.45382585457175156>,0.008750076994045604
    ,<0.30523779679028634,-1.5114891652172628,-0.45551783814728003>,0.008760905352682195
    ,<0.2999311040569872,-1.49649351100318,-0.45724295959534633>,0.008771285707989934
    ,<0.29187051022530264,-1.482784415538341,-0.4590025249924393>,0.008781245238899917
    ,<0.2812123659398558,-1.47098346399978,-0.4607738358983362>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
