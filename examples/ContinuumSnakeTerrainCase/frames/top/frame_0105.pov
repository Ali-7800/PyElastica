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
    ,<0.21258721700149077,-2.45349920348028,-0.3002629317369771>,0.0
    ,<0.21043000028015196,-2.4385282495688276,-0.3054713213792206>,0.001444405933878283
    ,<0.21101480243663173,-2.4234775019168073,-0.3108552890806048>,0.002733688514425582
    ,<0.21452518830797493,-2.4089572428123014,-0.31656624078346784>,0.0037941133653625076
    ,<0.22087455550599494,-2.3956019609095596,-0.32264902901173154>,0.0046307451971068355
    ,<0.229775398543773,-2.3839285590345867,-0.3289876688661819>,0.005283185474353696
    ,<0.24075216032077093,-2.37423744980665,-0.3354128433829786>,0.005794598874521764
    ,<0.2532348008310631,-2.366527563499189,-0.3417730603594993>,0.00620058003411749
    ,<0.26669991336021315,-2.36048831475073,-0.34793349802082374>,0.006527801879788091
    ,<0.280744139205981,-2.3555568491654193,-0.3537794420785302>,0.006795619711330263
    ,<0.29507594918618685,-2.3509990695299763,-0.3592146820826592>,0.007018006566011825
    ,<0.3094360110024758,-2.345992120561357,-0.3641567729023459>,0.007205119848667835
    ,<0.32347853373573493,-2.3397105625496755,-0.36851957065248825>,0.007364433711532417
    ,<0.33665496506300113,-2.3314406155857506,-0.3722055100748168>,0.0075015263935279105
    ,<0.3481614791120032,-2.320733361537613,-0.3751177089676828>,0.007620622272343326
    ,<0.35699905295080775,-2.307575809827044,-0.37717093030481574>,0.007724966207910139
    ,<0.36217635852783,-2.292500083321738,-0.3783175588399924>,0.007817084460335388
    ,<0.36299381292559707,-2.276541082636283,-0.37857697400564483>,0.007898968749670325
    ,<0.359284049490137,-2.261002751536468,-0.3780561720145432>,0.007972207813666372
    ,<0.3514780817595477,-2.2470962077338017,-0.3769511825993533>,0.008038082702723609
    ,<0.34044858782203635,-2.235609330466246,-0.3755267284864425>,0.008097636716798745
    ,<0.3272178381476037,-2.226745306107572,-0.37408858041049287>,0.008151727381894005
    ,<0.31270279052959515,-2.220133907452864,-0.3729157012487801>,0.008201065543276747
    ,<0.2975793142615414,-2.21496576669325,-0.372246371111912>,0.008246245102718756
    ,<0.2822925530318403,-2.210223058621947,-0.3722694861082162>,0.00828776588047385
    ,<0.2672235111604556,-2.204854333063332,-0.3730014663405028>,0.008326051367736582
    ,<0.25284293681330844,-2.197934868134007,-0.37431118983260164>,0.00836146264109268
    ,<0.2398194788276258,-2.1887898207319223,-0.37605680228748367>,0.008394309364827233
    ,<0.2290254751857229,-2.1771487909023266,-0.37808838899232977>,0.008424858562469344
    ,<0.22137293814100317,-2.1632626607887806,-0.3802333953352271>,0.00845334166411343
    ,<0.21754219758919,-2.1478719126034806,-0.38230048446019455>,0.008479960209706025
    ,<0.2177396831301291,-2.1319844683081057,-0.3840997666726334>,0.008504890496255251
    ,<0.22162523514094623,-2.1165375235299253,-0.3854744338306095>,0.008528287388947346
    ,<0.22844286775661835,-2.1021036068772356,-0.3863291305459933>,0.008550287465601714
    ,<0.23725643054816598,-2.0887708178465467,-0.3866461220570933>,0.008571011625971648
    ,<0.24714625891395692,-2.0762122160629506,-0.38648856083422856>,0.00859056726871202
    ,<0.25729272188208313,-2.063868256618002,-0.3859954535044653>,0.008609050116966811
    ,<0.2669459931978228,-2.051141043810808,-0.38535969369743>,0.008626545756733304
    ,<0.27538364641687046,-2.037572924424249,-0.38483264635514064>,0.008643130939168025
    ,<0.2818804035870505,-2.022963678186692,-0.3845717345842686>,0.00865887468788217
    ,<0.2857741297815358,-2.0074500048743937,-0.38454604613969046>,0.008673839244344611
    ,<0.2866317347442517,-1.9914791988540015,-0.3846885059561547>,0.008688080878257348
    ,<0.2843660430603898,-1.975647214570887,-0.3849483486838448>,0.008701650584808223
    ,<0.2792457884209367,-1.9604964128777886,-0.3852748544265849>,0.008714594686749191
    ,<0.271808009399203,-1.9463392776263277,-0.3856176845626159>,0.008726955356075762
    ,<0.2627216568990992,-1.9331790342627517,-0.3859319344682486>,0.008738771067525925
    ,<0.25267575548222576,-1.9207348937062534,-0.38618440862711545>,0.008750076994045604
    ,<0.24233269197768798,-1.9085355360741465,-0.38635906855009655>,0.008760905352682195
    ,<0.23234391382006833,-1.896043947508851,-0.3864618572078527>,0.008771285707989934
    ,<0.2233909719364141,-1.8827890720443858,-0.38652925902586543>,0.008781245238899917
    ,<0.21621653414682504,-1.868490331683589,-0.38660571048123055>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
