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
    ,<0.3276648732608833,-2.0311177557043916,-0.3691968707809948>,0.0
    ,<0.34101501881356844,-2.0249517409784192,-0.3755066532594597>,0.001444405933878283
    ,<0.35449511225369074,-2.0190457957926085,-0.38179986072724814>,0.002733688514425582
    ,<0.36789475252362697,-2.012853161627984,-0.3879930457367101>,0.0037941133653625076
    ,<0.38097627519442206,-2.0058356053838047,-0.39398019015242075>,0.0046307451971068355
    ,<0.39339998376453184,-1.9975165509465174,-0.39969279692765086>,0.005283185474353696
    ,<0.4047087566913518,-1.987561217141224,-0.40509355758741455>,0.005794598874521764
    ,<0.41435915109728944,-1.9758492847687097,-0.41017771279601023>,0.00620058003411749
    ,<0.4218026922087431,-1.962517579205116,-0.4149736889775805>,0.006527801879788091
    ,<0.4266020821551746,-1.9479484209804412,-0.4195376400907453>,0.006795619711330263
    ,<0.4285183466808088,-1.9326709739918355,-0.42390630846190136>,0.007018006566011825
    ,<0.4275905294520751,-1.9172295497557608,-0.4280199586359021>,0.007205119848667835
    ,<0.4241084992260211,-1.9020619941886925,-0.43176376390016175>,0.007364433711532417
    ,<0.41853774862962323,-1.8874235658368868,-0.4350568237854485>,0.0075015263935279105
    ,<0.4114546932893258,-1.873349294041947,-0.43786335168671847>,0.007620622272343326
    ,<0.4034822411812638,-1.8596695066849949,-0.4401899875590098>,0.007724966207910139
    ,<0.39525687809179416,-1.8460736712467125,-0.44208474383347784>,0.007817084460335388
    ,<0.38742583091261257,-1.8322045497724293,-0.4436372879430682>,0.007898968749670325
    ,<0.3806542909798246,-1.817766960030767,-0.4449822494052568>,0.007972207813666372
    ,<0.3756189531125782,-1.8026294850719158,-0.44627285605642036>,0.008038082702723609
    ,<0.3729742837258167,-1.7868976285555982,-0.4475886230711594>,0.008097636716798745
    ,<0.3732527527749848,-1.7709468928953391,-0.44889883222646587>,0.008151727381894005
    ,<0.376726929249144,-1.755373230992344,-0.45015417381402034>,0.008201065543276747
    ,<0.38331901729004914,-1.7408369546636564,-0.4513312172517051>,0.008246245102718756
    ,<0.3926110425601296,-1.7278530643152805,-0.45241866641087564>,0.00828776588047385
    ,<0.40396191301717344,-1.7166170194802468,-0.4534132287335527>,0.008326051367736582
    ,<0.4166713088744315,-1.7069370002245656,-0.45432255259935755>,0.00836146264109268
    ,<0.4300999604171869,-1.6982773842798247,-0.45517537042205497>,0.008394309364827233
    ,<0.4436911722918898,-1.689874307477622,-0.45601820616112315>,0.008424858562469344
    ,<0.456897296181895,-1.6808814383729391,-0.4569022147702675>,0.00845334166411343
    ,<0.46906836660379486,-1.6705387327354486,-0.4578806590981269>,0.008479960209706025
    ,<0.4793843230622885,-1.6583574665312963,-0.4590023940482646>,0.008504890496255251
    ,<0.48691128252599164,-1.6442981493835485,-0.4603207265454016>,0.008528287388947346
    ,<0.49079938406781065,-1.6288506379024519,-0.46183992136955115>,0.008550287465601714
    ,<0.49054017114132786,-1.6129362902387048,-0.4634768197008547>,0.008571011625971648
    ,<0.48613996755092764,-1.5976411368820964,-0.465105069606831>,0.00859056726871202
    ,<0.4781237561436016,-1.5838766053287467,-0.4665871470974554>,0.008609050116966811
    ,<0.4673462539460187,-1.5721171850120461,-0.46779014880823516>,0.008626545756733304
    ,<0.45472505709926847,-1.562324227336785,-0.4686037586489566>,0.008643130939168025
    ,<0.4410508523787108,-1.5540340987069454,-0.4689699825765559>,0.00865887468788217
    ,<0.4269266803444009,-1.546528049474875,-0.46887722608666255>,0.008673839244344611
    ,<0.412820641843709,-1.5390049683460956,-0.4683558379748426>,0.008688080878257348
    ,<0.39916319220172264,-1.5307251591382378,-0.467476067551921>,0.008701650584808223
    ,<0.3864197682786146,-1.5211231741059257,-0.46634851551020107>,0.008714594686749191
    ,<0.3750930062454475,-1.50989048896919,-0.46513439622971864>,0.008726955356075762
    ,<0.3657024217027635,-1.4969889453905934,-0.46396323707625814>,0.008738771067525925
    ,<0.3586991466274206,-1.4826423693975155,-0.46289498771646437>,0.008750076994045604
    ,<0.35432120451833393,-1.4672830660408487,-0.4619273480000789>,0.008760905352682195
    ,<0.3525357713568523,-1.4514091520608643,-0.4610166272760854>,0.008771285707989934
    ,<0.3530261417345271,-1.4354417700813027,-0.4601248145014507>,0.008781245238899917
    ,<0.35524795431463146,-1.419622499623149,-0.4592240019996038>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
