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
    ,<0.22231085061128558,-2.4018535035695816,-0.3241614210282973>,0.0
    ,<0.23444214251374423,-2.3938191231222907,-0.3308127120533473>,0.001444405933878283
    ,<0.24752904004010318,-2.3873898919156327,-0.33740104930379905>,0.002733688514425582
    ,<0.26123414524032607,-2.38220876815662,-0.3438362073397822>,0.0037941133653625076
    ,<0.2753001202834332,-2.37772678378673,-0.35001158533321425>,0.0046307451971068355
    ,<0.28952112189396634,-2.3732876243460743,-0.3558492671254852>,0.005283185474353696
    ,<0.3036770455215923,-2.3681878132250396,-0.361288342697528>,0.005794598874521764
    ,<0.31744907692213176,-2.361745071739139,-0.3662874953553829>,0.00620058003411749
    ,<0.33036746543829304,-2.3534040702815355,-0.37074769397245433>,0.006527801879788091
    ,<0.3417982852064917,-2.342845830038889,-0.3745121964802598>,0.006795619711330263
    ,<0.3509888260567183,-2.330069162757041,-0.37743883198779>,0.007018006566011825
    ,<0.3572054536526597,-2.3154506690132797,-0.37941060769398305>,0.007205119848667835
    ,<0.3599016795495808,-2.2997017525557517,-0.3803688749157112>,0.007364433711532417
    ,<0.35886572618617385,-2.2837285091055484,-0.3803331773131948>,0.0075015263935279105
    ,<0.35427575543115936,-2.268421769809621,-0.3794118462660948>,0.007620622272343326
    ,<0.3466338421977624,-2.2544495009754346,-0.3777978220016005>,0.007724966207910139
    ,<0.336627391719491,-2.2421254014830754,-0.37574296170126253>,0.007817084460335388
    ,<0.3249715745188176,-2.231381487900767,-0.37353070082077167>,0.007898968749670325
    ,<0.3123206269651598,-2.2218030209019735,-0.3714605141573244>,0.007972207813666372
    ,<0.29924704955243764,-2.2127330807023338,-0.3697671427999623>,0.008038082702723609
    ,<0.28626425799133814,-2.2034408020836915,-0.368611339628387>,0.008097636716798745
    ,<0.2739302927515456,-2.1932461043352762,-0.3680422274658021>,0.008151727381894005
    ,<0.26289978991272217,-2.1816392174612496,-0.36799946162579966>,0.008201065543276747
    ,<0.2538926235985385,-2.168405193822326,-0.36839282826743913>,0.008246245102718756
    ,<0.24760169454336856,-2.1536966135505278,-0.3690934822160249>,0.00828776588047385
    ,<0.24453349335715224,-2.1380062546727165,-0.3699456074813326>,0.008326051367736582
    ,<0.24486089891649923,-2.1220240839702,-0.3707869656569156>,0.00836146264109268
    ,<0.24836634470043312,-2.1064218993936348,-0.37147505088575494>,0.008394309364827233
    ,<0.2545058156596148,-2.091649066346397,-0.37191391759790604>,0.008424858562469344
    ,<0.2625524391375085,-2.0778184132919812,-0.37207571554345276>,0.00845334166411343
    ,<0.2717429087349989,-2.0647205555626167,-0.37201537348074726>,0.008479960209706025
    ,<0.281329915991544,-2.051911372723347,-0.37185467298049446>,0.008504890496255251
    ,<0.29058317161394287,-2.0388681668788275,-0.37177719678030136>,0.008528287388947346
    ,<0.29864433203154594,-2.0250607720242466,-0.37192547203831156>,0.008550287465601714
    ,<0.3045607998272297,-2.0102058975582278,-0.372301367254498>,0.008571011625971648
    ,<0.3075168767042094,-1.9944985467604652,-0.3728577548972986>,0.00859056726871202
    ,<0.30698373987221006,-1.9785301919814409,-0.373553785827189>,0.008609050116966811
    ,<0.3028695117098468,-1.9630967548602412,-0.37433699459256536>,0.008626545756733304
    ,<0.2955340986458531,-1.9489123986317927,-0.37515037674005103>,0.008643130939168025
    ,<0.28564933685289445,-1.9363695897010167,-0.37594206820680803>,0.00865887468788217
    ,<0.2739952706272673,-1.9254462831920935,-0.3766718134478215>,0.008673839244344611
    ,<0.2612956092964033,-1.9157513237677954,-0.37731477311265005>,0.008688080878257348
    ,<0.24815625210198547,-1.9066542295962907,-0.37786404404095164>,0.008701650584808223
    ,<0.23509851535468468,-1.8974345972692386,-0.3783333605878638>,0.008714594686749191
    ,<0.2226351893049156,-1.8874241351842187,-0.37876069237938337>,0.008726955356075762
    ,<0.21132926267654323,-1.8761227504647626,-0.3792121637355422>,0.008738771067525925
    ,<0.20179954125075084,-1.8632857028223953,-0.3797727695983423>,0.008750076994045604
    ,<0.19464692665536665,-1.848988095777118,-0.38044975533212017>,0.008760905352682195
    ,<0.19033846818172373,-1.83359599025303,-0.3811848229491588>,0.008771285707989934
    ,<0.1890991544024993,-1.81766174976522,-0.38194194374657997>,0.008781245238899917
    ,<0.19085453249966236,-1.801776572333481,-0.3826992505822131>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
