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
    ,<0.11413062905770524,-2.7069830916074973,-0.3083616111244857>,0.0
    ,<0.11383582648819972,-2.691019508146275,-0.3093667013935443>,0.001444405933878283
    ,<0.1163746623543871,-2.6752534284669434,-0.3103207124557664>,0.002733688514425582
    ,<0.12147043622841394,-2.6601145927196215,-0.311205666825043>,0.0037941133653625076
    ,<0.12865426895142792,-2.6458431561249323,-0.31202847436024567>,0.0046307451971068355
    ,<0.13736091236706657,-2.632443886741486,-0.31283169458195953>,0.005283185474353696
    ,<0.14700771850400848,-2.6197068147166003,-0.313681458024972>,0.005794598874521764
    ,<0.15703546253570702,-2.607274696468179,-0.3146489252575929>,0.00620058003411749
    ,<0.16690718542408828,-2.594729414918062,-0.3157709752081792>,0.006527801879788091
    ,<0.17608505114592582,-2.5816842458725833,-0.317065429485221>,0.006795619711330263
    ,<0.18400386619733586,-2.5678627900510294,-0.31855473709611615>,0.007018006566011825
    ,<0.19004906854375905,-2.553146115170197,-0.3201924840104847>,0.007205119848667835
    ,<0.1936293917126213,-2.5376525493688185,-0.3218851470898166>,0.007364433711532417
    ,<0.1943104160354956,-2.5217641864017653,-0.3235498604533219>,0.0075015263935279105
    ,<0.19191872586824454,-2.506033300273922,-0.325114881669824>,0.007620622272343326
    ,<0.18659524798102867,-2.4910245743573185,-0.3265294239347132>,0.007724966207910139
    ,<0.1787637553347438,-2.477142227624184,-0.32777839594880254>,0.007817084460335388
    ,<0.16903142736722032,-2.464508368194335,-0.32890896233618494>,0.007898968749670325
    ,<0.15806093085208273,-2.4529232949549713,-0.3299642482108301>,0.007972207813666372
    ,<0.14649675704122958,-2.4419192473119544,-0.3309363356002164>,0.008038082702723609
    ,<0.1349659226412308,-2.43087544366257,-0.3318494657547804>,0.008097636716798745
    ,<0.12413398018385478,-2.4191475341121156,-0.33278913690918743>,0.008151727381894005
    ,<0.11476227936962019,-2.4062341833142393,-0.3338772490492133>,0.008201065543276747
    ,<0.10770787871948094,-2.391943364772263,-0.3352228378169983>,0.008246245102718756
    ,<0.10382679065444927,-2.3765147121978405,-0.33687568046741717>,0.00828776588047385
    ,<0.1037830201620494,-2.3606397778387302,-0.3388348851759408>,0.008326051367736582
    ,<0.1078291376466161,-2.345325839242813,-0.3410643111549976>,0.00836146264109268
    ,<0.11568658307596913,-2.331610556453051,-0.34352137822699536>,0.008394309364827233
    ,<0.1266189632439725,-2.3202381618700167,-0.346179403705112>,0.008424858562469344
    ,<0.1396762938335835,-2.3114400894173857,-0.34901395052402995>,0.00845334166411343
    ,<0.1539716318779309,-2.304907997853919,-0.3520035241578913>,0.008479960209706025
    ,<0.1688496255529526,-2.29992078516051,-0.355130081043781>,0.008504890496255251
    ,<0.18388960789733982,-2.2955225033371107,-0.35836940047076465>,0.008528287388947346
    ,<0.1987786238090397,-2.2906851545885605,-0.36168170011594286>,0.008550287465601714
    ,<0.21313903473647694,-2.2844524118382328,-0.3649959811882994>,0.008571011625971648
    ,<0.22639649939277592,-2.27609258315261,-0.36821361245551143>,0.00859056726871202
    ,<0.23777432008079538,-2.2652603120726673,-0.37123783037138536>,0.008609050116966811
    ,<0.24644485670222768,-2.2521022480100257,-0.3739899920045685>,0.008626545756733304
    ,<0.2517724857289456,-2.237216796960473,-0.3764137189603111>,0.008643130939168025
    ,<0.25352353134831507,-2.2214542238000794,-0.3784806873984487>,0.00865887468788217
    ,<0.2519286366217012,-2.2056333228278517,-0.38019561152595116>,0.008673839244344611
    ,<0.2475866753243585,-2.190304508130207,-0.3815969086821933>,0.008688080878257348
    ,<0.24129116341650012,-2.1756468457836844,-0.3827495721805077>,0.008701650584808223
    ,<0.23387719440052787,-2.1615064205026164,-0.38370522434490467>,0.008714594686749191
    ,<0.22614715786725978,-2.147525632590608,-0.3844949831408938>,0.008726955356075762
    ,<0.2188651203487713,-2.1333002782170367,-0.3851669382673225>,0.008738771067525925
    ,<0.21277025319414855,-2.118525062613382,-0.38580500229580816>,0.008750076994045604
    ,<0.20856927544866502,-2.1031050738300867,-0.3864977016717636>,0.008760905352682195
    ,<0.20688672257425608,-2.087214378868042,-0.38727339766815794>,0.008771285707989934
    ,<0.2081728633476114,-2.0712884514388876,-0.38809721537290653>,0.008781245238899917
    ,<0.21260755745591745,-2.0559386669983506,-0.38892850238440013>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
