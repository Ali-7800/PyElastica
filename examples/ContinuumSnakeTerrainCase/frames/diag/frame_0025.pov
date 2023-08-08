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
    ,<0.07639254628660484,-2.9522041361787474,-0.21431874281594368>,0.0
    ,<0.07106583766625028,-2.9387277001508605,-0.2211005335232876>,0.001444405933878283
    ,<0.06719541353893357,-2.9247815852003174,-0.22791648229637887>,0.002733688514425582
    ,<0.06504657574819019,-2.9105090052541036,-0.2348142792197613>,0.0037941133653625076
    ,<0.06477813150039155,-2.896123654109437,-0.2418479685218409>,0.0046307451971068355
    ,<0.06635202640020754,-2.8818350141463136,-0.2489478151192415>,0.005283185474353696
    ,<0.06960495164100032,-2.867786333939143,-0.2559539445350049>,0.005794598874521764
    ,<0.07431824001784017,-2.854054528205734,-0.26275266152519205>,0.00620058003411749
    ,<0.08023004610260674,-2.8406456916740432,-0.26925017826956604>,0.006527801879788091
    ,<0.08705044297996685,-2.8274937046867836,-0.27536823002858396>,0.006795619711330263
    ,<0.09447326819274261,-2.814470586440865,-0.28104151998622556>,0.007018006566011825
    ,<0.1021825872458674,-2.8014075441081316,-0.286216418125924>,0.007205119848667835
    ,<0.1098554570521445,-2.7881231033829437,-0.2908509520703906>,0.007364433711532417
    ,<0.11716481470362139,-2.7744535295383033,-0.2949159513909721>,0.0075015263935279105
    ,<0.12378656790826784,-2.760282174207746,-0.29839674782495657>,0.007620622272343326
    ,<0.12941449698564675,-2.745561255573428,-0.30129547100313786>,0.007724966207910139
    ,<0.13378407914302196,-2.7303234153933755,-0.3036352473078953>,0.007817084460335388
    ,<0.1366850850562629,-2.7146691346070915,-0.30544378046519444>,0.007898968749670325
    ,<0.13797954752156716,-2.698752227690667,-0.3067493825217678>,0.007972207813666372
    ,<0.1376417700844447,-2.6827546414478034,-0.30757282171985517>,0.008038082702723609
    ,<0.13577484156449396,-2.666846210882433,-0.3079348364414149>,0.008097636716798745
    ,<0.1325910938237652,-2.6511449075866387,-0.3078869643459599>,0.008151727381894005
    ,<0.1283908821253184,-2.6356897182396075,-0.3075104214574524>,0.008201065543276747
    ,<0.123533424532774,-2.6204354071980056,-0.30691753160584845>,0.008246245102718756
    ,<0.11840684969665942,-2.605272377973181,-0.30625489431985015>,0.00828776588047385
    ,<0.11342747076056696,-2.590056484421257,-0.30567711666728303>,0.008326051367736582
    ,<0.10901946581917328,-2.5746575673686403,-0.3053522785479724>,0.00836146264109268
    ,<0.10556907292910465,-2.559012510277909,-0.3054800935227477>,0.008394309364827233
    ,<0.10339255056300452,-2.54316040169221,-0.30627230016497353>,0.008424858562469344
    ,<0.10275963538885592,-2.5272369414396163,-0.30787284891290956>,0.00845334166411343
    ,<0.10384709203195835,-2.511434927766294,-0.3102228228948966>,0.008479960209706025
    ,<0.10667018854030762,-2.4959444791533234,-0.313126504307252>,0.008504890496255251
    ,<0.11107094880314158,-2.4809025994393155,-0.31639791478113893>,0.008528287388947346
    ,<0.11674356192323498,-2.4663465196719905,-0.3198956746071197>,0.008550287465601714
    ,<0.12329253522169195,-2.4521911366058964,-0.3235025206175242>,0.008571011625971648
    ,<0.13028079810241283,-2.438248595686072,-0.3271107573551439>,0.00859056726871202
    ,<0.13725805051186699,-2.424277843958732,-0.33062965455831855>,0.008609050116966811
    ,<0.14377352800406987,-2.410046717461518,-0.3339860741203505>,0.008626545756733304
    ,<0.14939288835162556,-2.3953898161935565,-0.3371242546648926>,0.008643130939168025
    ,<0.15372872000030235,-2.3802535904991022,-0.340013541962313>,0.00865887468788217
    ,<0.15647973423949918,-2.364705087877128,-0.34265045599671484>,0.008673839244344611
    ,<0.15747074515609216,-2.3489089048193503,-0.34506220310487695>,0.008688080878257348
    ,<0.15667963106184624,-2.333077773871848,-0.34731670067722975>,0.008701650584808223
    ,<0.1542329851951258,-2.3174076367210756,-0.34948846908120146>,0.008714594686749191
    ,<0.15035710881001185,-2.302019758870654,-0.35157988100747256>,0.008726955356075762
    ,<0.1453589031998321,-2.286942663920791,-0.35354224795648476>,0.008738771067525925
    ,<0.13961130246623984,-2.272114672203856,-0.35534399573523034>,0.008750076994045604
    ,<0.1335280695599838,-2.2574025951454164,-0.3569813799503001>,0.008760905352682195
    ,<0.12754722489923345,-2.2426341082670618,-0.35848074277179987>,0.008771285707989934
    ,<0.12212361306062906,-2.227646340693636,-0.3599029854367242>,0.008781245238899917
    ,<0.11773315777877956,-2.21232433043317,-0.36130967719805956>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
