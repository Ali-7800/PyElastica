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
    ,<0.23012838573284713,-2.3710930192168007,-0.3654851402522867>,0.0
    ,<0.2456881190848382,-2.367946185255075,-0.3674827626121696>,0.001444405933878283
    ,<0.261087115221096,-2.3640117423320577,-0.3693224994528811>,0.002733688514425582
    ,<0.2760625656927583,-2.358611910034703,-0.37091884425348765>,0.0037941133653625076
    ,<0.29015527290108023,-2.3511720226350277,-0.37230685305811173>,0.0046307451971068355
    ,<0.30274418503009104,-2.341370456558664,-0.37346389064141083>,0.005283185474353696
    ,<0.31314159283535725,-2.3292321675906176,-0.37421572680021487>,0.005794598874521764
    ,<0.3206942157240782,-2.315125623208218,-0.37442647756027214>,0.00620058003411749
    ,<0.3249298833238219,-2.2996964373058066,-0.3740617770901979>,0.006527801879788091
    ,<0.32567700803748806,-2.283733948983486,-0.37317132645492374>,0.006795619711330263
    ,<0.32309718013669547,-2.267991093705698,-0.37188424290312977>,0.007018006566011825
    ,<0.3176259953217914,-2.2530248227797474,-0.3703977426012134>,0.007205119848667835
    ,<0.30986610684330773,-2.239105055255878,-0.3689652641808266>,0.007364433711532417
    ,<0.30044653393451415,-2.226228650488061,-0.36779859806222515>,0.0075015263935279105
    ,<0.28998684817639736,-2.214149560587323,-0.3670168062754515>,0.007620622272343326
    ,<0.2790934428888341,-2.202427761723372,-0.3666937134518144>,0.007724966207910139
    ,<0.26837963982681634,-2.1905298240558393,-0.3667952865906356>,0.007817084460335388
    ,<0.25849028333610796,-2.177945527307687,-0.3672119391916933>,0.007898968749670325
    ,<0.25009174290663533,-2.164331450379616,-0.36784899041051283>,0.007972207813666372
    ,<0.24386450285119288,-2.1496024931555526,-0.36861002574571383>,0.008038082702723609
    ,<0.2404204299189132,-2.1339862304511534,-0.36939399204495565>,0.008097636716798745
    ,<0.2401709065656388,-2.117995317991995,-0.3700989199005403>,0.008151727381894005
    ,<0.24321079332339357,-2.1022900070423147,-0.37063304769051586>,0.008201065543276747
    ,<0.2492827127625766,-2.087487160003567,-0.3709308515494884>,0.008246245102718756
    ,<0.2578472353948936,-2.0739733593111165,-0.3709710712948788>,0.00828776588047385
    ,<0.2682315577637311,-2.061807211692372,-0.3707974793959612>,0.008326051367736582
    ,<0.279787876341971,-2.050760402651892,-0.37055009070581046>,0.00836146264109268
    ,<0.2918809140207551,-2.0403178557480004,-0.3704019281557455>,0.008394309364827233
    ,<0.30375385710919134,-2.0296302565321507,-0.37047336743401144>,0.008424858562469344
    ,<0.31452352641706516,-2.0178236598476564,-0.3709021273067964>,0.00845334166411343
    ,<0.323238549428523,-2.004431736692576,-0.37166681246997>,0.008479960209706025
    ,<0.3289093568697198,-1.9895073816540778,-0.3725921720967493>,0.008504890496255251
    ,<0.33069272121264226,-1.9736499015328457,-0.3735760608512571>,0.008528287388947346
    ,<0.32816854239581816,-1.9578942117124192,-0.37454886513319946>,0.008550287465601714
    ,<0.3215175052084373,-1.9433874464639256,-0.3754620004011729>,0.008571011625971648
    ,<0.31146604862642835,-1.9309879863532242,-0.3762824656939859>,0.00859056726871202
    ,<0.2990117843923134,-1.920997158934215,-0.3769918811018161>,0.008609050116966811
    ,<0.2851092371645145,-1.9131362440955326,-0.3775880657357817>,0.008626545756733304
    ,<0.27048778036111415,-1.9067017971286613,-0.37808841218844996>,0.008643130939168025
    ,<0.25565073583986814,-1.9007769615510774,-0.3785351990603163>,0.00865887468788217
    ,<0.24099521126427073,-1.8944181070342427,-0.37900277087795775>,0.008673839244344611
    ,<0.22695462515899087,-1.8868066370721281,-0.3796048151406882>,0.008688080878257348
    ,<0.21407807796782144,-1.877381624050918,-0.38049861785159234>,0.008701650584808223
    ,<0.20300156257409527,-1.8659408813897296,-0.38188071692204273>,0.008714594686749191
    ,<0.1943173065574614,-1.8526609714053082,-0.38389628575533663>,0.008726955356075762
    ,<0.18840772082681587,-1.838007722545726,-0.3864406843410165>,0.008738771067525925
    ,<0.18536160323143447,-1.8225570000925801,-0.3892864256011552>,0.008750076994045604
    ,<0.1849531757152646,-1.8068400941161034,-0.3922689217039618>,0.008760905352682195
    ,<0.18669913834478424,-1.7912206182921746,-0.39527527773109383>,0.008771285707989934
    ,<0.1899408288511082,-1.7758357352022909,-0.3982467653702795>,0.008781245238899917
    ,<0.1939146112936924,-1.7606160435889942,-0.401176044926828>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
