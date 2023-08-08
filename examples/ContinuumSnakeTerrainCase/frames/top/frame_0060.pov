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
    ,<0.11749205820439443,-2.7777153408162607,-0.3030985910933965>,0.0
    ,<0.1072098910404024,-2.7654756507999854,-0.30378199099146397>,0.001444405933878283
    ,<0.09764833283638749,-2.75266519438366,-0.30447543762796037>,0.002733688514425582
    ,<0.08936764390594087,-2.7389930410947456,-0.3052063329460165>,0.0037941133653625076
    ,<0.082956450871446,-2.7243516582936653,-0.30597196615333383>,0.0046307451971068355
    ,<0.0789723468695621,-2.708872052735797,-0.3067385853246114>,0.005283185474353696
    ,<0.0778541898158606,-2.69292494685824,-0.3074677847380399>,0.005794598874521764
    ,<0.07982867252731544,-2.6770574916166274,-0.3081317295487179>,0.00620058003411749
    ,<0.08485050508571697,-2.6618725609915197,-0.3087161345886258>,0.006527801879788091
    ,<0.09260811281754407,-2.6478815881055935,-0.3092309033311284>,0.006795619711330263
    ,<0.10259463178264247,-2.6353827099724434,-0.3097178624606319>,0.007018006566011825
    ,<0.11421976323578711,-2.624394381610692,-0.31025002251262634>,0.007205119848667835
    ,<0.12691054984170635,-2.6146632690133034,-0.3109151386457062>,0.007364433711532417
    ,<0.1401610300062475,-2.605724761676688,-0.31177879244687867>,0.0075015263935279105
    ,<0.15352857499702594,-2.596983388970138,-0.3128394117075613>,0.007620622272343326
    ,<0.16658592729004887,-2.587805095461948,-0.31405437195245656>,0.007724966207910139
    ,<0.17884392366246685,-2.5775980555937794,-0.31537724424318464>,0.007817084460335388
    ,<0.18970015803868268,-2.565919715111531,-0.3167586654781096>,0.007898968749670325
    ,<0.19843074289310986,-2.552582909024445,-0.31814920809661096>,0.007972207813666372
    ,<0.20424310044108027,-2.5377406788029258,-0.31949797893589893>,0.008038082702723609
    ,<0.2064636940658863,-2.5219527431573057,-0.32077216792111946>,0.008097636716798745
    ,<0.20476487085590045,-2.506095284432022,-0.3219562456343478>,0.008151727381894005
    ,<0.19929216857324147,-2.491110640892149,-0.3230568298965115>,0.008201065543276747
    ,<0.19062800030812255,-2.4777118251294006,-0.32410263755618784>,0.008246245102718756
    ,<0.17960715377415504,-2.4661712347247216,-0.3251387382475288>,0.00828776588047385
    ,<0.16709447075898665,-2.456269907384563,-0.3262114606364551>,0.008326051367736582
    ,<0.1538438591452964,-2.447386954314062,-0.3273477000718455>,0.00836146264109268
    ,<0.14048776128723897,-2.4386699348953593,-0.32854547010346197>,0.008394309364827233
    ,<0.12764003821151657,-2.429227690748317,-0.32980535063967753>,0.008424858562469344
    ,<0.11601992718527451,-2.4183216896245,-0.3311611247242576>,0.00845334166411343
    ,<0.10649392188019154,-2.405564339448829,-0.3326898929968133>,0.008479960209706025
    ,<0.09997710719226674,-2.3910649071617893,-0.3344724778791547>,0.008504890496255251
    ,<0.09721824269853664,-2.3754431877932713,-0.3365291090863729>,0.008528287388947346
    ,<0.09854942145018061,-2.3596720653913446,-0.33884660800353816>,0.008550287465601714
    ,<0.10375406577215279,-2.3447649887826403,-0.3414089327730899>,0.008571011625971648
    ,<0.11215661666777645,-2.331438041854275,-0.3441776110760076>,0.00859056726871202
    ,<0.12286140677445401,-2.3199188910371094,-0.3471088775274567>,0.008609050116966811
    ,<0.13499261128371798,-2.3099490472989905,-0.3501616330050088>,0.008626545756733304
    ,<0.14782106506816725,-2.300924417786357,-0.3533035115680175>,0.008643130939168025
    ,<0.16075721991981862,-2.292073486909036,-0.35650003223926247>,0.00865887468788217
    ,<0.173264627747237,-2.2826268226870488,-0.3596940468466321>,0.008673839244344611
    ,<0.18476990510851415,-2.2719618066918774,-0.36281637044249226>,0.008688080878257348
    ,<0.19463625148498134,-2.259727723080035,-0.3657942213438296>,0.008701650584808223
    ,<0.202227601333708,-2.2459235476567043,-0.36856805909002616>,0.008714594686749191
    ,<0.2070391581695746,-2.2308804396880486,-0.3710988543816777>,0.008726955356075762
    ,<0.20882875952631855,-2.2151499869581297,-0.37337337036635815>,0.008738771067525925
    ,<0.207682616866198,-2.1993267100454923,-0.37540701861523046>,0.008750076994045604
    ,<0.2039876435116387,-2.183872457644134,-0.3772374205363676>,0.008760905352682195
    ,<0.198337612316188,-2.169002178102191,-0.3789096577847532>,0.008771285707989934
    ,<0.19143620388866123,-2.154657672331526,-0.38047974764827996>,0.008781245238899917
    ,<0.18403397294969975,-2.1405602123750724,-0.38202941278260943>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
