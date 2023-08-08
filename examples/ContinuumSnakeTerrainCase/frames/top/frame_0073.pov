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
    ,<0.11026320465199295,-2.682658927286444,-0.3112869568735706>,0.0
    ,<0.11491314902457513,-2.6673581029986875,-0.31174076137341344>,0.001444405933878283
    ,<0.1216800748772167,-2.6528686182704004,-0.3122022751862817>,0.002733688514425582
    ,<0.13011011213849002,-2.6392800543612793,-0.3127049034089419>,0.0037941133653625076
    ,<0.13967907316619546,-2.626471181117843,-0.31330765837622904>,0.0046307451971068355
    ,<0.14985316463209303,-2.6141451922321655,-0.3140638852684058>,0.005283185474353696
    ,<0.16011160936432597,-2.6019021890633574,-0.31501242816227004>,0.005794598874521764
    ,<0.16993670561500993,-2.5893269701421957,-0.3161832468963041>,0.00620058003411749
    ,<0.17877826745460737,-2.57606421995965,-0.317552374401382>,0.006527801879788091
    ,<0.18604075972872278,-2.5618905921724533,-0.3190483331420637>,0.006795619711330263
    ,<0.19114140905004276,-2.5468085232940916,-0.32060168089802427>,0.007018006566011825
    ,<0.19359607925059055,-2.5310777064845205,-0.32215247381169376>,0.007205119848667835
    ,<0.19311150351778086,-2.5151582923246534,-0.32364710006177116>,0.007364433711532417
    ,<0.18966493107234572,-2.4996007957064172,-0.32504641343793195>,0.0075015263935279105
    ,<0.1835175360715649,-2.484890361706519,-0.3263352545668303>,0.007620622272343326
    ,<0.1751551801961689,-2.4713080733184127,-0.3275310495601359>,0.007724966207910139
    ,<0.16518756661862233,-2.4588538243274285,-0.3286950240220462>,0.007817084460335388
    ,<0.15425227956695733,-2.4472403503958255,-0.3298815822983978>,0.007898968749670325
    ,<0.14296642833567552,-2.435966601737009,-0.33108455299674977>,0.007972207813666372
    ,<0.1319480455119069,-2.424433406055035,-0.33230221342858357>,0.008038082702723609
    ,<0.12187698141941575,-2.4120710235918237,-0.33358054553209154>,0.008097636716798745
    ,<0.11353297854401907,-2.39849627576208,-0.3350018634282024>,0.008151727381894005
    ,<0.10776692723858525,-2.383662033940488,-0.33663024061930347>,0.008201065543276747
    ,<0.10537688284164674,-2.3679506647137076,-0.3384867823508394>,0.008246245102718756
    ,<0.10690481784339063,-2.352160339359806,-0.34057409812418066>,0.00828776588047385
    ,<0.11243965390777291,-2.3373266561925226,-0.3428837659920826>,0.008326051367736582
    ,<0.12155241113466113,-2.324419673959155,-0.34540708470561005>,0.00836146264109268
    ,<0.13342579694439988,-2.3140452456445226,-0.34812552376686345>,0.008394309364827233
    ,<0.1471134837928304,-2.3062824474436,-0.35102119217640637>,0.008424858562469344
    ,<0.16179466123652064,-2.3007000856124353,-0.35407296908810076>,0.00845334166411343
    ,<0.17690263485136673,-2.2965011581116577,-0.3572635291757888>,0.008479960209706025
    ,<0.1920920191593517,-2.292691739125174,-0.3605632879682475>,0.008504890496255251
    ,<0.20709207861677076,-2.2882300531126787,-0.3639130154491833>,0.008528287388947346
    ,<0.22152270274295918,-2.2821600427039677,-0.3672312819589817>,0.008550287465601714
    ,<0.23476911719099253,-2.2737744828887902,-0.37043389535122445>,0.008571011625971648
    ,<0.24600491082632517,-2.26278961935213,-0.37344417283944203>,0.00859056726871202
    ,<0.2543737320279814,-2.249434830286891,-0.37618420832187316>,0.008609050116966811
    ,<0.2592520521452592,-2.2343944737944073,-0.3785987440615966>,0.008626545756733304
    ,<0.2604575835405626,-2.2185791213812758,-0.3806548572022271>,0.008643130939168025
    ,<0.25828864148577163,-2.2028233514811943,-0.3823404263255386>,0.00865887468788217
    ,<0.2534045122100065,-2.1876518422680937,-0.38366669655509766>,0.008673839244344611
    ,<0.24663740010697013,-2.1731952609729377,-0.3846680737599>,0.008688080878257348
    ,<0.2388414250762129,-2.159249815808178,-0.3854125289544112>,0.008701650584808223
    ,<0.23082523343253586,-2.1454203192307513,-0.38597974030613136>,0.008714594686749191
    ,<0.22335025395708186,-2.1312869533964443,-0.3864505784817364>,0.008726955356075762
    ,<0.21714380114166168,-2.1165521690510625,-0.3869402000141586>,0.008738771067525925
    ,<0.21288593691927085,-2.101143748837446,-0.3875546530143726>,0.008750076994045604
    ,<0.21115413399537977,-2.0852572437052777,-0.38831865360944184>,0.008760905352682195
    ,<0.21233338689068762,-2.0693253453883083,-0.3891892681432857>,0.008771285707989934
    ,<0.21653600573366738,-2.053915414526587,-0.3901075796150349>,0.008781245238899917
    ,<0.22357534401858092,-2.0395778771389748,-0.3910354292820733>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
