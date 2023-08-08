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
    ,<0.11447070333583773,-2.7413436855385758,-0.3023587358165792>,0.0
    ,<0.10669619488462301,-2.727440260000457,-0.3038552692441318>,0.001444405933878283
    ,<0.10177169511685709,-2.7122903805444785,-0.3053542760442632>,0.002733688514425582
    ,<0.10006520941716646,-2.6964465250512544,-0.3067996003291394>,0.0037941133653625076
    ,<0.1016566415431876,-2.6805806861987285,-0.30812173967573164>,0.0046307451971068355
    ,<0.10631077432215096,-2.665317785884566,-0.30929006500135864>,0.005283185474353696
    ,<0.11354243010959467,-2.651082939851072,-0.3103119107642567>,0.005794598874521764
    ,<0.12273116056828387,-2.638017927829868,-0.3112319476103769>,0.00620058003411749
    ,<0.1332315146395598,-2.6259796507499376,-0.3121312751812617>,0.006527801879788091
    ,<0.14443785737138196,-2.6146001613489642,-0.3131017300687841>,0.006795619711330263
    ,<0.1557945786994264,-2.6033825476780486,-0.31421086738317106>,0.007018006566011825
    ,<0.1667675359974613,-2.5918067877090265,-0.315489454825549>,0.007205119848667835
    ,<0.17679787352035553,-2.579425316400608,-0.31694156825280717>,0.007364433711532417
    ,<0.18527638227530174,-2.5659555174874322,-0.31855824549695716>,0.0075015263935279105
    ,<0.19155085264689392,-2.551344898272149,-0.32028904956440735>,0.007620622272343326
    ,<0.19501726883486958,-2.5358314066913357,-0.3220493332991089>,0.007724966207910139
    ,<0.19526760648145017,-2.5199333462504505,-0.32375751064083946>,0.007817084460335388
    ,<0.1922041298249519,-2.5043194159787836,-0.32534103735553843>,0.007898968749670325
    ,<0.18607782407885987,-2.4896160173233626,-0.32674712005798967>,0.007972207813666372
    ,<0.1774275655639858,-2.476220271486718,-0.3279549091007317>,0.008038082702723609
    ,<0.16695126562620774,-2.4641876110944865,-0.32900838961953704>,0.008097636716798745
    ,<0.1553663332091337,-2.453208628621755,-0.3299586576957895>,0.008151727381894005
    ,<0.14334321961015123,-2.4426974564102233,-0.33081653241327214>,0.008201065543276747
    ,<0.13153011112907473,-2.43194594733443,-0.3316369342685783>,0.008246245102718756
    ,<0.12062831507762475,-2.4202755443812456,-0.3325165014103322>,0.00828776588047385
    ,<0.11144749377855162,-2.4072210857385334,-0.33357330399836044>,0.008326051367736582
    ,<0.10487514921951874,-2.3927012786924826,-0.3349361761544322>,0.00836146264109268
    ,<0.10173999212673303,-2.3771084203185078,-0.33665527956934876>,0.008394309364827233
    ,<0.10259592688045584,-2.361264549819505,-0.3386925428773455>,0.008424858562469344
    ,<0.10751304695157082,-2.346216742919317,-0.34099091346478966>,0.00845334166411343
    ,<0.11602563152970768,-2.3329085397855622,-0.34350616359198927>,0.008479960209706025
    ,<0.12729179691529033,-2.3218775468002915,-0.34620551725837817>,0.008504890496255251
    ,<0.14036880041631833,-2.3131186054556783,-0.3490645186268278>,0.008528287388947346
    ,<0.15444641384892235,-2.3061384724569023,-0.35206741638571387>,0.008550287465601714
    ,<0.168936749350555,-2.3001194590250384,-0.3551911043571606>,0.008571011625971648
    ,<0.18341181522082273,-2.2941038994306404,-0.3583944125016608>,0.00859056726871202
    ,<0.19745954010214106,-2.2871594736524137,-0.3616192797011666>,0.008609050116966811
    ,<0.21055185873984475,-2.2785277874224232,-0.3647839942467488>,0.008626545756733304
    ,<0.22200235637881086,-2.267773266569653,-0.3678009772381063>,0.008643130939168025
    ,<0.2310589625174431,-2.254890208450622,-0.37059820331385884>,0.00865887468788217
    ,<0.23709618895049606,-2.240297143373244,-0.37312149942367345>,0.008673839244344611
    ,<0.23980613479984111,-2.22469323260941,-0.3753420968905787>,0.008688080878257348
    ,<0.23929104414026067,-2.208824928146392,-0.37726447836967136>,0.008701650584808223
    ,<0.23601406069597514,-2.1932576306693994,-0.37890731059230054>,0.008714594686749191
    ,<0.23066433821862617,-2.1782477944610346,-0.3802800937684364>,0.008726955356075762
    ,<0.22402104568178843,-2.1637404671192515,-0.38138325196572437>,0.008738771067525925
    ,<0.21687227901899522,-2.149457766400467,-0.38223380099150134>,0.008750076994045604
    ,<0.20999571176182374,-2.1350313917901813,-0.3828841181290113>,0.008760905352682195
    ,<0.20416866252165577,-2.1201452054091448,-0.3834267904397386>,0.008771285707989934
    ,<0.20016846220027226,-2.1046651454014205,-0.3839615084426325>,0.008781245238899917
    ,<0.19872774287066283,-2.0887403816848096,-0.3845120571188315>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
