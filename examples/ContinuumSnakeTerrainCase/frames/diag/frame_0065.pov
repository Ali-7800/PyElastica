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
    ,<0.11452822719487665,-2.751686301354724,-0.30423884270793905>,0.0
    ,<0.10466842711491106,-2.739131454274268,-0.305327334377344>,0.001444405933878283
    ,<0.09716551439903169,-2.7250387992017004,-0.30640723448965124>,0.002733688514425582
    ,<0.09259923229936913,-2.7097363502717164,-0.30744141144301185>,0.0037941133653625076
    ,<0.09133703816141721,-2.6938109722969488,-0.30838258109729993>,0.0046307451971068355
    ,<0.09344368585152385,-2.6779682380177285,-0.3092008125723622>,0.005283185474353696
    ,<0.09866661991520162,-2.6628572773864536,-0.30989523766939825>,0.005794598874521764
    ,<0.10650503706148756,-2.648918217647923,-0.31049624236658624>,0.00620058003411749
    ,<0.11632859553122014,-2.6362977663282217,-0.31106403768486546>,0.006527801879788091
    ,<0.12749225782548732,-2.624847934627392,-0.31168054587121297>,0.006795619711330263
    ,<0.1394038601336018,-2.6141856257396574,-0.3124275747395372>,0.007018006566011825
    ,<0.151533033561559,-2.603785439454342,-0.3133569615299663>,0.007205119848667835
    ,<0.1633780620606599,-2.5930812232013625,-0.3144762177370459>,0.007364433711532417
    ,<0.17441632467809426,-2.5815668688159388,-0.31578073372564097>,0.0075015263935279105
    ,<0.18406400578342555,-2.568889914796557,-0.3172762433192901>,0.007620622272343326
    ,<0.1916460428111775,-2.5549001546400136,-0.3189130416541605>,0.007724966207910139
    ,<0.1964735695203407,-2.5397451164019365,-0.3205964821709029>,0.007817084460335388
    ,<0.1980166866845463,-2.523913341714267,-0.32224791388571355>,0.007898968749670325
    ,<0.1960502184188383,-2.5081204300077973,-0.3238049035279472>,0.007972207813666372
    ,<0.19073351959441595,-2.493107375334933,-0.32522778091026205>,0.008038082702723609
    ,<0.1825740036269918,-2.479415606534256,-0.32650518200783535>,0.008097636716798745
    ,<0.17228838167321414,-2.4672261071571944,-0.32765547768187303>,0.008151727381894005
    ,<0.16064037677006437,-2.456319255828304,-0.32872198157850024>,0.008201065543276747
    ,<0.1483398372842288,-2.446146224284866,-0.32973445004500296>,0.008246245102718756
    ,<0.13604043473940422,-2.4359695343502787,-0.33071900925163394>,0.00828776588047385
    ,<0.12441477785623462,-2.4250339368553595,-0.3317454518747517>,0.008326051367736582
    ,<0.11423641409246928,-2.4127526795158123,-0.3329175828352105>,0.00836146264109268
    ,<0.10638621275520019,-2.3988883556501635,-0.3343408532479861>,0.008394309364827233
    ,<0.10174326072336388,-2.383676067463282,-0.3360627634362322>,0.008424858562469344
    ,<0.10097759059941562,-2.3678224854215735,-0.3380641277835652>,0.00845334166411343
    ,<0.10431504934290982,-2.3523402937873974,-0.3403162815833668>,0.008479960209706025
    ,<0.11143459757180268,-2.338230577257101,-0.34279096055574476>,0.008504890496255251
    ,<0.12158295167950776,-2.3261557865634686,-0.34545585927584765>,0.008528287388947346
    ,<0.13382917918512066,-2.31626025031696,-0.3482858220179357>,0.008550287465601714
    ,<0.14731266110311741,-2.3081845471469506,-0.3512684000292672>,0.008571011625971648
    ,<0.16136917595737185,-2.3012061995505366,-0.35437612415049907>,0.00859056726871202
    ,<0.1755068939178206,-2.294427622174778,-0.35756178930616017>,0.008609050116966811
    ,<0.18928419875868172,-2.286947676241184,-0.36075843290126797>,0.008626545756733304
    ,<0.2021815903229964,-2.278009351765762,-0.3638746360820404>,0.008643130939168025
    ,<0.2135492462990465,-2.2671501993130945,-0.366834494216066>,0.00865887468788217
    ,<0.22267472269771277,-2.25430367994413,-0.3695828301957845>,0.008673839244344611
    ,<0.22894546877401475,-2.23980214750205,-0.37207559248667427>,0.008688080878257348
    ,<0.23202368731341078,-2.224264682612343,-0.37429422483092134>,0.008701650584808223
    ,<0.23194948023482953,-2.2083885615623786,-0.3762334314276041>,0.008714594686749191
    ,<0.2291171828621905,-2.1927335200656284,-0.37788962521131436>,0.008726955356075762
    ,<0.22416065070248256,-2.1775870345769786,-0.37926145692479657>,0.008738771067525925
    ,<0.217826098688442,-2.1629392403927437,-0.38035103082471783>,0.008750076994045604
    ,<0.21088837358367576,-2.1485501013497044,-0.3811852331406488>,0.008760905352682195
    ,<0.20412681411029124,-2.1340674033508633,-0.38182236268213143>,0.008771285707989934
    ,<0.19833516866587228,-2.1191655537127465,-0.38235222510752265>,0.008781245238899917
    ,<0.19432764802075728,-2.1036856282656187,-0.3828599984218076>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
