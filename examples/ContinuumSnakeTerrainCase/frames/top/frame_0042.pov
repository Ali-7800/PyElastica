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
    ,<0.04912289309706133,-2.898002144352971,-0.2436273353860178>,0.0
    ,<0.06144447322851924,-2.8898117198839097,-0.24971665696916764>,0.001444405933878283
    ,<0.07291132910122393,-2.8804262347097014,-0.25575027084116936>,0.002733688514425582
    ,<0.08313315923581137,-2.8696610802220976,-0.26171691535055036>,0.0037941133653625076
    ,<0.09167870519988472,-2.857488873952018,-0.2676105040589157>,0.0046307451971068355
    ,<0.09814763718941077,-2.8440332397097223,-0.2733517163645825>,0.005283185474353696
    ,<0.10224888237590993,-2.8295694139229792,-0.27881007209623226>,0.005794598874521764
    ,<0.10385216279389681,-2.814485243380363,-0.28387637161548646>,0.00620058003411749
    ,<0.10303427403238129,-2.7991891433995555,-0.28846941247297925>,0.006527801879788091
    ,<0.10007584686532565,-2.784010480013493,-0.2925384511805938>,0.006795619711330263
    ,<0.0954142958209934,-2.769125226068648,-0.296058672534268>,0.007018006566011825
    ,<0.08958367727654243,-2.754534505148598,-0.299027558351694>,0.007205119848667835
    ,<0.08316689329301187,-2.7400909972707135,-0.3014595098122768>,0.007364433711532417
    ,<0.0767696733213658,-2.72556237376316,-0.3033843274202647>,0.0075015263935279105
    ,<0.07101142054873351,-2.710715671364802,-0.30485238799584996>,0.007620622272343326
    ,<0.06651861247805632,-2.6954043276497357,-0.3059278663666039>,0.007724966207910139
    ,<0.06390227039052118,-2.6796426712893675,-0.3066711752667822>,0.007817084460335388
    ,<0.06369776931029525,-2.6636548274602942,-0.3071400781435596>,0.007898968749670325
    ,<0.06627975539951385,-2.6478690626723522,-0.30738321219388937>,0.007972207813666372
    ,<0.07178280088764019,-2.632847339708609,-0.30743746420798035>,0.008038082702723609
    ,<0.08005443507809575,-2.6191529826879405,-0.30736534472845484>,0.008097636716798745
    ,<0.09068139448381708,-2.6071929312223348,-0.30729680457052555>,0.008151727381894005
    ,<0.10309412137794761,-2.5970964501842038,-0.3073673478184879>,0.008201065543276747
    ,<0.11669086633888932,-2.5886622043631697,-0.3076422039220687>,0.008246245102718756
    ,<0.13093587165461973,-2.5813876316665008,-0.30814602354785536>,0.00828776588047385
    ,<0.1453910006512809,-2.5745582735766996,-0.30883465396271553>,0.008326051367736582
    ,<0.1596588103911779,-2.567353483479579,-0.30959759904678896>,0.00836146264109268
    ,<0.17327074681253896,-2.5589730362067704,-0.31033970751632384>,0.008394309364827233
    ,<0.1855939101147656,-2.548786758546073,-0.31100936663990625>,0.008424858562469344
    ,<0.19582442990723142,-2.5364964883353713,-0.31160204999359375>,0.00845334166411343
    ,<0.20310727773371667,-2.522258809925701,-0.3121595628456803>,0.008479960209706025
    ,<0.20675805795417512,-2.506691003037319,-0.31276568456545156>,0.008504890496255251
    ,<0.20648680996342433,-2.4907107424605526,-0.31353665681256004>,0.008528287388947346
    ,<0.20250029847712792,-2.4752519765497936,-0.3146079944528034>,0.008550287465601714
    ,<0.19543333276657512,-2.460981646409526,-0.3161314539161856>,0.008571011625971648
    ,<0.18611721788383884,-2.4481430137846814,-0.3181694110791912>,0.00859056726871202
    ,<0.17537536836782022,-2.4365620530606282,-0.3207021808763327>,0.008609050116966811
    ,<0.1639525766472703,-2.4257418999925062,-0.32362668149150153>,0.008626545756733304
    ,<0.15252050509986656,-2.4149934785137606,-0.3267693240137186>,0.008643130939168025
    ,<0.14169463056149129,-2.403662062325483,-0.3300064842412716>,0.00865887468788217
    ,<0.13209933054358555,-2.3912734281308694,-0.333249966535581>,0.008673839244344611
    ,<0.12436081170783858,-2.377636432356529,-0.33644513344837074>,0.008688080878257348
    ,<0.1190246812349251,-2.3628779441848926,-0.3395722618847219>,0.008701650584808223
    ,<0.11644797767089032,-2.347384110777562,-0.3426331662793998>,0.008714594686749191
    ,<0.11671515384443809,-2.3316717072480033,-0.3456517508474814>,0.008726955356075762
    ,<0.11961546970089686,-2.3162275309833076,-0.3486756730380594>,0.008738771067525925
    ,<0.12472324572465755,-2.301369782438471,-0.3517131470296876>,0.008750076994045604
    ,<0.13147921321258288,-2.2871821054223775,-0.35473106213905187>,0.008760905352682195
    ,<0.1392495424454084,-2.2735151226506685,-0.35770640985718954>,0.008771285707989934
    ,<0.14737349324778914,-2.260045967429306,-0.36063504183217127>,0.008781245238899917
    ,<0.15516482233028733,-2.2463779498861687,-0.3635458575067701>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
