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
    ,<0.05486790444196308,-2.9443559148977094,-0.22673187992675492>,0.0
    ,<0.05683156003247692,-2.929616405475126,-0.23261623776339097>,0.001444405933878283
    ,<0.06061450039017542,-2.915239814319929,-0.2385085569589506>,0.002733688514425582
    ,<0.06595245903889421,-2.9013846678442907,-0.24451397398992275>,0.0037941133653625076
    ,<0.07247739387626637,-2.8880963017679266,-0.2506715199614117>,0.0046307451971068355
    ,<0.07984074898207444,-2.8752941553175977,-0.25690750179580174>,0.005283185474353696
    ,<0.08767511865934167,-2.8627803686742754,-0.2631513942612549>,0.005794598874521764
    ,<0.09559345249248319,-2.8502729446878963,-0.2692983979069944>,0.00620058003411749
    ,<0.10321240348080589,-2.837485625982804,-0.2752423604114802>,0.006527801879788091
    ,<0.11014953982173808,-2.824182117003582,-0.2808771571834936>,0.006795619711330263
    ,<0.11603372592436871,-2.810224469894563,-0.2861097903162856>,0.007018006566011825
    ,<0.12053625439598488,-2.7955995630228343,-0.29086501775143764>,0.007205119848667835
    ,<0.12340609399618145,-2.7804111796051942,-0.29508443134673973>,0.007364433711532417
    ,<0.12450642828861652,-2.7648486787943223,-0.2987272721105386>,0.0075015263935279105
    ,<0.12383912578588495,-2.749135508394458,-0.30177226684707875>,0.007620622272343326
    ,<0.12154810882645595,-2.7334724787157936,-0.3042198987928666>,0.007724966207910139
    ,<0.11790237272834388,-2.7179892243073067,-0.3060942224464743>,0.007817084460335388
    ,<0.1132745667073741,-2.702714376123292,-0.307438497170653>,0.007898968749670325
    ,<0.1081284765217851,-2.6875690093727314,-0.3082757784579804>,0.007972207813666372
    ,<0.10298812006989871,-2.6724025297958134,-0.3086198990952085>,0.008038082702723609
    ,<0.09839269826908366,-2.6570581317878195,-0.30851644386758476>,0.008097636716798745
    ,<0.09487902730148666,-2.641436723159095,-0.30804008776965086>,0.008151727381894005
    ,<0.09294700721534256,-2.6255512886695795,-0.3072965062147697>,0.008201065543276747
    ,<0.09300722817514658,-2.609554659729018,-0.30642394049361765>,0.008246245102718756
    ,<0.09531768266935772,-2.5937227121460222,-0.3055977387990486>,0.00828776588047385
    ,<0.09994685082172115,-2.578396706024095,-0.30500424766334794>,0.008326051367736582
    ,<0.10676921584364037,-2.563900674594905,-0.30478028628464116>,0.00836146264109268
    ,<0.11547554723147137,-2.5504542375929966,-0.30502560399232886>,0.008394309364827233
    ,<0.12561331747773502,-2.5380804296324273,-0.3057549372909375>,0.008424858562469344
    ,<0.13667853499412683,-2.526565501037185,-0.3068547101182236>,0.00845334166411343
    ,<0.14817363243484175,-2.5155045770695295,-0.3081921768399392>,0.008479960209706025
    ,<0.1596037033373201,-2.5043926492987656,-0.30967733190569574>,0.008504890496255251
    ,<0.17045688186353466,-2.4927288260273786,-0.3112603396273082>,0.008528287388947346
    ,<0.1801900200480731,-2.480125888844336,-0.31293297170404827>,0.008550287465601714
    ,<0.1882496968230611,-2.4664073906393584,-0.31473353833942846>,0.008571011625971648
    ,<0.19414971246336846,-2.4516575894808232,-0.31674770485876413>,0.00859056726871202
    ,<0.1975585601378249,-2.436191223660346,-0.31909657640703654>,0.008609050116966811
    ,<0.19824954859225644,-2.420442635959446,-0.32187632089488905>,0.008626545756733304
    ,<0.1961898508246673,-2.404895381979831,-0.32507996452347543>,0.008643130939168025
    ,<0.19167038865159353,-2.3899490615413184,-0.32860242397813727>,0.00865887468788217
    ,<0.18521421972263252,-2.3757775150184006,-0.33230013496327093>,0.008673839244344611
    ,<0.17741205428079934,-2.3623159237792306,-0.3360505627334253>,0.008688080878257348
    ,<0.16885112688924767,-2.349312586117867,-0.33975828822793275>,0.008701650584808223
    ,<0.16008943727049976,-2.3364133465808834,-0.34335670038699295>,0.008714594686749191
    ,<0.15165924695837024,-2.3232567151150962,-0.3468100384919977>,0.008726955356075762
    ,<0.14407420141165703,-2.309559189991781,-0.3501167193649021>,0.008738771067525925
    ,<0.13783773605336308,-2.295166623201666,-0.35328617143130253>,0.008750076994045604
    ,<0.13343068586457127,-2.280083668197105,-0.3563122209450343>,0.008760905352682195
    ,<0.13125193540155555,-2.2644958054334663,-0.3591965298957371>,0.008771285707989934
    ,<0.13154015638701685,-2.248740337978559,-0.3619735660622771>,0.008781245238899917
    ,<0.13433337487869998,-2.2332254473247617,-0.3647094284657834>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
