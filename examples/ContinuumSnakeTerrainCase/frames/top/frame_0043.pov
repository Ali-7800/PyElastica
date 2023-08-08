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
    ,<0.051141467396438364,-2.8862918351911913,-0.2506940776662469>,0.0
    ,<0.06419396563519579,-2.8790428210180954,-0.25644545264964763>,0.001444405933878283
    ,<0.0762309214299765,-2.8701960522452445,-0.2621742940050588>,0.002733688514425582
    ,<0.08677837628287706,-2.859588831760174,-0.2678466162248774>,0.0037941133653625076
    ,<0.0953266770075904,-2.8472649924363145,-0.27340826219830816>,0.0046307451971068355
    ,<0.10143013504022956,-2.8334847466484296,-0.27876283981390204>,0.005283185474353696
    ,<0.10480441816413978,-2.8186826165237906,-0.2837892082544483>,0.005794598874521764
    ,<0.10539611706149224,-2.8033769407317015,-0.28838121633505537>,0.00620058003411749
    ,<0.10340964131964295,-2.7880449816406516,-0.29246140209412735>,0.006527801879788091
    ,<0.09926585456806263,-2.773009498374938,-0.2959839656393973>,0.006795619711330263
    ,<0.09352720725294163,-2.7583809715184575,-0.2989352378656613>,0.007018006566011825
    ,<0.08682554015016201,-2.744063672805377,-0.3013311918709944>,0.007205119848667835
    ,<0.07981753386469681,-2.729815789913935,-0.3032104941727996>,0.007364433711532417
    ,<0.07317149406515085,-2.7153429814500827,-0.30463343532334214>,0.0075015263935279105
    ,<0.06756909937065336,-2.700402952127106,-0.30567301118182144>,0.007620622272343326
    ,<0.06369577343405654,-2.6849057660630478,-0.3063943198653367>,0.007724966207910139
    ,<0.062188097487151886,-2.6689934391914174,-0.3068650720345074>,0.007817084460335388
    ,<0.06354126684333532,-2.6530622267261994,-0.30714715651253305>,0.007898968749670325
    ,<0.0680078899669119,-2.6377078970281143,-0.3072843360659137>,0.007972207813666372
    ,<0.07552148537703854,-2.623591346915954,-0.30733193202479836>,0.008038082702723609
    ,<0.0856988602041782,-2.611255686025241,-0.30738102523929706>,0.008097636716798745
    ,<0.09793898415313426,-2.600962887814027,-0.3075752051104542>,0.008151727381894005
    ,<0.11157450929565182,-2.5926117041165857,-0.30803197281988115>,0.008201065543276747
    ,<0.1260136592537654,-2.585759027604124,-0.30875920647478516>,0.008246245102718756
    ,<0.14079835003183794,-2.5797073608430514,-0.3096726273266766>,0.00828776588047385
    ,<0.15556113957873816,-2.5736103585373913,-0.3106469151024234>,0.008326051367736582
    ,<0.16990956820221678,-2.5665866748363606,-0.31157246517187004>,0.00836146264109268
    ,<0.18329714198526748,-2.557859500397218,-0.31237977210406803>,0.008394309364827233
    ,<0.1949607559535637,-2.5469265727057864,-0.3130428095308954>,0.008424858562469344
    ,<0.2039836801399424,-2.533725521497994,-0.3135790768748076>,0.00845334166411343
    ,<0.20950103336892956,-2.5187166890505672,-0.31404819502999537>,0.008479960209706025
    ,<0.21097181104591944,-2.5027956971249163,-0.31454583204087583>,0.008504890496255251
    ,<0.20837371204599836,-2.4870252124772505,-0.31519231528409797>,0.008528287388947346
    ,<0.20220743023253077,-2.4722945916918513,-0.3161195601500117>,0.008550287465601714
    ,<0.19332295237134642,-2.4590653923392147,-0.31747011759448696>,0.008571011625971648
    ,<0.18263128192085645,-2.4473227952724033,-0.3193215372289395>,0.00859056726871202
    ,<0.17094245193014715,-2.4366626494936097,-0.3216919703557348>,0.008609050116966811
    ,<0.15894836113705607,-2.426438286605856,-0.324484369475495>,0.008626545756733304
    ,<0.14728173769824357,-2.4159045978115627,-0.32750548314880057>,0.008643130939168025
    ,<0.13655618087421287,-2.4044405963909163,-0.33062405332389644>,0.00865887468788217
    ,<0.12741277258033357,-2.39168225989571,-0.3337505163382993>,0.008673839244344611
    ,<0.12047141479553201,-2.377595859184761,-0.33683404724376426>,0.008688080878257348
    ,<0.11621773171644623,-2.3624685825330927,-0.3398598963532798>,0.008701650584808223
    ,<0.11489000115700053,-2.346802231707519,-0.34283541716531735>,0.008714594686749191
    ,<0.11642254789965413,-2.3311511403370546,-0.3457878110600968>,0.008726955356075762
    ,<0.12046924479604684,-2.315958942244584,-0.3487611900324043>,0.008738771067525925
    ,<0.12648539977274542,-2.301438007609782,-0.3517554626819558>,0.008750076994045604
    ,<0.1338214442872838,-2.2875337968526237,-0.3547310499976763>,0.008760905352682195
    ,<0.14180010961377842,-2.273979206458521,-0.357662984248987>,0.008771285707989934
    ,<0.14973366368261576,-2.260388631141738,-0.3605482429995541>,0.008781245238899917
    ,<0.15690469933223694,-2.2463770027121126,-0.3634170458719664>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
