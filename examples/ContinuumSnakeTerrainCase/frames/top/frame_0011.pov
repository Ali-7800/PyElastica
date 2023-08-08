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
    ,<0.07511224493567036,-3.0945067882488257,-0.24597850825449977>,0.0
    ,<0.07606840352371996,-3.078531507989042,-0.2459696080219711>,0.001444405933878283
    ,<0.07642667812678192,-3.0625244499023645,-0.24591729239893706>,0.002733688514425582
    ,<0.07638104017333798,-3.046506138892792,-0.24579657208841396>,0.0037941133653625076
    ,<0.07607997843391517,-3.0304829396017916,-0.24559522930445848>,0.0046307451971068355
    ,<0.07562475046424753,-3.0144558847275507,-0.2453124937647454>,0.005283185474353696
    ,<0.07507995480302421,-2.9984324945265812,-0.2449659876445185>,0.005794598874521764
    ,<0.0744810438387152,-2.982411712321966,-0.2446061506506469>,0.00620058003411749
    ,<0.07385167549742826,-2.966386571534561,-0.24430644069222918>,0.006527801879788091
    ,<0.07323025813681204,-2.9503528809707076,-0.24413933527230433>,0.006795619711330263
    ,<0.07267980194938596,-2.9343144215117527,-0.2442304406786173>,0.007018006566011825
    ,<0.07228892974604373,-2.9182892614722324,-0.244882530755416>,0.007205119848667835
    ,<0.07211232578908118,-2.9023074099544015,-0.2462675800572705>,0.007364433711532417
    ,<0.0721445624649261,-2.8863785347141824,-0.24821771268937717>,0.0075015263935279105
    ,<0.07237456366627625,-2.870498715149813,-0.2505273056356062>,0.007620622272343326
    ,<0.07278395677659423,-2.8546585809714515,-0.25304669914913>,0.007724966207910139
    ,<0.0733483894055935,-2.8388367812444883,-0.25566322368939876>,0.007817084460335388
    ,<0.07403872384923363,-2.8230154758819492,-0.25829795790742993>,0.007898968749670325
    ,<0.07482218615506703,-2.8071896136511856,-0.26090022748186126>,0.007972207813666372
    ,<0.07566384962300551,-2.79134830348931,-0.2634422030537348>,0.008038082702723609
    ,<0.0765259947323257,-2.7754795169156408,-0.26590948606121584>,0.008097636716798745
    ,<0.07736706287337368,-2.7595877032396703,-0.26829342170493714>,0.008151727381894005
    ,<0.07814489052652814,-2.7436853001953114,-0.27058941248044655>,0.008201065543276747
    ,<0.07882025331053519,-2.7277688227938883,-0.2727972640156992>,0.008246245102718756
    ,<0.0793565297952147,-2.7118328517115353,-0.2749174156540091>,0.00828776588047385
    ,<0.0797229354974456,-2.6958824904173517,-0.2769496278870348>,0.008326051367736582
    ,<0.07990026035685378,-2.679930501031109,-0.2788939040456269>,0.00836146264109268
    ,<0.07988263724561581,-2.6639735666720203,-0.2807523078415005>,0.008394309364827233
    ,<0.0796766022758031,-2.648001158520952,-0.2825268977293762>,0.008424858562469344
    ,<0.07930330236829312,-2.6320240855194292,-0.28421725430791234>,0.00845334166411343
    ,<0.07879805455599212,-2.616041906881616,-0.2858241367814894>,0.008479960209706025
    ,<0.07820756859571498,-2.6000606568840574,-0.2873475630624442>,0.008504890496255251
    ,<0.07758537444557811,-2.584077065467526,-0.2887882026597464>,0.008528287388947346
    ,<0.07698816435534293,-2.568086693803268,-0.2901467524455881>,0.008550287465601714
    ,<0.07647158175799022,-2.552097252434034,-0.29142327864316053>,0.008571011625971648
    ,<0.07608250748580214,-2.536102776046466,-0.29261883610132106>,0.00859056726871202
    ,<0.07585544027474117,-2.5201069177289455,-0.2937337880665051>,0.008609050116966811
    ,<0.07581051222999216,-2.5041061743477973,-0.2947692200756372>,0.008626545756733304
    ,<0.07595221412885653,-2.488105870442327,-0.2957257057950865>,0.008643130939168025
    ,<0.07626966479781891,-2.47210714085145,-0.29660373581834604>,0.00865887468788217
    ,<0.07673963641557792,-2.4560974343958892,-0.2974040689318897>,0.008673839244344611
    ,<0.07732775714910438,-2.4400902208304203,-0.29812645551938477>,0.008688080878257348
    ,<0.07799403004484833,-2.4240871335664655,-0.2987714666928676>,0.008701650584808223
    ,<0.07869618671425117,-2.4080822386521152,-0.29933979120607823>,0.008714594686749191
    ,<0.07938910740400795,-2.3920756813301045,-0.2998316432200384>,0.008726955356075762
    ,<0.08002521150741468,-2.37606904776233,-0.3002474471845935>,0.008738771067525925
    ,<0.08055158106567298,-2.360065048994851,-0.3005885523556436>,0.008750076994045604
    ,<0.08089523993626101,-2.3440590137181445,-0.30085755926965696>,0.008760905352682195
    ,<0.08094598069761166,-2.328054943285555,-0.3010594479649032>,0.008771285707989934
    ,<0.08054245639169284,-2.3120609237779757,-0.3012047276649261>,0.008781245238899917
    ,<0.07946593667670834,-2.296096047073739,-0.3013131352716375>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
