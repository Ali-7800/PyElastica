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
    ,<0.01373348036132094,-3.1874171920786973,-0.20374966892318697>,0.0
    ,<0.013744218591770031,-3.1714109802113803,-0.20373306239796024>,0.001444405933878283
    ,<0.013769543760579322,-3.155397506238235,-0.20369996367858512>,0.002733688514425582
    ,<0.013799491071536054,-3.139380789063767,-0.2036628820305957>,0.0037941133653625076
    ,<0.013830853003270702,-3.1233620169503182,-0.2036189021286904>,0.0046307451971068355
    ,<0.013866236646674161,-3.1073448558853323,-0.2035733378004472>,0.005283185474353696
    ,<0.013904411468862283,-3.0913293084542715,-0.20352232289156102>,0.005794598874521764
    ,<0.013941965447597415,-3.075312246063896,-0.2034616837089706>,0.00620058003411749
    ,<0.013982312848814887,-3.059292952928565,-0.20339647694827756>,0.006527801879788091
    ,<0.014026649743753958,-3.0432725016202977,-0.20332407993356294>,0.006795619711330263
    ,<0.014071048998676484,-3.0272516849330158,-0.20324685525755476>,0.007018006566011825
    ,<0.014116740480982668,-3.0112309670973487,-0.20316848149907996>,0.007205119848667835
    ,<0.014163005158887515,-2.9952107577994838,-0.20308512595507436>,0.007364433711532417
    ,<0.014208724240928313,-2.9791920641535707,-0.20300109467823527>,0.0075015263935279105
    ,<0.014255216346063007,-2.9631755930517354,-0.20291677209076714>,0.007620622272343326
    ,<0.014302212214370027,-2.9471607939257187,-0.2028301738650205>,0.007724966207910139
    ,<0.01434934667769897,-2.9311474376528137,-0.20274341846922753>,0.007817084460335388
    ,<0.014395595640032989,-2.915136076768988,-0.20265606169169342>,0.007898968749670325
    ,<0.014440349022639215,-2.8991257703158855,-0.2025677311739225>,0.007972207813666372
    ,<0.014483289268997245,-2.8831167791557766,-0.20247972290748978>,0.008038082702723609
    ,<0.014526306108290083,-2.867109004936902,-0.2023900067799437>,0.008097636716798745
    ,<0.014567801270079977,-2.851102047284085,-0.2022974169777523>,0.008151727381894005
    ,<0.014607332767082726,-2.8350978500169157,-0.20220408514885446>,0.008201065543276747
    ,<0.014646028284161946,-2.8190953902270133,-0.20211169073115645>,0.008246245102718756
    ,<0.014684911743657216,-2.803091704561041,-0.2020197618763118>,0.00828776588047385
    ,<0.014725708769231443,-2.7870892666400513,-0.20193023360522983>,0.008326051367736582
    ,<0.014765851245023071,-2.771088436853602,-0.20184199929397265>,0.00836146264109268
    ,<0.01480317249681135,-2.7550852777786234,-0.20175039143046625>,0.008394309364827233
    ,<0.01483952820092716,-2.7390806905772775,-0.20166027229516245>,0.008424858562469344
    ,<0.014878374952654533,-2.7230780867692896,-0.2015757296727876>,0.00845334166411343
    ,<0.014917542507204182,-2.707074531573613,-0.20149780170304307>,0.008479960209706025
    ,<0.014954377410287194,-2.6910696910121024,-0.20142476133442794>,0.008504890496255251
    ,<0.014989642983582324,-2.6750642337445267,-0.20135263413432886>,0.008528287388947346
    ,<0.015021837360329941,-2.6590574920193704,-0.20128403883137416>,0.008550287465601714
    ,<0.015051443256576072,-2.643051127274657,-0.20121886876816825>,0.008571011625971648
    ,<0.015081333462183637,-2.6270455641935877,-0.20115280301890012>,0.00859056726871202
    ,<0.015110989619079343,-2.6110416579897704,-0.2010892900603641>,0.008609050116966811
    ,<0.015139740461295549,-2.5950389712014865,-0.20102769149185315>,0.008626545756733304
    ,<0.01516823423048919,-2.579037259535964,-0.20096765451942236>,0.008643130939168025
    ,<0.015194844891026463,-2.563037000315598,-0.20091213846680825>,0.00865887468788217
    ,<0.015220812005112116,-2.547036935607278,-0.20086089318444564>,0.008673839244344611
    ,<0.015246008761680443,-2.5310373617193775,-0.200813569325338>,0.008688080878257348
    ,<0.015268425899608024,-2.515039230078861,-0.2007705078584297>,0.008701650584808223
    ,<0.015288281115379332,-2.4990413716861477,-0.20073266023780564>,0.008714594686749191
    ,<0.01530464182702014,-2.4830413689161523,-0.2006999953302335>,0.008726955356075762
    ,<0.015318904821543636,-2.4670400884835626,-0.20067227337257215>,0.008738771067525925
    ,<0.015331928707933913,-2.4510400597486983,-0.20064735888130386>,0.008750076994045604
    ,<0.015342055187097805,-2.435039768165147,-0.20062890503662764>,0.008760905352682195
    ,<0.015349744222855712,-2.4190399287803483,-0.20061650949351653>,0.008771285707989934
    ,<0.01535469641316449,-2.403040700031876,-0.2006074785395517>,0.008781245238899917
    ,<0.01535612495892764,-2.387040754682751,-0.20060444160378454>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
