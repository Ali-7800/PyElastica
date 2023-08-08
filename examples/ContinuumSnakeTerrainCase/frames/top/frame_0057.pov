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
    ,<0.11114268057894036,-2.8019255576830164,-0.29866923386162375>,0.0
    ,<0.10432320258944756,-2.7875060346768206,-0.29991609369778477>,0.001444405933878283
    ,<0.09697858960656995,-2.773344128553516,-0.3011335852548135>,0.002733688514425582
    ,<0.08964381321277484,-2.759173096967687,-0.3023100332940951>,0.0037941133653625076
    ,<0.08288514620907335,-2.744714418951409,-0.3034354667731478>,0.0046307451971068355
    ,<0.0772864425540884,-2.729763907464717,-0.30450192415375144>,0.005283185474353696
    ,<0.07342468810442261,-2.7142683580765867,-0.30549749704374335>,0.005794598874521764
    ,<0.07182075597223854,-2.6983742553824897,-0.3064069718862457>,0.00620058003411749
    ,<0.07286385515573574,-2.6824277613406267,-0.3072115906320694>,0.006527801879788091
    ,<0.07673366944437161,-2.6669178144537065,-0.30789716705303644>,0.006795619711330263
    ,<0.08335540556943298,-2.652363766100232,-0.30846710148781725>,0.007018006566011825
    ,<0.09241408209735821,-2.6391841573785566,-0.3089569569762257>,0.007205119848667835
    ,<0.10342682276223573,-2.6275869010006088,-0.30944054054768755>,0.007364433711532417
    ,<0.1158443687781201,-2.617513410414122,-0.3100302918839004>,0.0075015263935279105
    ,<0.12913874165436554,-2.6086482034953793,-0.3108603818105245>,0.007620622272343326
    ,<0.14285303596416596,-2.6004793389031104,-0.3119622792179133>,0.007724966207910139
    ,<0.15659174950090374,-2.592376954457083,-0.3132456822237638>,0.007817084460335388
    ,<0.16995420209263123,-2.583679823999803,-0.3146213675041514>,0.007898968749670325
    ,<0.1824702766632833,-2.5738113354796903,-0.3160434264242016>,0.007972207813666372
    ,<0.19349308120042283,-2.5623017388490563,-0.31744302425011245>,0.008038082702723609
    ,<0.2021961945809693,-2.5489412179941695,-0.31872383197071824>,0.008097636716798745
    ,<0.20774484025899334,-2.533979861895081,-0.31983739580717824>,0.008151727381894005
    ,<0.2094945979988526,-2.5181086471557212,-0.32076939567543444>,0.008201065543276747
    ,<0.2071968663736495,-2.5022993342300683,-0.3215419466710294>,0.008246245102718756
    ,<0.20110047529508218,-2.487528586698413,-0.3222234751449372>,0.00828776588047385
    ,<0.1918769571408869,-2.47448022085093,-0.3229339562326223>,0.008326051367736582
    ,<0.18041595993856838,-2.4633661669623192,-0.3238532928897572>,0.00836146264109268
    ,<0.16759232507957397,-2.4538904851249828,-0.32504053149094975>,0.008394309364827233
    ,<0.15414017246508682,-2.4453534340413614,-0.32641078857914924>,0.008424858562469344
    ,<0.14067188155194638,-2.4368589555104387,-0.3278937554817453>,0.00845334166411343
    ,<0.12779178368472852,-2.42750813993734,-0.32944546591115675>,0.008479960209706025
    ,<0.11622057925239919,-2.4165905621070807,-0.3310705768266862>,0.008504890496255251
    ,<0.10681752938032742,-2.4037743771739013,-0.33282370431291075>,0.008528287388947346
    ,<0.10045809652207557,-2.3892301083544973,-0.3347728571721825>,0.008550287465601714
    ,<0.0978094352332053,-2.373608760830293,-0.33695049930933507>,0.008571011625971648
    ,<0.09910395566360054,-2.357852862065113,-0.3393714501604847>,0.00859056726871202
    ,<0.10405504115518982,-2.342880315724654,-0.3420419194685521>,0.008609050116966811
    ,<0.11198040922721983,-2.329286547144519,-0.3449261515710385>,0.008626545756733304
    ,<0.12202708335726209,-2.3172116392317514,-0.3479675639301943>,0.008643130939168025
    ,<0.13336718681155887,-2.3063682518745776,-0.35109960885904373>,0.00865887468788217
    ,<0.14529077770037355,-2.2961799311256392,-0.35426302425583495>,0.008673839244344611
    ,<0.15718645030703676,-2.285952873413855,-0.3574080177717377>,0.008688080878257348
    ,<0.16847062452437506,-2.2750326212346077,-0.36047422598635576>,0.008701650584808223
    ,<0.1785292022135182,-2.2629406575574564,-0.3634030563752471>,0.008714594686749191
    ,<0.18672282153889838,-2.2494777813648303,-0.3661520212162433>,0.008726955356075762
    ,<0.19246664723970094,-2.2347659041387855,-0.36870040351851757>,0.008738771067525925
    ,<0.19534786006530713,-2.2192061842751594,-0.3710445440146469>,0.008750076994045604
    ,<0.19522811093527434,-2.2033570846567985,-0.3732066856960896>,0.008760905352682195
    ,<0.19227825575354882,-2.187766416507887,-0.37523289459826514>,0.008771285707989934
    ,<0.1869322962153879,-2.1728163094280935,-0.37717754530506115>,0.008781245238899917
    ,<0.17979873420499587,-2.1586270251868336,-0.3791039677071758>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
