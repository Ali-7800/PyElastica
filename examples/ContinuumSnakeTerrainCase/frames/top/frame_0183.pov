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
    ,<0.29868681655419643,-2.0464952603707296,-0.3600916839555094>,0.0
    ,<0.29634214635576567,-2.0321248245796255,-0.36671692280674434>,0.001444405933878283
    ,<0.29634414028068407,-2.0175598220307744,-0.37334291787010904>,0.002733688514425582
    ,<0.2990350454916655,-2.003199570259493,-0.3798862902140548>,0.0037941133653625076
    ,<0.3045201650642056,-1.9895574300965446,-0.38621538641045533>,0.0046307451971068355
    ,<0.3126519980822576,-1.9771580909921984,-0.3922449071482331>,0.005283185474353696
    ,<0.3230592269187639,-1.9664055911333285,-0.3979258915927027>,0.005794598874521764
    ,<0.3352336424194214,-1.9574799947248502,-0.4032446655865516>,0.00620058003411749
    ,<0.3486415944409817,-1.9502944941206788,-0.4082189714929982>,0.006527801879788091
    ,<0.36281439987307607,-1.9445150694114546,-0.4128929890933773>,0.006795619711330263
    ,<0.3773892885569818,-1.9396090478344825,-0.4173199317681786>,0.007018006566011825
    ,<0.3920912665231725,-1.9349186648695849,-0.42155412385209173>,0.007205119848667835
    ,<0.406664489935054,-1.9297405001913115,-0.42566323190339284>,0.007364433711532417
    ,<0.42078857007918763,-1.9233952850930696,-0.42970406173354164>,0.0075015263935279105
    ,<0.43400328262227106,-1.9152957252620275,-0.4336900377587695>,0.007620622272343326
    ,<0.445662571559306,-1.905036413122366,-0.4375527365491758>,0.007724966207910139
    ,<0.4549553234212026,-1.8925231404040026,-0.4411705212736135>,0.007817084460335388
    ,<0.4610564925206727,-1.8780966607796012,-0.4444239682720265>,0.007898968749670325
    ,<0.463335567117302,-1.8625098731096021,-0.4471994253889306>,0.007972207813666372
    ,<0.46155512430387396,-1.846770325801836,-0.44940710289712127>,0.008038082702723609
    ,<0.4559599978322382,-1.831874221482722,-0.45099933467713293>,0.008097636716798745
    ,<0.4472025516176415,-1.8185299842395997,-0.45197929078428783>,0.008151727381894005
    ,<0.4361434203264717,-1.8069866934966625,-0.4524011899145225>,0.008201065543276747
    ,<0.42363725771584343,-1.797019241496879,-0.4523693116374084>,0.008246245102718756
    ,<0.4104093071180735,-1.7880361028427159,-0.45202564505975085>,0.00828776588047385
    ,<0.39706070707963437,-1.7792392199365266,-0.4515183510815459>,0.008326051367736582
    ,<0.3841632365396301,-1.7697945405627036,-0.45097828316136196>,0.00836146264109268
    ,<0.3723591290595541,-1.7590094579695938,-0.4505448445990993>,0.008394309364827233
    ,<0.36239945539531054,-1.7464944477045943,-0.45035931485998604>,0.008424858562469344
    ,<0.35507027499773663,-1.7322772941235272,-0.4505195208203735>,0.00845334166411343
    ,<0.3510066101805397,-1.7168158922360734,-0.45106554987666325>,0.008479960209706025
    ,<0.3504830669429081,-1.7008542695231013,-0.4519632256756929>,0.008504890496255251
    ,<0.353309604407689,-1.6851548175062783,-0.45314067766903693>,0.008528287388947346
    ,<0.3588969383890089,-1.6702312615186317,-0.4545285638658131>,0.008550287465601714
    ,<0.3664278269714629,-1.656201514770145,-0.4560561635305615>,0.008571011625971648
    ,<0.37502161767381426,-1.6428024564901793,-0.4576453105731201>,0.00859056726871202
    ,<0.38381536051112564,-1.6295291664048568,-0.45919902977836546>,0.008609050116966811
    ,<0.3919635043804973,-1.6158358459325237,-0.46061918070476887>,0.008626545756733304
    ,<0.3986148586421475,-1.6013370592180025,-0.4618228151964226>,0.008643130939168025
    ,<0.40294175145557365,-1.5859640290325003,-0.4627402200505115>,0.00865887468788217
    ,<0.40425170976485214,-1.5700316893798458,-0.46331917470790884>,0.008673839244344611
    ,<0.4021413490481532,-1.5541760797839699,-0.4635317801434065>,0.008688080878257348
    ,<0.39660557238854205,-1.5391676424506917,-0.46338166546811865>,0.008701650584808223
    ,<0.3880289915335076,-1.5256706397264146,-0.46290822893004174>,0.008714594686749191
    ,<0.3770563635470427,-1.5140488114855946,-0.4621865850873216>,0.008726955356075762
    ,<0.36440970918624127,-1.5042845110209848,-0.46132545230357247>,0.008738771067525925
    ,<0.3507401359311096,-1.4960101003089914,-0.46046193629809196>,0.008750076994045604
    ,<0.3365728959140325,-1.4886074432788308,-0.4597165452141539>,0.008760905352682195
    ,<0.32232665110222875,-1.4813484487076658,-0.4591133169737735>,0.008771285707989934
    ,<0.3084061072573265,-1.4734780123394822,-0.45859787207748626>,0.008781245238899917
    ,<0.2953049683342652,-1.4643064112877613,-0.45811692051045433>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
