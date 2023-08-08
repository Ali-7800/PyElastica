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
    ,<0.058052298515972274,-2.8714879593487876,-0.26165427666669266>,0.0
    ,<0.07129306247707806,-2.8640959603159932,-0.2667553187510779>,0.001444405933878283
    ,<0.08307484039258275,-2.8545344076262507,-0.2718263536520229>,0.002733688514425582
    ,<0.09277243912022903,-2.842844164288611,-0.2768446532572126>,0.0037941133653625076
    ,<0.09983200272084082,-2.8293481201375137,-0.28172856818836073>,0.0046307451971068355
    ,<0.10390795061979617,-2.814587731285799,-0.2863487489777678>,0.005283185474353696
    ,<0.104938944533825,-2.799197936210566,-0.2905816921490829>,0.005794598874521764
    ,<0.10316500527403848,-2.7837501035212284,-0.2943269126270822>,0.00620058003411749
    ,<0.09907192130608078,-2.7686244919156637,-0.2975303813544759>,0.006527801879788091
    ,<0.09329504266822158,-2.753947748898054,-0.3001796437591067>,0.006795619711330263
    ,<0.0865333786867936,-2.739608952763964,-0.30229491495665>,0.007018006566011825
    ,<0.07950131171625711,-2.7253361837743446,-0.3039187147896017>,0.007205119848667835
    ,<0.07291858049321148,-2.7108087860594363,-0.30511305925620497>,0.007364433711532417
    ,<0.06751496977667942,-2.695778926880136,-0.30595437658993635>,0.0075015263935279105
    ,<0.0640165385093009,-2.6801827114984818,-0.30651754360456124>,0.007620622272343326
    ,<0.0630825561190754,-2.664220455472436,-0.3068729290922412>,0.007724966207910139
    ,<0.06519595617294843,-2.6483686478200776,-0.3070802080105243>,0.007817084460335388
    ,<0.0705505517436597,-2.6332989452160445,-0.30718849046847396>,0.007898968749670325
    ,<0.07898706819670986,-2.619712286744007,-0.3072681803988332>,0.007972207813666372
    ,<0.09002538723495536,-2.608139977412602,-0.30743860700519166>,0.008038082702723609
    ,<0.10299294427220847,-2.598784001117973,-0.3078466117269291>,0.008097636716798745
    ,<0.1171925723942925,-2.5914501605234834,-0.3085807376403506>,0.008151727381894005
    ,<0.13204307319536765,-2.585591883317324,-0.3096407058061316>,0.008201065543276747
    ,<0.14712994479449756,-2.580422346126309,-0.3109159104480682>,0.008246245102718756
    ,<0.16213272747389018,-2.57502576189156,-0.3122408796336498>,0.00828776588047385
    ,<0.17667312918146577,-2.5684695595194205,-0.31348099407965757>,0.008326051367736582
    ,<0.19017318160834734,-2.5599515551992065,-0.3145331129322355>,0.00836146264109268
    ,<0.20179278626737657,-2.5489874876671945,-0.3153414609337683>,0.008394309364827233
    ,<0.21052242721171296,-2.5355960892242337,-0.31589872223157095>,0.008424858562469344
    ,<0.21543593703760863,-2.520378987644616,-0.31624867396120004>,0.00845334166411343
    ,<0.21600194825096655,-2.504396131009608,-0.31648563605987223>,0.008479960209706025
    ,<0.2122841962532459,-2.4888406799687792,-0.316748240808998>,0.008504890496255251
    ,<0.2049098687267444,-2.4746524771932763,-0.3172084110991098>,0.008528287388947346
    ,<0.1948428169178445,-2.462255400068015,-0.3180702299552023>,0.008550287465601714
    ,<0.18306351949670374,-2.4515318587199233,-0.3194450059646253>,0.008571011625971648
    ,<0.17040679190447366,-2.441939467125403,-0.32135025950321444>,0.00859056726871202
    ,<0.15756371032718078,-2.432691259741267,-0.3237209739794436>,0.008609050116966811
    ,<0.14514075411362662,-2.42296215773226,-0.3263947545445456>,0.008626545756733304
    ,<0.13375537221795494,-2.412081189021917,-0.3292390848651497>,0.008643130939168025
    ,<0.12408551711220447,-2.399670971266906,-0.33216382285064316>,0.00865887468788217
    ,<0.116802182038614,-2.385732301655813,-0.33511382537684553>,0.008673839244344611
    ,<0.11242490615425565,-2.370627857807719,-0.338064987113153>,0.008688080878257348
    ,<0.1111793950452848,-2.354951533133368,-0.3410188814440113>,0.008701650584808223
    ,<0.11294167940525705,-2.33932620486994,-0.3439791148812015>,0.008714594686749191
    ,<0.11728479041165614,-2.3242163436044887,-0.34695325091916945>,0.008726955356075762
    ,<0.12358565402958874,-2.309818020454075,-0.3499582512871234>,0.008738771067525925
    ,<0.13114650227173627,-2.296040702240594,-0.3529726655799063>,0.008750076994045604
    ,<0.1392647989790787,-2.2825774179227536,-0.35595479308918215>,0.008760905352682195
    ,<0.14724027514827442,-2.269018322412266,-0.3588857132692438>,0.008771285707989934
    ,<0.15435083731417323,-2.2549760162949406,-0.3617631405618946>,0.008781245238899917
    ,<0.15983205201072737,-2.240214799434527,-0.364603340891073>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
