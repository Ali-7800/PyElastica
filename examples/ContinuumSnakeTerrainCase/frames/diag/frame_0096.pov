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
    ,<0.18639926133023035,-2.5756506814386593,-0.3156402193197978>,0.0
    ,<0.18266996703263486,-2.5601298357517974,-0.31672426065082987>,0.001444405933878283
    ,<0.17811769058121496,-2.5448243062109532,-0.31772681136496483>,0.002733688514425582
    ,<0.17332388063804352,-2.529584489304141,-0.3185964690053835>,0.0037941133653625076
    ,<0.16890798335557483,-2.5142227662868097,-0.3193114303711877>,0.0046307451971068355
    ,<0.16550284517217978,-2.498599960671243,-0.31988316869560024>,0.005283185474353696
    ,<0.16372026480275875,-2.482706835296996,-0.3203608260967145>,0.005794598874521764
    ,<0.16409364784985572,-2.466718356979333,-0.3208360121674868>,0.00620058003411749
    ,<0.16700030919944273,-2.450996498473746,-0.32144688895447465>,0.006527801879788091
    ,<0.172587372922713,-2.4360323433216045,-0.32237967958448155>,0.006795619711330263
    ,<0.1807315922566295,-2.4223373878632244,-0.32386139008216525>,0.007018006566011825
    ,<0.19107141822987958,-2.4103194530942247,-0.32605101484749743>,0.007205119848667835
    ,<0.2030902906058401,-2.400173245845161,-0.3290159730161263>,0.007364433711532417
    ,<0.2162575091228664,-2.391832476276989,-0.3326638625202375>,0.0075015263935279105
    ,<0.23013307035405753,-2.3849919221335396,-0.3367746951205122>,0.007620622272343326
    ,<0.24437936795926035,-2.3791401077820784,-0.3411388107039014>,0.007724966207910139
    ,<0.2587413775450058,-2.373605857650237,-0.345540340739378>,0.007817084460335388
    ,<0.2729702490746867,-2.36762921089524,-0.34979039237441795>,0.007898968749670325
    ,<0.2867291999397459,-2.3604636772821643,-0.35373383234054273>,0.007972207813666372
    ,<0.29949795994606937,-2.3514759191965524,-0.3572468345943239>,0.008038082702723609
    ,<0.31053755284262297,-2.3402826379553656,-0.3602403971603378>,0.008097636716798745
    ,<0.3189568768225347,-2.3268915508920767,-0.36266399772664676>,0.008151727381894005
    ,<0.3238920416023748,-2.311783279653757,-0.3645150514230541>,0.008201065543276747
    ,<0.32475413666501524,-2.295861860730637,-0.3658481713246929>,0.008246245102718756
    ,<0.32142781655536307,-2.2802394921725146,-0.3667726936064381>,0.00828776588047385
    ,<0.31431459194581135,-2.2659250467660623,-0.3674408339256838>,0.008326051367736582
    ,<0.30419321511440844,-2.2535502354745476,-0.36802883411683573>,0.00836146264109268
    ,<0.29197652041171696,-2.243246075718451,-0.368725620484321>,0.008394309364827233
    ,<0.27850167804214115,-2.2346628090951755,-0.36964622150769777>,0.008424858562469344
    ,<0.2644528462430022,-2.2270715414474154,-0.3707806589971469>,0.00845334166411343
    ,<0.2504144769881999,-2.219489194144745,-0.37207897857559535>,0.008479960209706025
    ,<0.23695667253183927,-2.2109391287469506,-0.3734834192406814>,0.008504890496255251
    ,<0.2247603557541058,-2.2006845247847826,-0.3749595846579036>,0.008528287388947346
    ,<0.21464089197014705,-2.1883877362754744,-0.3764944538812595>,0.008550287465601714
    ,<0.2074333159408282,-2.174196321287341,-0.3780685760744415>,0.008571011625971648
    ,<0.20379096936133975,-2.1587062159744472,-0.37962957404435344>,0.00859056726871202
    ,<0.20395879122019253,-2.142792199639192,-0.3811131412487872>,0.008609050116966811
    ,<0.20767888703815368,-2.1273093691244864,-0.3824604613069779>,0.008626545756733304
    ,<0.21429805454128711,-2.1128136126625803,-0.38362804146385465>,0.008643130939168025
    ,<0.22296762079387844,-2.0994285284395686,-0.3846064365788003>,0.00865887468788217
    ,<0.2328316960749236,-2.0868862884320167,-0.3854228953288241>,0.008673839244344611
    ,<0.2431155727627777,-2.074679313666039,-0.3861413457078434>,0.008688080878257348
    ,<0.25312218053879143,-2.062244334099327,-0.38686694770357>,0.008701650584808223
    ,<0.2621914901691914,-2.0491188039937986,-0.38774175434434405>,0.008714594686749191
    ,<0.26970121552796217,-2.035064848255281,-0.388959281364643>,0.008726955356075762
    ,<0.2750729028673923,-2.0200961453889192,-0.3906401665619995>,0.008738771067525925
    ,<0.277871808652115,-2.004475641568715,-0.3926810052203939>,0.008750076994045604
    ,<0.2778715466625027,-1.9886344315081954,-0.39493063361781555>,0.008760905352682195
    ,<0.2751245915110576,-1.9730481491618401,-0.3972806071096186>,0.008771285707989934
    ,<0.2699398309171021,-1.9580995165308794,-0.39965963959555>,0.008781245238899917
    ,<0.2628166582149401,-1.9439709397505136,-0.40203432891831264>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
