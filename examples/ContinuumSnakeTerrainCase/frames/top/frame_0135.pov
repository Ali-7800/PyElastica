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
    ,<0.3076411602934059,-2.225045631750135,-0.36878189038735715>,0.0
    ,<0.2998311818359028,-2.21110537836825,-0.36960631115709797>,0.001444405933878283
    ,<0.29106045296317634,-2.1977445786144907,-0.37043611684837563>,0.002733688514425582
    ,<0.28185153630202897,-2.1846739944734774,-0.3712086666726654>,0.0037941133653625076
    ,<0.2727481145917223,-2.1715204395186443,-0.3718121260481015>,0.0046307451971068355
    ,<0.2643273093927631,-2.157911644621603,-0.3721751344644894>,0.005283185474353696
    ,<0.25721164854886414,-2.143574056257005,-0.37225351392948325>,0.005794598874521764
    ,<0.2520536889294066,-2.128424194047906,-0.3720296323338666>,0.00620058003411749
    ,<0.24947962379118638,-2.1126364938259767,-0.3715151519954971>,0.006527801879788091
    ,<0.24999281492001804,-2.096659207189547,-0.3707564562788234>,0.006795619711330263
    ,<0.25386589496517853,-2.081158951177241,-0.3698404990885754>,0.007018006566011825
    ,<0.26106433980921706,-2.066897917815625,-0.3688987768494592>,0.007205119848667835
    ,<0.27123199552889066,-2.054566532377615,-0.3681019171117958>,0.007364433711532417
    ,<0.2837687118262871,-2.044634814680541,-0.3676562585007331>,0.0075015263935279105
    ,<0.2979332054254405,-2.037194823392892,-0.3677533001719614>,0.007620622272343326
    ,<0.3129952493870258,-2.0318491575019055,-0.36852952675656614>,0.007724966207910139
    ,<0.32840209894035044,-2.027783934160306,-0.37003656927955053>,0.007817084460335388
    ,<0.34379005503139065,-2.023965492571564,-0.37225896379379847>,0.007898968749670325
    ,<0.35884436246701656,-2.019350222550048,-0.3751564613171624>,0.007972207813666372
    ,<0.3731033933257007,-2.0129815458616394,-0.3786727185377765>,0.008038082702723609
    ,<0.3857865261233182,-2.00412017112421,-0.38275827902847714>,0.008097636716798745
    ,<0.39575265634778023,-1.9924772265292254,-0.3873378163959743>,0.008151727381894005
    ,<0.4016765785533178,-1.9784673825656405,-0.3922601891429354>,0.008201065543276747
    ,<0.4024592680785245,-1.963311392520007,-0.39726473769489057>,0.008246245102718756
    ,<0.39771599929313467,-1.9488080742577512,-0.401992125611305>,0.00828776588047385
    ,<0.3880361515034286,-1.9367689688402372,-0.4060530812678935>,0.008326051367736582
    ,<0.37478803856803006,-1.9283893390474622,-0.40912809341235185>,0.00836146264109268
    ,<0.3595670635065396,-1.923920429316025,-0.4110438405896262>,0.008394309364827233
    ,<0.34364246724087805,-1.9227711226367876,-0.4117904520512215>,0.008424858562469344
    ,<0.3276935656910619,-1.923859392224347,-0.4114894652981671>,0.00845334166411343
    ,<0.3118737807414284,-1.9259338972873492,-0.41033167630926176>,0.008479960209706025
    ,<0.2960800749282293,-1.9276913875550796,-0.4084539361046324>,0.008504890496255251
    ,<0.28027655445641586,-1.9278665330064941,-0.40595951319975954>,0.008528287388947346
    ,<0.2647514172654496,-1.925389012509963,-0.40301139864327473>,0.008550287465601714
    ,<0.25020896536428244,-1.919543614092103,-0.39984788943733296>,0.008571011625971648
    ,<0.23762749562409044,-1.910166513562116,-0.39680725352686497>,0.00859056726871202
    ,<0.22793511361936775,-1.897712563086782,-0.3943062518654474>,0.008609050116966811
    ,<0.2216680325204762,-1.883095615794791,-0.3927844241148255>,0.008626545756733304
    ,<0.21880845437732094,-1.867374665180694,-0.3925470091132417>,0.008643130939168025
    ,<0.2188134432754998,-1.8514228294111084,-0.39365636334048937>,0.00865887468788217
    ,<0.22079753602550695,-1.8357085055896598,-0.3959085466357892>,0.008673839244344611
    ,<0.22377313030602003,-1.820292362427831,-0.39898775129583725>,0.008688080878257348
    ,<0.2268002766214792,-1.805011389532524,-0.4026403259820727>,0.008701650584808223
    ,<0.22902324360422827,-1.7896867509679928,-0.40666886381899187>,0.008714594686749191
    ,<0.2296932720197602,-1.7742782618240216,-0.4109273585037327>,0.008726955356075762
    ,<0.22821660696059365,-1.7589620691185197,-0.41531141461472393>,0.008738771067525925
    ,<0.22422431955720984,-1.744118551041125,-0.41974882731224095>,0.008750076994045604
    ,<0.21762826224463963,-1.7302374791240858,-0.4241927496762572>,0.008760905352682195
    ,<0.20862398993907255,-1.7177766297877144,-0.4286173124439123>,0.008771285707989934
    ,<0.19762644268912585,-1.7070238111982432,-0.4330158315518966>,0.008781245238899917
    ,<0.18515965037938545,-1.6980081761624073,-0.4374041052624836>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
