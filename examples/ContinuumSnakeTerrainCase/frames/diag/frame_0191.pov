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
    ,<0.2974418636651009,-2.0411559070426954,-0.3581789596060706>,0.0
    ,<0.3061368061676889,-2.029405714755718,-0.3646706719455833>,0.001444405933878283
    ,<0.31653665192063546,-2.0191390996836303,-0.3711821548941701>,0.002733688514425582
    ,<0.32821453721129107,-2.010299812549832,-0.37764201162469785>,0.0037941133653625076
    ,<0.3407639165868333,-2.0026170001993546,-0.3839431464467123>,0.0046307451971068355
    ,<0.35382911750830287,-1.9956429390118209,-0.3900138920927751>,0.005283185474353696
    ,<0.3670983205566918,-1.988822553918858,-0.39580891281762465>,0.005794598874521764
    ,<0.3802600578458911,-1.9815655427233978,-0.401309258417703>,0.00620058003411749
    ,<0.39294765668236104,-1.973321593685511,-0.40652413374896024>,0.006527801879788091
    ,<0.4046993301669277,-1.9636615081300066,-0.4114934707784395>,0.006795619711330263
    ,<0.4149587879365452,-1.952358607050358,-0.4162977045340183>,0.007018006566011825
    ,<0.4231338826349272,-1.9394202468892148,-0.42096826663284664>,0.007205119848667835
    ,<0.42869447235355,-1.9251051648433346,-0.42546366893966364>,0.007364433711532417
    ,<0.4312843617371761,-1.9098976567772412,-0.42971856339967457>,0.0075015263935279105
    ,<0.4308100918890669,-1.8943879365227934,-0.43362826036349067>,0.007620622272343326
    ,<0.42745919062531257,-1.8791296204177694,-0.4370890083424919>,0.007724966207910139
    ,<0.4216613927556762,-1.8645100133848658,-0.4400292307601334>,0.007817084460335388
    ,<0.41400197407607187,-1.8506680704383274,-0.4424161943058865>,0.007898968749670325
    ,<0.40513175076159286,-1.8374824911688785,-0.4442629193747943>,0.007972207813666372
    ,<0.395709931412238,-1.8246259087412744,-0.4456315193530392>,0.008038082702723609
    ,<0.3863872586377422,-1.8116637589031444,-0.44663930505763105>,0.008097636716798745
    ,<0.3778228779167203,-1.7981730947986447,-0.4474531438117283>,0.008151727381894005
    ,<0.37071666392994485,-1.7838543912541103,-0.44822189432318615>,0.008201065543276747
    ,<0.3658003729120461,-1.7686437376006432,-0.4490081309993455>,0.008246245102718756
    ,<0.3637237522859017,-1.752797019294357,-0.4498527736367783>,0.00828776588047385
    ,<0.3648904960885887,-1.736863497848555,-0.45080329273059566>,0.008326051367736582
    ,<0.36934306814074735,-1.7215278914587873,-0.4518636469100158>,0.00836146264109268
    ,<0.37674095758653303,-1.7073843781791496,-0.45302095277577387>,0.008394309364827233
    ,<0.386461931428369,-1.694734114899669,-0.4542700754627326>,0.008424858562469344
    ,<0.39776726064783224,-1.683488619034875,-0.455610647039867>,0.00845334166411343
    ,<0.40993775234796104,-1.6731998271382988,-0.4570544255116309>,0.008479960209706025
    ,<0.4223153134941872,-1.6631779789184922,-0.4586170119932354>,0.008504890496255251
    ,<0.43425184357211677,-1.6526494674412353,-0.4602819767136351>,0.008528287388947346
    ,<0.44502570292874516,-1.6409416146557234,-0.4620031803445311>,0.008550287465601714
    ,<0.4538063152660262,-1.6276726303930094,-0.46371907257220685>,0.008571011625971648
    ,<0.4597331848490713,-1.6128987923017195,-0.4653650622569142>,0.00859056726871202
    ,<0.46208952662714986,-1.5971412676363823,-0.4668542372953353>,0.008609050116966811
    ,<0.4604995734857828,-1.581265733764488,-0.4680582595264011>,0.008626545756733304
    ,<0.45506909064940426,-1.5662381274556236,-0.4688556222026227>,0.008643130939168025
    ,<0.44635431286651217,-1.5528269231608323,-0.46916797311265257>,0.00865887468788217
    ,<0.43517162852682034,-1.5413906006163078,-0.46897176155555664>,0.008673839244344611
    ,<0.4223645056940213,-1.5318301299801191,-0.4682950722588202>,0.008688080878257348
    ,<0.4086433235250168,-1.5236789191432367,-0.4672088184442127>,0.008701650584808223
    ,<0.3945434722178786,-1.516251841957318,-0.4658177348523195>,0.008714594686749191
    ,<0.38047797111307285,-1.5087926059172783,-0.4642557413056748>,0.008726955356075762
    ,<0.3668307791980312,-1.5005919804441656,-0.4626812289241293>,0.008738771067525925
    ,<0.3540661414458425,-1.4910533183666397,-0.4612304786345187>,0.008750076994045604
    ,<0.34277506717166684,-1.4797885795313424,-0.4599576376440124>,0.008760905352682195
    ,<0.3335752867968343,-1.466748349965541,-0.4588244770131295>,0.008771285707989934
    ,<0.32699851541101926,-1.4522027177403982,-0.4577572429756195>,0.008781245238899917
    ,<0.3233731614799326,-1.4366553606270942,-0.45670254119242404>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
