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
    ,<0.24239818091021006,-2.0851076038673577,-0.3441881764383694>,0.0
    ,<0.25542731699117643,-2.0774000468226745,-0.3493627401125568>,0.001444405933878283
    ,<0.26674596242364706,-2.067373026610305,-0.3545847860969552>,0.002733688514425582
    ,<0.2760076733683983,-2.055469012973032,-0.35991612934840805>,0.0037941133653625076
    ,<0.28312620940220407,-2.042250630111463,-0.36543070735858446>,0.0046307451971068355
    ,<0.28822685853841146,-2.028231578644111,-0.3712012918577089>,0.005283185474353696
    ,<0.29162579231894736,-2.0138188730551887,-0.377286131409058>,0.005794598874521764
    ,<0.2938401613943722,-1.9992525527858689,-0.3835857602865845>,0.00620058003411749
    ,<0.2954764173204926,-1.9845984118722217,-0.3898700172970262>,0.006527801879788091
    ,<0.29715037510029535,-1.9698708689396958,-0.3959670656279803>,0.006795619711330263
    ,<0.29944607825545555,-1.9551039039538456,-0.4017550036111977>,0.007018006566011825
    ,<0.30288260857774113,-1.9404137403805441,-0.4071574726414095>,0.007205119848667835
    ,<0.30786879974329595,-1.9260254004441015,-0.4121441993785072>,0.007364433711532417
    ,<0.314655827441497,-1.9122556099507442,-0.41673135158230185>,0.0075015263935279105
    ,<0.3233046242264132,-1.8994591118898745,-0.4209846694489947>,0.007620622272343326
    ,<0.3336688346501558,-1.887906224506351,-0.42494683797729993>,0.007724966207910139
    ,<0.3454511104662493,-1.8777001527198767,-0.42862876017459184>,0.007817084460335388
    ,<0.3582834594801329,-1.8687541378573858,-0.4320645046222039>,0.007898968749670325
    ,<0.3717879333046488,-1.860774214548033,-0.4352925575945706>,0.007972207813666372
    ,<0.38560865333289335,-1.8532833740384207,-0.4383458746730199>,0.008038082702723609
    ,<0.3993980877519312,-1.845675268654598,-0.4412446643063621>,0.008097636716798745
    ,<0.4127623086131315,-1.837291738836349,-0.443992314214646>,0.008151727381894005
    ,<0.4251885469623873,-1.827525971061927,-0.4465734015004041>,0.008201065543276747
    ,<0.4359978269736303,-1.8159526534210817,-0.448955309422889>,0.008246245102718756
    ,<0.4443698616644779,-1.80246945233585,-0.45109392351539385>,0.00828776588047385
    ,<0.4494693509806125,-1.7874020226727514,-0.4529438353873468>,0.008326051367736582
    ,<0.4506503460555887,-1.7715053482759229,-0.45446572826869003>,0.00836146264109268
    ,<0.44765534979211935,-1.7558201728808762,-0.45562206472175953>,0.008394309364827233
    ,<0.4407210442188742,-1.7414132825968474,-0.4563867716545407>,0.008424858562469344
    ,<0.43053250674148374,-1.72907612968311,-0.4567471782187055>,0.00845334166411343
    ,<0.41801460208816105,-1.7191064787717536,-0.45669754912607>,0.008479960209706025
    ,<0.4040668459710103,-1.7112743201035978,-0.45625385864026075>,0.008504890496255251
    ,<0.38939422913879457,-1.7049373050098526,-0.45545358596550173>,0.008528287388947346
    ,<0.3744884537625915,-1.6992199385888545,-0.45434530467018375>,0.008550287465601714
    ,<0.35972534781414517,-1.6931933418244047,-0.45298740818030064>,0.008571011625971648
    ,<0.34549536031678474,-1.686031174163188,-0.45146005776461423>,0.00859056726871202
    ,<0.33228017741311305,-1.6771431394356073,-0.4498811793201233>,0.008609050116966811
    ,<0.3206283844520983,-1.6662696735302212,-0.44841447077233354>,0.008626545756733304
    ,<0.3110533684279844,-1.6534995751647912,-0.447244675951429>,0.008643130939168025
    ,<0.30393328580750617,-1.6391873646073283,-0.4465051150164502>,0.00865887468788217
    ,<0.2993869937345764,-1.6238469127749158,-0.44626643121010073>,0.008673839244344611
    ,<0.2972035208548243,-1.6079978594782491,-0.4465287370981546>,0.008688080878257348
    ,<0.29689588041370696,-1.5920161721804644,-0.4472387672245244>,0.008701650584808223
    ,<0.29779496858821536,-1.5760779602678243,-0.44832846591199627>,0.008714594686749191
    ,<0.29912657680721555,-1.5601915368209822,-0.4497017423412501>,0.008726955356075762
    ,<0.3000839822065054,-1.5442949202328276,-0.45125800347876915>,0.008738771067525925
    ,<0.29989019047684357,-1.5283811146286914,-0.45291649558604985>,0.008750076994045604
    ,<0.29784952329485703,-1.5126026204154486,-0.4546218577156969>,0.008760905352682195
    ,<0.293416108987161,-1.4973240974469932,-0.45633623071950447>,0.008771285707989934
    ,<0.28628196339837814,-1.4831036895705023,-0.4580418319637704>,0.008781245238899917
    ,<0.2764486465110176,-1.4705971035957373,-0.45974434485539173>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
