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
    ,<0.23394111412105675,-2.0797328781235587,-0.34709922005763394>,0.0
    ,<0.2480815331125055,-2.0739395176047823,-0.3518352098307725>,0.001444405933878283
    ,<0.2607942489048307,-2.0654906764573138,-0.35662353506475086>,0.002733688514425582
    ,<0.27157183328625684,-2.0547374934885467,-0.3615338680017688>,0.0037941133653625076
    ,<0.2801349725238884,-2.042234786197127,-0.3666488351722015>,0.0046307451971068355
    ,<0.2864364662631694,-2.0285666671661495,-0.37204990232806373>,0.005283185474353696
    ,<0.29066022951199905,-2.014239710293379,-0.37781592407357556>,0.005794598874521764
    ,<0.2932674142105329,-1.9996130423184468,-0.38383912364409334>,0.00620058003411749
    ,<0.2948637473808967,-1.9848439655866568,-0.38986497535375586>,0.006527801879788091
    ,<0.2960829905410203,-1.969970230389754,-0.39571514783650036>,0.006795619711330263
    ,<0.29755019591897786,-1.9550032463636584,-0.4012593756128595>,0.007018006566011825
    ,<0.29985028044600287,-1.9400037679245263,-0.40641531860508945>,0.007205119848667835
    ,<0.3034883249048758,-1.9251320597866588,-0.4111495083931439>,0.007364433711532417
    ,<0.3088398039589746,-1.9106612459746626,-0.4154766886118017>,0.0075015263935279105
    ,<0.3161042549450698,-1.8969470091839582,-0.41945950928102727>,0.007620622272343326
    ,<0.3252594135996399,-1.8843296735338473,-0.42315414765620457>,0.007724966207910139
    ,<0.3360879771035813,-1.873037757841375,-0.42659846736607826>,0.007817084460335388
    ,<0.34824976647971456,-1.8631316500637933,-0.42984539838649544>,0.007898968749670325
    ,<0.36135235136103927,-1.8544575344154497,-0.43294935650084354>,0.007972207813666372
    ,<0.37500871218699794,-1.8466469612703078,-0.4359524983426564>,0.008038082702723609
    ,<0.3888575851575052,-1.8391538297383234,-0.4388787400222597>,0.008097636716798745
    ,<0.40253318641614105,-1.8313242483361507,-0.4417343895971939>,0.008151727381894005
    ,<0.4155981299485981,-1.8224863647279563,-0.4445050219384426>,0.008201065543276747
    ,<0.42746997414202276,-1.8120691069842896,-0.4471542374131007>,0.008246245102718756
    ,<0.43739400892870484,-1.799744623577056,-0.44962539429145154>,0.00828776588047385
    ,<0.4445065007922252,-1.7855687213891371,-0.4518472713819656>,0.008326051367736582
    ,<0.4480049651673423,-1.7700571085644599,-0.4537468067793995>,0.00836146264109268
    ,<0.4473734440472508,-1.7541285015405244,-0.4552567886180114>,0.008394309364827233
    ,<0.4425626407690251,-1.7388962674702384,-0.45632525644341176>,0.008424858562469344
    ,<0.4340304398196964,-1.7253676154461506,-0.45693513515897555>,0.00845334166411343
    ,<0.4226103451455518,-1.7141573443933,-0.4570963473765885>,0.008479960209706025
    ,<0.40926256729324506,-1.7053353684808625,-0.45681993620255534>,0.008504890496255251
    ,<0.3948271901289063,-1.6984661015387323,-0.4561280924852397>,0.008528287388947346
    ,<0.3799099479540349,-1.6927747784692815,-0.4550656042658894>,0.008550287465601714
    ,<0.36492300073717215,-1.6873366249547186,-0.45369604734497204>,0.008571011625971648
    ,<0.3502094769986907,-1.6812485542974223,-0.4521096208674427>,0.00859056726871202
    ,<0.33615884503400956,-1.673773053898595,-0.45043945696743276>,0.008609050116966811
    ,<0.3232515730456616,-1.6644445638473917,-0.4488588110105108>,0.008626545756733304
    ,<0.31201306534894674,-1.6531263976378894,-0.4475564616715264>,0.008643130939168025
    ,<0.3029309840555855,-1.6399815725445033,-0.44666863390142314>,0.00865887468788217
    ,<0.2963066996182452,-1.6254219457992067,-0.44624503404776816>,0.008673839244344611
    ,<0.2921310610053053,-1.6099759185930727,-0.4462971296951556>,0.008688080878257348
    ,<0.29008793421875784,-1.5941152186070011,-0.4468108946644945>,0.008701650584808223
    ,<0.2896173610823857,-1.5781482535301654,-0.4477226096740713>,0.008714594686749191
    ,<0.29001058829332377,-1.5621993763796473,-0.4489405536953412>,0.008726955356075762
    ,<0.2904964453170677,-1.5462715086955856,-0.45038642681535745>,0.008738771067525925
    ,<0.2902951149631373,-1.5303512395594352,-0.45198016940005104>,0.008750076994045604
    ,<0.28866345438864227,-1.5145211972511683,-0.4536474226150679>,0.008760905352682195
    ,<0.28496472572788883,-1.499046538108556,-0.4553444168182348>,0.008771285707989934
    ,<0.2787515924661084,-1.484401418996467,-0.45705706248533584>,0.008781245238899917
    ,<0.26984006252702236,-1.4712247629387605,-0.45877668960424706>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
