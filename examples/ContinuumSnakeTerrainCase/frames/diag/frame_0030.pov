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
    ,<0.06107845106717291,-2.9474988935584507,-0.22608348450854018>,0.0
    ,<0.05949573451454453,-2.9327866693082436,-0.23215965733125415>,0.001444405933878283
    ,<0.060056758790531184,-2.9180085837338208,-0.23825876001909496>,0.002733688514425582
    ,<0.06270004052033673,-2.9034976557482937,-0.24449948233212287>,0.0037941133653625076
    ,<0.06717433952865737,-2.8894890831824274,-0.2508770621505305>,0.0046307451971068355
    ,<0.07316860630902158,-2.8760811774027,-0.25729342622359136>,0.005283185474353696
    ,<0.08031436332109532,-2.863225642667678,-0.2636568854041388>,0.005794598874521764
    ,<0.08821449376550175,-2.8507348519944284,-0.2698492116693304>,0.00620058003411749
    ,<0.09648341543371676,-2.838354169016187,-0.2757716924726085>,0.006527801879788091
    ,<0.10474549564745796,-2.825807019020721,-0.2813400115430483>,0.006795619711330263
    ,<0.11262445608697252,-2.812837920768347,-0.28647804798793863>,0.007018006566011825
    ,<0.11974365433423473,-2.7992592888164753,-0.29112308068460024>,0.007205119848667835
    ,<0.1257453418942711,-2.78498590670067,-0.2952295868756459>,0.007364433711532417
    ,<0.13031612773281293,-2.7700476408125807,-0.29876872091315093>,0.0075015263935279105
    ,<0.13322335060416404,-2.754578845254674,-0.30173006514252987>,0.007620622272343326
    ,<0.13434866825315714,-2.738784629297989,-0.3041237706521731>,0.007724966207910139
    ,<0.1337074655224544,-2.722893475159205,-0.30598575363996006>,0.007817084460335388
    ,<0.13145821577113656,-2.707099561373534,-0.3073476792987816>,0.007898968749670325
    ,<0.12788841800395626,-2.691516084400605,-0.3082116803159486>,0.007972207813666372
    ,<0.12338407551121769,-2.6761570633048963,-0.3085858028560265>,0.008038082702723609
    ,<0.11839215943170674,-2.6609450944266317,-0.3085088206677674>,0.008097636716798745
    ,<0.11339522175907811,-2.645741354451118,-0.30804766808397055>,0.008151727381894005
    ,<0.10888953654071111,-2.6303953615891515,-0.30729774801005216>,0.008201065543276747
    ,<0.10536158070158766,-2.614803434417373,-0.3063837278748948>,0.008246245102718756
    ,<0.10325375860554413,-2.5989565091086515,-0.30546032171452986>,0.00828776588047385
    ,<0.10293218337473967,-2.5829648131439678,-0.3046926342970878>,0.008326051367736582
    ,<0.10464144839507618,-2.5670494142097406,-0.30424159116101834>,0.00836146264109268
    ,<0.10844793345511723,-2.5515019251069617,-0.3042735889963995>,0.008394309364827233
    ,<0.11425949042188922,-2.536606496894582,-0.30483040677262063>,0.008424858562469344
    ,<0.12182329134960568,-2.5225445650170015,-0.30581103404205306>,0.00845334166411343
    ,<0.1307377190141358,-2.5093254551723616,-0.30711352469643377>,0.008479960209706025
    ,<0.1405180878320306,-2.496760390992672,-0.30865396939228873>,0.008504890496255251
    ,<0.15064893900181442,-2.484500170000394,-0.3103775512298908>,0.008528287388947346
    ,<0.1606003769250227,-2.472117734207426,-0.31226619552752>,0.008550287465601714
    ,<0.16982554737269187,-2.459213572209261,-0.3143431113940131>,0.008571011625971648
    ,<0.1777724515588377,-2.445525779251872,-0.31667187459525475>,0.00859056726871202
    ,<0.18390621479069544,-2.4309936702640567,-0.3193537862513799>,0.008609050116966811
    ,<0.18780920903155268,-2.415781067143311,-0.3224299521033111>,0.008626545756733304
    ,<0.18924653334222424,-2.4002099239448365,-0.32584119245170884>,0.008643130939168025
    ,<0.18820454018911448,-2.3846586587587852,-0.329483690643858>,0.00865887468788217
    ,<0.18487422573024215,-2.369457540221533,-0.33322807200902416>,0.008673839244344611
    ,<0.1796176761430473,-2.3548083780912026,-0.33696314594231963>,0.008688080878257348
    ,<0.1729003811690928,-2.3407424737601223,-0.3405961457927745>,0.008701650584808223
    ,<0.16523035324094457,-2.3271274810294407,-0.34405483894517197>,0.008714594686749191
    ,<0.15712134474666425,-2.313713709597922,-0.34728891066274786>,0.008726955356075762
    ,<0.1490831855797527,-2.300198368021012,-0.350264118555894>,0.008738771067525925
    ,<0.14162523254371565,-2.286298269485159,-0.35296289592117397>,0.008750076994045604
    ,<0.1352563636601425,-2.271819227825354,-0.3553921090584982>,0.008760905352682195
    ,<0.13046955534111052,-2.2567070463189025,-0.35758458007592414>,0.008771285707989934
    ,<0.12770517412113522,-2.2410743807543563,-0.35960168569344253>,0.008781245238899917
    ,<0.12729841932691416,-2.225196128364613,-0.36153803143198976>,0.00879080897407521
        pigment{color rgb<20/255,14/255,14/255> transmit 0.000000
    }
    }
