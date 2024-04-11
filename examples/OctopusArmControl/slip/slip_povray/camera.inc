//--------------------------------------------------
#version 3.6; // 3.7;
global_settings{ assumed_gamma 1.0 }
#default{ finish{ ambient 0.1 diffuse 0.9 }}

//--------------------------------------------------
#include "colors.inc"
#include "textures.inc"
#include "glass.inc"
#include "metals.inc"
#include "golds.inc"
#include "stones.inc"
#include "woods.inc"
#include "shapes.inc"
#include "shapes2.inc"
#include "functions.inc"
#include "math.inc"
#include "transforms.inc"

//--------------------------------------------------
#declare Camera_Position  = <1.500000, -5.000000, 0.600000>;
#declare Camera_Look_At   = <1.500000, 0.000000, 0.600000>;
#declare Camera_Sky       = <0.000000, 0.000000, 1.000000>;
#declare Camera_Angle     = 40.000000;
#declare Camera_Rotate    = <0.000000, 0.000000, 0.000000>;
#declare Camera_Translate = <0.000000, 0.000000, 0.000000>;
camera{ location  Camera_Position
        right     -x*image_width/image_height
        sky       Camera_Sky
        angle     Camera_Angle
        look_at   Camera_Look_At
        rotate    Camera_Rotate
        translate Camera_Translate
}

#declare Sun_Position = <-1500.000000, -1000.000000, 2000.000000>;
light_source{ Sun_Position color White}                      // sun light
light_source{ Camera_Position color rgb<0.9, 0.9, 1.0>*0.1}  // flash light
background { color White }
