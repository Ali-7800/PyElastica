"""
Created on Dec. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from examples.OctopusArmControl.povray import POVRAYBase
from examples.OctopusArmControl.povray.muscles.muscle import POVRAYRingMuscle


class POVRAYTransverseMuscle(POVRAYBase, POVRAYRingMuscle):
    def __init__(self, **kwargs):
        POVRAYBase.__init__(self, **kwargs)
        POVRAYRingMuscle.__init__(self, **kwargs)
        self.color_string = self.to_color_string(self.muscle_color)
        self.muscle_label = "// transverse muscle data\n"
