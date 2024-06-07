clc; clear; close all;

super_room_temp = xlsread('super_122mm.xls',1);
super_150C = xlsread('super_122mm.xls',4);

fiber_room_temp = xlsread('single_fiber_133mm.xls',1);
fiber_150C = xlsread('single_fiber_133mm.xls',4);

super_room_temp_strain = super_room_temp(4:end, 2)/12.2;
super_room_temp_force = super_room_temp(4:end, 1);

super_150C_strain = super_150C(4:end, 2)/12.2;
super_150C_force = super_150C(4:end, 1);

fiber_room_temp_strain = fiber_room_temp(4:end, 2)/13.3;
fiber_room_temp_force = fiber_room_temp(4:end, 1);

fiber_150C_strain = fiber_150C(4:end, 2)/13.3;
fiber_150C_force = fiber_150C(4:end, 1);

plot(super_room_temp_strain*100, super_room_temp_force, 'k--','linewidth', 1)
hold on
plot(super_150C_strain*100, super_150C_force, 'k-', 'linewidth', 1)
plot(fiber_room_temp_strain*100, 3*fiber_room_temp_force, 'r--', 'linewidth', 1)
plot(fiber_150C_strain*100, 3*fiber_150C_force, 'r-', 'linewidth', 1)

xlim([-30 40])
ylim([-0.5 8])

legend('Super (25 ^{o}C)', 'Super (150 ^{o}C)', '3 Fibers (25 ^{o}C)', '3 Fibers (150 ^{o}C)')
legend box off

xlabel('Strain [%]')
ylabel('Force [N]')

set(gca, 'box', 'on', 'fontsize', 14)
