clc; clear; close all;

nogap_25C = xlsread('04222024_nogap.xls',1);
nogap_90C = xlsread('04222024_nogap.xls',4);

smallgap_25C = xlsread('04222024_smallgap.xls',1);
smallgap_90C = xlsread('04222024_smallgap.xls',4);

largegap_25C = xlsread('04222024_largegap.xls',1);
largegap_90C = xlsread('04222024_largegap.xls',4);


nogap_25C_strain = nogap_25C(4:end, 2)/17.3;
nogap_25C_force = nogap_25C(4:end, 1);

nogap_90C_strain = nogap_90C(4:end, 2)/17.3;
nogap_90C_force = nogap_90C(4:end, 1);

smallgap_25C_strain = smallgap_25C(4:end, 2)/10.3;
smallgap_25C_force = smallgap_25C(4:end, 1);

smallgap_90C_strain = smallgap_90C(4:end, 2)/10.3;
smallgap_90C_force = smallgap_90C(4:end, 1);

largegap_25C_strain = largegap_25C(4:end, 2)/13.6;
largegap_25C_force = largegap_25C(4:end, 1);

largegap_90C_strain = largegap_90C(4:end, 2)/13.6;
largegap_90C_force = largegap_90C(4:end, 1);

%%
figure(1)
plot(nogap_25C_strain*100, nogap_25C_force, 'b-','linewidth', 1)
hold on
plot(nogap_90C_strain*100, nogap_90C_force, 'r-', 'linewidth', 1)

xline(0, ':')
xlim([-20 25])
ylim([0 1])

legend('22 ^{o}C', '85 ^{o}C', 'location', 'southeast')
legend box off

title('No gap')
xlabel('Strain [%]')
ylabel('Force [N]')

set(gca, 'box', 'on', 'fontsize', 14)

%%
% figure(2)
% plot(smallgap_25C_strain*100, smallgap_25C_force, 'b-','linewidth', 1)
% hold on
% plot(smallgap_90C_strain*100, smallgap_90C_force, 'r-', 'linewidth', 1)
%
% xline(0, ':')
% xlim([-20 40])
% ylim([0 1])
%
% legend('22 ^{o}C', '85 ^{o}C', 'location', 'southeast')
% legend box off
%
% title('Small gap')
% xlabel('Strain [%]')
% ylabel('Force [N]')
%
% set(gca, 'box', 'on', 'fontsize', 14)

%%
figure(3)
plot(largegap_25C_strain*100, largegap_25C_force, 'b-','linewidth', 1)
hold on
plot(largegap_90C_strain*100, largegap_90C_force, 'r-', 'linewidth', 1)

xline(0, ':')
xlim([-20 25])
ylim([0 1])

legend('22 ^{o}C', '85 ^{o}C', 'location', 'southeast')
legend box off

title('Large gap')
xlabel('Strain [%]')
ylabel('Force [N]')

set(gca, 'box', 'on', 'fontsize', 14)

%%
figure(4)
plot(nogap_25C_strain*100, nogap_25C_force, 'b--','linewidth', 1)
hold on
plot(nogap_90C_strain*100, nogap_90C_force, 'r--', 'linewidth', 1)
plot(largegap_25C_strain*100, largegap_25C_force, 'b-','linewidth', 1)
plot(largegap_90C_strain*100, largegap_90C_force, 'r-', 'linewidth', 1)

xline(0, ':')
xlim([-20 25])
ylim([0 1])

legend('22 ^{o}C no gap', '85 ^{o}C no gap', '22 ^{o}C large gap', '85 ^{o}C large gap', 'location', 'southeast')
legend box off

% title('Large gap')
xlabel('Strain [%]')
ylabel('Force [N]')

set(gca, 'box', 'on', 'fontsize', 14)
