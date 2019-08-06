clc,clear
x = [0.97 0.029 0.001; 
    0.756 0.24 0.004; 
    0.77 0.12 0.12];
num = [3 3 3];
name = cell(1,3);
name{1} = 'TG4'; 
name{2} = 'TG5';
name{3} = 'TG6';
set(gca,'Position',[0.07 0.08 0.9 0.9]);
draw_cm(x, name, 3);