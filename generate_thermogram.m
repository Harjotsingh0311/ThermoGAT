%script to graph the themperature measures (*.csv) as a thermal image
%NOTE: colors of the generated thermogram may change respect to the *.png 
%of the database because of the used color palette but the temperature 
%measures remain the same despite the change of color used in the 
%representation.

clc
%loading the source file
I=load('CG001_M_L.csv');

%liminating the backgrounf pixels
v=find(I==0);
I(v)=NaN;


%plot a single foot thermogram
figure;
mesh(I)
colormap jet
set(gca, 'YDir','reverse');
c=colorbar;
c.Label.String = 'Color Temperature';
title('Thermogram');
xlabel('Columns')
ylabel('Rows')
zlabel('Temperature')
%view(0,90); %uncomment to obtain the frontal 2D view of the thermogam
rotate3d on
fprintf('\nRotate the object by setting the mouse pointer, clicking and dragging for 3D visualization.')




