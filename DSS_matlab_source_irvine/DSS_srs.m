%
%  DSS_srs.m  ver 1.0  October 5, 2012
%
function[xmin,xmax]=DSS_srs(~,last,~,freq,damp,dt,yy)
%
%******************************** initialize arrays ******************/
%
clear xmax;
clear xmin;
xmax=zeros(last,1);
xmin=zeros(last,1);
%
%*********************** initialize filter coefficients ***************/
%
[a1,a2,b1,b2,b3,~,~,~,~,~]=srs_coefficients(freq,damp,dt);   
%
for j=1:last
       
        clear resp;
        clear forward;
        clear back;
       
        forward=[ b1(j),  b2(j),  b3(j) ];    
        back   =[     1, -a1(j), -a2(j) ];    
%    
        resp=filter(forward,back,yy);
%
        xmax(j)= abs(max(resp));
        xmin(j)= abs(min(resp));
end     
%  
for j=1:last
    if(abs(xmin(j)) <1.0e-90)
			out1=sprintf('  warning: abs(xmin[%ld])=%8.4g ',j,xmin(j));
            disp(out1);
            break;
    end
    if(abs(xmax(j)) <1.0e-90)
			out1=sprintf('  warning: abs(xmax[%ld])=%8.4g ',j,xmax(j));
            disp(out1);
            break;
    end
end