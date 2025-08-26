%
%  integrate_function.m  ver 1.1  October 13, 2012
%
%  by Tom Irvine
%
function[v]=integrate_function(y,dt)
%
v=dt*cumtrapz(y);