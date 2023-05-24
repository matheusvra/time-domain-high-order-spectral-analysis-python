% Exemplo identificação da eq. logisitica
% (c) Luis Aguirre, 22/01/99
close all
clear
clc

load u.csv;
load y.csv;
l = length(y);

% Montar matriz de regressores
reg1=ones(l-3,1);		% constante
reg2=y(3:end-1);		% y(k-1)
reg3=y(2:end-2);		% y(k-2)
reg4=y(1:end-3);		% y(k-3)
reg5=u(3:end-1);		% u(k-1)
reg6=u(2:end-2);		% u(k-2)
reg7=u(1:end-3);		% u(k-3)
reg8=y(3:end-1).^2;		% y(k-1)²
reg9=y(2:end-2).^2;		% y(k-2)²
reg10=y(1:end-3).^2;    % y(k-3)²
reg11=u(3:end-1).^2;	% u(k-1)²
reg12=u(2:end-2).^2;	% u(k-2)²
reg13=u(1:end-3).^2;	% u(k-3)²
reg14=y(3:end-1).^3;	% y(k-1)³
reg15=y(2:end-2).^3;	% y(k-2)³
reg16=y(1:end-3).^3;	% y(k-3)³
reg17=u(3:end-1).^3;	% u(k-1)³
reg18=u(2:end-2).^3;	% u(k-2)³
reg19=u(1:end-3).^3;	% u(k-3)³


psi=[reg1 reg2 reg3 reg4 reg5 reg6 reg7 reg8 reg9 reg10 reg11 reg12 reg13 reg14 reg15 reg16 reg17 reg18 reg19];
vec=y(4:end);
Psi=[psi vec];

[A,err,piv]=myhouse(Psi,size(psi,2));
% numero de parametros no modelo final
np=5;
Psit=Psi(:,piv(1:np));
teta=(Psit'*Psit)\Psit'*vec
% numero de parametros no modelo final
np=3;
Psit=Psi(:,piv(1:np));
teta=(Psit'*Psit)\Psit'*vec



	