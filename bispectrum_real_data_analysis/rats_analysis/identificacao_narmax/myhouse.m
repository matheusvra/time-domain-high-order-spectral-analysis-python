function [A,err,Piv]=myhouse(Psi,np);
% [A,err,Piv]=myhouse(Psi,np);
% Do livro Matrix Computations 2a Ed. pg 212
% Dada a matriz Psi (m,n), esta rotina acha Q de forma
% que Q'*Psi=V é triangular superior. A parte triangular
% superior de A é substituída pela parte triangular
% superior de V.
% 
% Assume-se que a última coluna de Psi é o vetor de
% observacoes a ser explicado, y(k).
% np é o número de regressores escolhidos para compor o modelo
% err é um vetor de np valores que contem as taxas de redução de erro
%	de cada um dos regressores escolhidos
% Piv é um vetor que contem os indices dos regressores escolhidos, ou
% 	seja são os indices das colunas usadas para pivotar a matriz
%	Psi np vezes.

[m,n]=size(Psi);
A=Psi;
yy=Psi(:,n)'*Psi(:,n);
piv=1:n-1;

for j=1:np	% Opera por colunas, ate o numero de termos final
   
      
  	% Determina err para demais regressores e volta a escolher
   % o de maior valor
   
   for k=j:n-1 % ate completar o numero de termos candidatos
   	c(k)=((A(j:m,k)'*A(j:m,n))^2)/((A(j:m,k)'*A(j:m,k))*yy);  % err do regressor k
	end;
   
   [ans aux]=max(c(j:n-1));
   jm=j+aux-1;
   err(j)=ans;
   aux=A(:,jm); % column of regressor with greatest err
	A(:,jm)=A(:,j);
   A(:,j)=aux;
   aux=piv(jm); % indice do regressor com maior err
	piv(jm)=piv(j);
	piv(j)=aux;
   
   
   x=A(j:m,j);
   % v=house(x)
	% Do livro Matrix Computations 2a Ed. pg 196
	% Dado um vetor x, volta-se um vetor v de tal forma
	% que (I-2vv'/v'v)x é zero à excecao do primeiro elemento

	nx=length(x);
	u=norm(x,2);
	v=x;
	if u ~= 0
	  b=x(1) + sign(x(1))*u;
	  v(2:nx) = v(2:nx)/b;
	end;
	v(1)=1;
   % fim house(x)

   a=A(j:m,j:n);
   
	% a=rowhouse(a,v)
	% Do livro Matrix Computations 2a Ed. pg 197
	% Dada uma matriz A (m,n), e um vetor de comprimento m, v, 
	% cujo primeiro elemento é 1, este algoritmo substitui
	% A por P*A onde P=I-2vv'/v'v

	b=-2/(v'*v);
	w=b*a'*v;
   a=a+v*w';
   % fim rowhouse(a,v)
   
   A(j:m,j:n)=a;   

end;
% fim myhouse(A)
Piv=piv(1:np);

