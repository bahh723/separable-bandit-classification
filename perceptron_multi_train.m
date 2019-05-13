function model = perceptron_multi_train(X,Y,model)
% PERCEPTRON_MULTI_TRAIN Perceptron multiclass algorithm
%
%    MODEL = PERCEPTRON_MULTI_TRAIN(X,Y,MODEL) trains a multiclass classifier
%    according to the Perceptron algorithm.
%
%    Additional parameters:
%    none
%
%   References:
%
%    This file is part of the DOGMA library for MATLAB.
%    Copyright (C) 2009-2012, Francesco Orabona
%    If you use this software in a paper you have to cite
%     - Francesco Orabona. (2009)
%       DOGMA: a MATLAB toolbox for Online Learning
%       Software available at http://dogma.sourceforge.net
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    Contact the author: francesco [at] orabona.com

n = length(Y);   % number of training samples

if isfield(model,'n_cla')==0
    model.n_cla=max(Y);
end
    
if isfield(model,'iter')==0
    model.iter=0;
    model.w=zeros(model.n_cla,size(X,1));
    model.w2=zeros(model.n_cla,size(X,1));
    model.errTot=zeros(numel(Y,1));
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(model.n_cla,numel(Y));
end

for i=1:n
    model.iter=model.iter+1;
    
    val_f=model.w*X(:,i);

    Yi=Y(i);
    
    tmp=val_f; tmp(Yi)=-inf;
    [mx_val,idx_mx_val]=max(tmp);

    if model.iter>1
        model.errTot(model.iter) = model.errTot(model.iter-1)+(val_f(Yi)<=mx_val);
    else
        model.errTot(model.iter) + (val_f(Yi)<=mx_val);
    end
    model.aer(model.iter) = model.errTot(model.iter)/model.iter;
    model.pred(:,model.iter)=val_f;
    
    if val_f(Yi)<=mx_val
        model.w(Yi,:)=model.w(Yi,:)+X(:,i)';
        model.w(idx_mx_val,:)=model.w(idx_mx_val,:)-X(:,i)';
    end

    %model.w2=model.w2+model.w;

    %model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f\tAER:%5.2f\n', ...
            ceil(i/1000),model.aer(model.iter)*100);
            %fflush(stdout);
    end
end
