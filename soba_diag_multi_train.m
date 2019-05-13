function model = soba_diag_multi_train(X,Y,model)
% SOBA_MULTI_TRAIN Banditron algorithm
%
%    MODEL = SOBA_MULTI_TRAIN(X,Y,MODEL) trains a multiclass
%    classifier using SOBA.
%
%    Additional parameters:
%    - model.n_cla is the number of classes.
%    - model.gamma is the parameter that controls the trade-off between
%      exploration and exploitation.
%      Default value is 0.01.
%
%   References:
%     - Kakade, S. M., Shalev-Shwartz, S., & Tewari, A. (2008).
%       Efficient bandit algorithms for online multiclass prediction.
%       Proceedings of the 25th International Conference on Machine
%       Learning (pp. 440??47).
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
%rand('state',0);

if isfield(model,'gamma')==0
    model.gamma = .01;
end

if isfield(model,'a')==0
    %model.a = 1/model.gamma;
    %model.a = sqrt(n);
    model.a = 1;
end

if isfield(model,'iter')==0
    model.iter = 0;
    model.w = zeros(model.n_cla,size(X,1));
    model.errTot=zeros(numel(Y,1));
    model.numSV = zeros(numel(Y),1);
    model.aer = zeros(numel(Y),1);
    model.pred = zeros(model.n_cla,numel(Y));
    
    %model.invA=eye(size(X,1)*model.n_cla)/model.a;
    model.A=ones(size(X,1)*model.n_cla,1)*model.a;
    model.theta = zeros(model.n_cla,size(X,1));
    model.sum_q=0;
    model.gamma_rate = zeros(numel(Y),1);
    model.mupd=0;
    model.sum_m=0;
end

for i=1:n
    model.iter = model.iter+1;
    
    %model.gamma=min(sqrt(model.n_cla*(1+model.sum_q)/model.iter),1);
    %model.gamma=0.01;
    %model.gamma=1/sqrt(model.iter);
    model.gamma_rate(model.iter)=model.gamma;
    
    val_f = model.w*X(:,i);
    
    Yi = Y(i);
    
    [mx_f,y_hat] = max(val_f);
    Prob = zeros(1,model.n_cla)+model.gamma/model.n_cla;
    Prob(y_hat) = Prob(y_hat)+1-model.gamma;
    random_vect = (rand<cumsum(Prob));
    [dummy,y_tilde] = max(random_vect);
    
    if model.iter>1
        model.errTot(model.iter) = model.errTot(model.iter-1)+(y_tilde~=Yi);
    else
        model.errTot(model.iter) + (y_tilde~=Yi);
    end
    model.aer(model.iter) = model.errTot(model.iter)/model.iter;
    model.pred(:,model.iter) = val_f;

    if y_tilde==Yi
      if y_tilde~=y_hat
        tmp = zeros(model.n_cla,size(X,1));
        tmp(y_hat,:) = -1/Prob(y_tilde)*X(:,i)';
        tmp(Yi,:) = 1/Prob(y_tilde)*X(:,i)';
        model.theta=model.theta+tmp;
        res=sqrt(Prob(y_tilde))*tmp(:)'*(tmp(:)./model.A);

        diff=val_f(y_hat)-mx_f; 
        m=(diff^2/(1+res)+2*diff/(1+res))/Prob(y_tilde);
        model.sum_q=model.sum_q+res/(1+res);

        model.A=model.A+Prob(y_tilde)*tmp(:).^2;
        model.w=reshape(model.theta(:)./model.A,model.n_cla,size(X,1));
        model.sum_q=model.sum_q+res/(1+res);
      else
        val_f(y_tilde)=-inf;
        [mx_f2,y_hat2] = max(val_f);
        tmp = zeros(model.n_cla,size(X,1));
        tmp(y_hat2,:) = -1/Prob(y_tilde)*X(:,i)';
        tmp(Yi,:) = 1/Prob(y_tilde)*X(:,i)';
        res=Prob(y_tilde)*(tmp(:)'*(tmp(:)./model.A));
        diff=mx_f2-mx_f; 
        m=(diff^2/(1+res)+2*diff/(1+res))/Prob(y_tilde);
        if model.sum_m+m>=0
        %if m>=0
          model.theta=model.theta+tmp;
          model.A=model.A+Prob(y_tilde)*tmp(:).^2;
          model.w=reshape(model.theta(:)./model.A,model.n_cla,size(X,1));
          model.sum_m=model.sum_m+m;
          model.sum_q=model.sum_q+res/(1+res);
          model.mupd=model.mupd+1;
        end
      end
    end
    
    %model.numSV(model.iter) = numel(model.S);

    if mod(i,model.step)==0
      fprintf('#%.0f AER:%5.2f Log term:%f MarginUpdates:%.0f\n', ...
            ceil(i/1000),model.aer(model.iter)*100, model.sum_q, model.mupd);
            %fflush(stdout);
    end
end
