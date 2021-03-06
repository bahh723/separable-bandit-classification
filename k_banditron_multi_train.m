function model = k_banditron_multi_train(X,Y,model)
% BANDITRON_MULTI_TRAIN Banditron algorithm
%
%    MODEL = BANDITRON_MULTI_TRAIN(X,Y,MODEL) trains a multiclass
%    classifier according to the Banditron algorithm.
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

if isfield(model,'iter')==0
    model.iter = 0;
    %model.w = zeros(size(X,1),model.n_cla);
    model.beta_list= cell(model.n_cla, 1); 
    model.SV_list=cell(model.n_cla, 1); 
    model.errTot = zeros(numel(Y),1);
    model.numSV = zeros(numel(Y),1);
    model.aer = zeros(numel(Y),1);
    model.pred = zeros(model.n_cla,numel(Y));
    model.dim = size(X,1);
end

for j=1:model.n_cla
    model.beta_list{j} = [];
    model.numSV_list{j} = [];
    model.SV_list{j} = zeros(model.dim,0);
end

if isfield(model,'gamma')==0
    model.gamma = .01;
end

for i=1:n
    model.iter = model.iter+1;
    
    Xi=X(:,i);
    % val_f = model.w'*Xi;
    val_f = zeros(1, model.n_cla); 
    for j=1:model.n_cla
        if numel(model.SV_list{j})>0
           subK_f = feval(model.ker, model.SV_list{j}, X(:,i), model.kerparam); 
           val_f(j) = model.beta_list{j} * subK_f; 
        else
           val_f(j) = 0;
        end
    end
    
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

    if size(model.SV_list{y_hat},2) >= model.maxSV
         mn_idx = ceil(model.maxSV*rand); 
    elseif numel(model.SV_list{y_hat})==0
         mn_idx = 1; 
    else
         mn_idx = size(model.SV_list{y_hat},2)+1;
    end
    model.beta_list{y_hat}(mn_idx) = -1; 
    model.SV_list{y_hat}(:,mn_idx) = X(:,i); 
    
    % model.w(:,y_hat) = model.w(:,y_hat)-Xi;
    
    if y_tilde==Yi
        %model.w(:,Yi) = model.w(:,Yi)+Xi/Prob(y_tilde);
        if size(model.SV_list{Yi},2) >= model.maxSV
            mn_idx = ceil(model.maxSV*rand); 
        elseif numel(model.SV_list{Yi})==0
            mn_idx = 1; 
        else
            mn_idx = size(model.SV_list{Yi},2)+1;
        end
        model.beta_list{Yi}(mn_idx) = 1/Prob(y_tilde); 
        model.SV_list{Yi}(:,mn_idx) = X(:,i);
    end
    
    %model.numSV(model.iter) = numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f AER:%5.2f\n', ...
            ceil(i/1000),model.aer(model.iter)*100);
            %fflush(stdout);
    end
end
