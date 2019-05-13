function model = ova_k_perceptron_train(X,Y,model)
n = length(Y);   % number of training samples
if isfield(model,'n_cla')==0
    model.n_cla=max(Y);
end

%%%%%% the following implementation are similar to k_perceptron_train.m %%%%%%
if isfield(model,'iter')==0
    model.iter=0;
    model.errTot=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(model.n_cla,numel(Y));
    model.beta_list= cell(model.n_cla, 1);
    model.SV_list=cell(model.n_cla, 1); 
    model.dim = size(X,1);
end

for j=1:model.n_cla
    model.beta_list{j} = [];
    model.SV_list{j} = zeros(model.dim,0);
end

for i=1:n
    model.iter = model.iter+1;
    val_f = zeros(1, model.n_cla); 
    for j=1:model.n_cla
        if numel(model.SV_list{j})>0
           subK_f = feval(model.ker, model.SV_list{j}, X(:,i), model.kerparam); 
           val_f(j) = model.beta_list{j} * subK_f; 
        else
           val_f(j) = 0;
        end
    end
    
    % if any of val_f is positive, pick the largest one 
    % else uniformly randomly pick one
    [maxval_f, maxidx] = max(val_f);
    if maxval_f >= 0
        Yhat = maxidx; 
    else 
        Yhat = unidrnd(model.n_cla);
    end
    
    if(model.iter>1)
       model.errTot(model.iter) = model.errTot(model.iter-1) + (Yhat~=Y(i));
    else
       model.errTot(model.iter) = (Yhat~=Y(i)); 
    end
    model.aer(model.iter)=model.errTot(model.iter)/model.iter;
    model.pred(:,model.iter)=val_f;
    
    if Yhat==Y(i)   % guess correctly
        for j=1:model.n_cla
            if(j~= Yhat && val_f(j)>=0)    % can possibly use val_f(j)>-margin
                if size(model.SV_list{j},2) >= model.maxSV
                    mn_idx = ceil(model.maxSV*rand); 
                elseif numel(model.SV_list{j})==0
                    mn_idx = 1; 
                else
                    mn_idx = size(model.SV_list{j},2)+1;
                end
                model.beta_list{j}(mn_idx) = -1; 
                model.SV_list{j}(:,mn_idx) = X(:,i); 
            end    
        end
        if val_f(Yhat) <= 0 % can possibly use f(Yhat) < gamma
            if size(model.SV_list{Yhat},2) >= model.maxSV
                mn_idx = ceil(model.maxSV*rand); 
            elseif numel(model.SV_list{Yhat})==0
                mn_idx = 1; 
            else
                mn_idx = size(model.SV_list{Yhat},2)+1;
            end
            model.beta_list{Yhat}(mn_idx) = 1; 
            model.SV_list{Yhat}(:,mn_idx) = X(:,i); 
        end
    else   % guess incorrectly
        if size(model.SV_list{Yhat},2) >= model.maxSV
            mn_idx = ceil(model.maxSV*rand); 
        elseif numel(model.SV_list{Yhat})==0
            mn_idx = 1; 
        else
            mn_idx = size(model.SV_list{Yhat},2)+1;
        end
        if val_f(Yhat) >= 0
            model.beta_list{Yhat}(mn_idx) = -1; 
            model.SV_list{Yhat}(:,mn_idx) = X(:,i); 
        end
    end
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:\tAER:%5.2f\n', ...
            ceil(i/1000),model.aer(model.iter)*100);
    end    
end
end