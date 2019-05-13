function model = ova_perceptron_train(X,Y,model)
n = length(Y);   % number of training samples
if isfield(model,'n_cla')==0
    model.n_cla=max(Y);
end

%%%%%% the following implementation are similar to k_perceptron_train.m %%%%%%
if isfield(model,'iter')==0
    model.iter=0;
    model.errTot=zeros(numel(Y,1));
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(model.n_cla,numel(Y));
    model.w=zeros(model.n_cla,size(X,1));
end

if isfield(model, 'thres')==0
    model.thres= 0;
end

for i=1:n
    model.iter = model.iter+1;
    val_f = zeros(1, model.n_cla); 
    for j=1:model.n_cla
        val_f(j) = model.w(j,:)*X(:,i);
    end
    
    % if any of val_f is positive, pick the largest one 
    % else uniformly randomly pick one
    [maxval_f, maxidx] = max(val_f);
    if maxval_f >= model.thres
        Yhat = maxidx; 
    else 
        Yhat = unidrnd(model.n_cla);
    end
    
    if model.iter > 1
       model.errTot(model.iter) = model.errTot(model.iter-1) + (Yhat~=Y(i));
    else
       model.errTot(model.iter) = (Yhat~=Y(i)); 
    end
    model.aer(model.iter)=model.errTot(model.iter)/model.iter;
    model.pred(:,model.iter)=val_f;
    
    if Yhat==Y(i)   % guess correctly
        for j=1:model.n_cla
            if(j~= Yhat && val_f(j)>=0)    % can possibly use val_f(j)>-margin
                model.w(j,:) = model.w(j,:) - X(:,i)';
            end    
        end
        if val_f(Yhat) <= 0 % can possibly use f(Yhat) < gamma
            model.w(Yhat,:) = model.w(Yhat,:) + X(:,i)';
        end
    else   % guess incorrectly
        if val_f(Yhat) >= model.thres
            model.w(Yhat,:) = model.w(Yhat,:) - X(:,i)';
        end
    end
    
    if mod(i,model.step)==0
      fprintf('#%.0f \tAER:%5.2f\n', ...
            ceil(i/1000), model.aer(model.iter)*100);
    end    
end
%model.w
end