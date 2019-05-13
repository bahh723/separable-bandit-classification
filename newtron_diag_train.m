function model = newtron_diag_train(X,Y,model)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

n = length(Y);   % number of training samples

if isfield(model,'gamma')==0
    model.gamma = .01;
end

if isfield(model,'alpha')==0
    model.alpha = 10;
end

if isfield(model,'D')==0
    model.D = 1;
end

if isfield(model,'beta')==0
    model.beta = 0.01;
end

if isfield(model,'iter')==0
    model.iter = 0;
    model.w = zeros(model.n_cla,size(X,1));
    model.errTot=zeros(numel(Y,1));
    model.numSV = zeros(numel(Y),1);
    model.aer = zeros(numel(Y),1);
    model.pred = zeros(model.n_cla,numel(Y));
    
    model.diagA=ones(size(X,1)*model.n_cla,1)/model.D;
    model.b = zeros(model.n_cla*size(X,1),1);
end

for i=1:n
    model.iter = model.iter+1;
        
    val_f = exp(model.alpha*model.w*X(:,i));
    val_f=val_f/sum(val_f);
    
    Yi = Y(i);
    
    Prob = (1-model.gamma)*val_f+model.gamma/model.n_cla;
    explore=(rand()<model.gamma);
    if explore==1
        y_tilde=randi(model.n_cla);
    else
        random_vect = (rand<cumsum(val_f));
        [dummy,y_tilde] = max(random_vect);
    end
    if model.iter>1
        model.errTot(model.iter) = model.errTot(model.iter-1)+(y_tilde~=Yi);
    else
        model.errTot(model.iter) + (y_tilde~=Yi);
    end
    model.aer(model.iter) = model.errTot(model.iter)/model.iter;
    model.pred(:,model.iter) = val_f;
    
    if y_tilde==Yi
        f1=(1-val_f(y_tilde))/(Prob(y_tilde)*model.n_cla);
        f1Xi=X(:,i)'*f1;
        tmp = repmat(f1Xi,[model.n_cla,1]); 
        tmp(y_tilde,:) = -f1Xi;
        kbeta=Prob(y_tilde)*model.beta;
    else
        f1=val_f(y_tilde)/(Prob(y_tilde)*model.n_cla);
        f1Xi=f1*X(:,i)';
        tmp = repmat(-f1Xi,[model.n_cla,1]); 
        tmp(y_tilde,:) = f1Xi;
        kbeta=model.beta;
    end
    tmprow=tmp(:);
    if explore==0
        model.b=model.b+(1-kbeta*(tmprow'*model.w(:)))*tmprow;
    else
        model.b=model.b+tmprow;
    end
    %model.b=model.b+(1-(kbeta*(explore==0))*tmp(:)'*model.w(:))*tmp(:);
    model.diagA=model.diagA+kbeta*tmprow.^2;
    w2=-model.b./model.diagA;
    f2=model.D/norm(w2);
    model.w=reshape(w2*f2,model.n_cla,size(X,1));
    
    if mod(i,model.step)==0
      fprintf('#%.0f AER:%5.2f\n', ...
            ceil(i/1000),model.aer(model.iter)*100);
            %fflush(stdout);
    end
end

