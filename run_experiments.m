clear all; 
close all;
rng(99990);
N = 50000; 
K = 3;    % now only support K=3 
gamma = 0.05;
W = [1,0; cos(pi*30/180), sin(pi*30/180); cos(-pi*30/180), sin(-pi*30/180)];

%%%%%%%%%%% generate samples %%%%%%%%%%%
X = ones(N,2);
X_ext = zeros(N,3);
Y = zeros(N,1); 
for i=1:N
    % pick angle 
    if rand() > 0.2
       clas = 1; 
    else
       clas = 2;
    end
    % pick length
    values = zeros(2,1);
    while (values(1)-values(2) < gamma || norm(X(i,:)) > 1)
       if clas==1
           angle = (30*rand()-15)*pi/180;
       else
           angle = (330*rand()+15)*pi/180;
       end
       len = 1;  % this is for strongly separable
                 % change to rand() when 
                 % generating weakly separable 
       X(i,:) = [len*cos(angle), len*sin(angle)];
       [values, idxs] = sort(W*X(i,:)', 'descend');
    end
    Y(i) = idxs(1);
    resid_norm = 1-norm(X(i,:));
    X_ext(i,1:2) = X(i,:);
    X_ext(i,3) = 1;
    if mod(i,10000)==0
        fprintf('generating %d-th sample\n', i);
    end
end

X_ext = X_ext/sqrt(2);

%%%%% scatter samples %%%%%
plot_idx = 1:100:N;
Nplot = length(plot_idx);
color_param = zeros(Nplot,3);
color_param(:,1) = (Y(plot_idx)==1); 
color_param(:,2) = (Y(plot_idx)==2); 
color_param(:,3) = (Y(plot_idx)==3); 
scatter(X(plot_idx,1), X(plot_idx,2), 30, color_param); 


%%%%%%%%% runs for different algorithms %%%%%%%%%
% (for code simplicity, we only show one run here)
see = randi(100000);
rng(see);
legend_list = {};

% 1. our algorithm with linear kernel
fprintf('--Our Algorithm (linear)--\n')
model_linova = model_init();
model_linova = ova_perceptron_train(X_ext', Y, model_linova);
errTot = model_linova.errTot;
figure(300); plot(plot_idx, errTot(plot_idx), '.-'); hold on;
legend_list = [legend_list, 'Our Algorithm (linear)'];

% 2. Banditron
fprintf('--Banditron--\n')
for g = [0.1, 0.05, 0.005, 0.0005]
   model_ban = model_init();
   model_ban.gamma = g;
   model_ban.n_cla = K;
   model_ban = banditron_multi_train(X_ext', Y, model_ban);
   errTot = model_ban.errTot;  
   figure(300); plot(plot_idx, errTot(plot_idx));
   legend_list = [legend_list, sprintf('Banditron (%0.4f)', g)];
end
 
% 3. our algorithm with rational kernel
fprintf('--Our Algorithm (rational)--\n')
hp.nu = 0.5; 
hp.type = 'rational';
model_ova = model_init(@compute_kernel,hp);
model_ova.n_cla = K; 
model_ova.maxSV = Inf;
model_ova = ova_k_perceptron_train(X_ext',Y, model_ova);
errTot = model_ova.errTot;
figure(300); plot(plot_idx, errTot(plot_idx), '--');
legend_list = [legend_list, 'Our Algorithm (rational kernel)'];

% 4. kernel banditron with rational kernel
fprintf('--Kernel Banditron--\n')
for g = [0.1, 0.05, 0.005, 0.0005]
   hp.nu = 0.5; 
   hp.type = 'rational';
   model_kban = model_init(@compute_kernel, hp);
   model_kban.gamma = g;
   model_kban.n_cla = K;
   model_kban.maxSV = Inf;
   model_kban = k_banditron_multi_train(X_ext', Y, model_kban);
   errTot = model_kban.errTot;  
   figure(300); plot(plot_idx, errTot(plot_idx));
   legend_list = [legend_list, sprintf('Kernel Banditron (%0.4f)', g)];
end

% % 5. SOBA 
fprintf('--SOBA--\n')
for g = [0.1, 0.05, 0.005, 0.0005]
   model_soba = model_init();
   model_soba.gamma = g;
   model_soba.n_cla = K;
   model_soba = soba_diag_multi_train(X_ext', Y, model_soba); 
   errTot = model_soba.errTot;  
   figure(300); plot(plot_idx, errTot(plot_idx));
   legend_list = [legend_list, sprintf('SOBA (%0.4f)', g)];
end

% 6. Newtron
fprintf('--Newtron--\n')
for g = [0.1, 0.05, 0.005, 0.0005]
   model_newt = model_init();
   model_newt.gamma = g;
   model_newt.n_cla = K;
   model_newt = newtron_diag_train(X_ext', Y, model_newt); 
   errTot = model_newt.errTot;  
   figure(300); plot(plot_idx, errTot(plot_idx));
   legend_list = [legend_list, sprintf('Newtron (%0.4f)', g)];
end

% 7. Perceptron (baseline)
fprintf('--Perceptron (baseline)--\n')
model_per = model_init();
model_per = perceptron_multi_train(X_ext', Y, model_per);
errTot = model_per.errTot;  
figure(300); plot(plot_idx, errTot(plot_idx));
legend_list = [legend_list, 'Perceptron'];

legend(legend_list);