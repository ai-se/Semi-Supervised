function result = run_code(data_X,data_y)

setpaths

data_y = reshape(data_y, [length(data_y),1]);

X=data_X;
Y=data_y;


% generating default options
options=make_options('gamma_I',1,'gamma_A',1e-5,'NN',6,'KernelParam',0.35);
options.Verbose=1;
options.UseBias=1;
options.UseHinge=1;
options.LaplacianNormalize=0;
options.NewtonLineSearch=0;

% creating the 'data' structure
data.X=X;
data.Y=Y;
disp(size(data.Y))


fprintf('Computing Gram matrix and Laplacian...\n\n');
data.K=calckernel_1(options,X,X);
data.L=laplacian_1(options,X);

disp(size(data.X))
disp(size(data.Y))
disp(size(data.K))
disp(size(data.L))

% training the classifier
fprintf('Training LapSVM in the primal with Newton''s method...\n');
classifier=lapsvmp(options,data);

% computing error rate
fprintf('It took %f seconds.\n',classifier.traintime);
out=sign(data.K(:,classifier.svs)*classifier.alpha+classifier.b);
er=100*(length(data.Y)-nnz(out==Y))/length(data.Y);
fprintf('Error rate=%.1f\n\n',er);

% training the classifier
fprintf('Training LapSVM in the primal with early stopped PCG...\n');
options.Cg=1; % PCG
options.MaxIter=1000; % upper bound
options.CgStopType=1; % 'stability' early stop
options.CgStopParam=0.015; % tolerance: 1.5%
options.CgStopIter=3; % check stability every 3 iterations
classifier=lapsvmp(options,data);
fprintf('It took %f seconds.\n',classifier.traintime);

% computing error rate
out=sign(data.K(:,classifier.svs)*classifier.alpha+classifier.b);
er=100*(length(data.Y)-nnz(out==Y))/length(data.Y);
fprintf('Error rate=%.1f\n\n',er);
result = out;